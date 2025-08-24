# 导入必要的库
import json
import re
import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import networkx as nx

# PyTorch相关库
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout

# PyTorch Geometric相关库
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
# --- 【修改】重新导入 Set2Set ---
from torch_geometric.nn import GATConv, Set2Set
from torch_geometric.utils import to_networkx, degree

# 句子嵌入模型
from sentence_transformers import SentenceTransformer

# ==============================================================================
# 1. 数据预处理模块 (已修改)
# ==============================================================================

# parse_mermaid 函数保持不变
def parse_mermaid(mermaid_code):
    """
    解析Mermaid流程图代码，提取节点和边的信息（支持链式边和多种格式）
    """
    nodes = {}
    edges = []
    
    if not isinstance(mermaid_code, str):
        return nodes, edges

    lines = mermaid_code.strip().split('\n')
    
    # 更强大的节点匹配模式
    node_patterns = [
        re.compile(r'([A-Za-z0-9_]+)\s*\[\s*["\']([^"\']*)["\'\s]*\]'),  # A["text"] 或 A['text']
        re.compile(r'([A-Za-z0-9_]+)\s*\[\s*([^\[\]]*)\s*\]'),          # A[text]
        re.compile(r'([A-Za-z0-9_]+)\s*\(\s*["\']([^"\']*)["\'\s]*\)'),  # A("text") 或 A('text')
        re.compile(r'([A-Za-z0-9_]+)\s*\(\s*([^\(\)]*)\s*\)'),          # A(text)
    ]
    
    # 链式边匹配模式 - 支持一行多个连续的箭头
    chain_edge_pattern = re.compile(
        r'([A-Za-z0-9_]+)(?:\s*\[[^\[\]]*\]|\s*\([^\(\)]*\))?\s*--+>\s*([A-Za-z0-9_]+)(?:\s*\[[^\[\]]*\]|\s*\([^\(\)]*\))?\s*(?:--+>\s*([A-Za-z0-9_]+)(?:\s*\[[^\[\]]*\]|\s*\([^\(\)]*\))?)*'
    )
    
    # 单个边匹配模式
    single_edge_pattern = re.compile(
        r'([A-Za-z0-9_]+)(?:\s*\[[^\[\]]*\]|\s*\([^\(\)]*\))?\s*--+>\s*([A-Za-z0-9_]+)(?:\s*\[[^\[\]]*\]|\s*\([^\(\)]*\))?'
    )
    
    all_node_ids = set()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%%') or line.lower().startswith('graph'):
            continue
        
        # 1. 先提取所有节点定义
        for pattern in node_patterns:
            matches = pattern.findall(line)
            for match in matches:
                node_id = match[0].strip()
                text = match[1].strip() if len(match) > 1 and match[1] else node_id
                nodes[node_id] = text
                all_node_ids.add(node_id)
        
        # 2. 处理链式边（一行多个箭头）
        # 移除分号并处理链式边
        line_clean = line.rstrip(';').strip()
        
        # 查找所有箭头分割的部分
        arrow_split = re.split(r'\s*--+>\s*', line_clean)
        
        if len(arrow_split) > 1:
            # 提取每个部分的节点ID（去除方括号和圆括号内容）
            chain_nodes = []
            for part in arrow_split:
                # 提取节点ID，忽略节点定义中的文本部分
                node_match = re.match(r'([A-Za-z0-9_]+)', part.strip())
                if node_match:
                    node_id = node_match.group(1)
                    chain_nodes.append(node_id)
                    all_node_ids.add(node_id)
                    
                    # 同时提取节点文本（如果有的话）
                    for pattern in node_patterns:
                        text_match = pattern.search(part.strip())
                        if text_match and text_match.group(1) == node_id:
                            text = text_match.group(2) if len(text_match.groups()) > 1 and text_match.group(2) else node_id
                            nodes[node_id] = text.strip()
                            break
            
            # 构建链式边
            for i in range(len(chain_nodes) - 1):
                edges.append((chain_nodes[i], chain_nodes[i + 1]))
        
        # 3. 处理单个边（作为备用，防止遗漏）
        else:
            matches = single_edge_pattern.findall(line)
            for match in matches:
                if len(match) >= 2:
                    source_id = match[0].strip()
                    target_id = match[1].strip()
                    if source_id and target_id:
                        edges.append((source_id, target_id))
                        all_node_ids.add(source_id)
                        all_node_ids.add(target_id)
    
    # 4. 确保所有节点都有文本描述
    for node_id in all_node_ids:
        if node_id not in nodes:
            nodes[node_id] = node_id
    
    # 5. 兜底处理：如果仍然没有解析到任何内容
    if not nodes and not edges:
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.lower().startswith('graph') and not line.startswith('%%'):
                node_id = f"node_{i}"
                nodes[node_id] = line
                if i > 0:
                    edges.append((f"node_{i-1}", node_id))

    return nodes, edges

class MermaidDataset(InMemoryDataset):
    def __init__(self, root, raw_dataframe, transform=None, pre_transform=None):
        self.raw_dataframe = raw_dataframe
        super(MermaidDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        # --- 【新增】加载归一化参数 ---
        self.norm_params = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ['new_data.csv'] 

    @property
    def processed_file_names(self):
        # --- 【修改】增加一个文件用于存储归一化参数 ---
        return ['processed_data_normalized.pt', 'norm_params.pt']

    def download(self):
        pass
    
    def process(self):
        """
        核心数据处理方法（已修复字符串转义问题和设备一致性问题）
        """
        print("初始化句子嵌入模型...")
        device_for_embedding = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # --- 【关键修改】直接指定完整的本地缓存路径 ---
        local_model_path = '/data2/huangwenbo/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d'
        
        print(f"正在从本地缓存加载模型：{local_model_path}")
        
        try:
            embedding_model = SentenceTransformer(
                local_model_path, 
                device=device_for_embedding
            )
        except Exception as e:
            print(f"错误：从本地路径 '{local_model_path}' 加载模型失败。请检查路径是否正确以及文件是否完整。")
            print(f"具体错误: {e}")
            exit()
        
        # PCA维度设为16
        pca = PCA(n_components=16)
        
        data_list = []
        raw_graph_features_list = []
        
        print("将CSV原始数据处理为图结构 (使用最终版特征)...")
        for index, row in tqdm(self.raw_dataframe.iterrows(), total=self.raw_dataframe.shape[0]):
            try:
                mermaid_code_raw = row['Mermaid 代码']
                
                # 【关键修复】处理转义字符，与gnn_8_19.py保持一致
                mermaid_code = mermaid_code_raw.replace('\\n', '\n').replace('\\"', '"')
                
                score_values = [
                    row['Structure_Logic'],
                    row['Content_Completeness'],
                    row['Hierarchy_Clarity'],
                    row['Code_Syntax'],
                ]
                
                nodes, edges = parse_mermaid(mermaid_code)
                
                if not nodes:
                    print(f"\n警告：无法解析行索引 {index} 的节点。跳过该数据。")
                    continue

                node_ids = list(nodes.keys())
                node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
                num_nodes = len(node_ids)

                # --- 文本嵌入与PCA降维 (16维) ---
                node_texts = [nodes[node_id] for node_id in node_ids]
                full_node_features_tensor = embedding_model.encode(
                    node_texts, 
                    convert_to_tensor=True, 
                    show_progress_bar=False,
                    device=device_for_embedding
                )
                
                # 【关键修复】确保文本特征在CPU上进行后续处理
                full_node_features_cpu = full_node_features_tensor.cpu()
                
                if num_nodes > 16:
                    node_features_reduced_np = pca.fit_transform(full_node_features_cpu.numpy())
                    text_features = torch.tensor(node_features_reduced_np, dtype=torch.float)
                else:
                    text_features = full_node_features_cpu[:, :16]
                    if text_features.shape[1] < 16:
                        padding = torch.zeros(num_nodes, 16 - text_features.shape[1])
                        text_features = torch.cat([text_features, padding], dim=1)
                
                if edges:
                    valid_edges = [edge for edge in edges if edge[0] in node_id_to_idx and edge[1] in node_id_to_idx]
                    if not valid_edges:
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                    else:
                        source_indices = [node_id_to_idx[s] for s, t in valid_edges]
                        target_indices = [node_id_to_idx[t] for s, t in valid_edges]
                        edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                
                max_scores = [35, 35, 20, 10]
                normalized_scores = [float(s) / max_s for s, max_s in zip(score_values, max_scores)]
                y = torch.tensor(normalized_scores, dtype=torch.float).unsqueeze(0)
                
                temp_data_for_nx = Data(edge_index=edge_index, num_nodes=num_nodes)
                g_nx = to_networkx(temp_data_for_nx, to_undirected=False)

                # --- 节点级别结构特征 (共5个) ---
                if num_nodes > 0:
                    # 【关键修复】所有结构特征都在CPU上计算
                    in_degree = degree(edge_index[1], num_nodes=num_nodes).view(-1, 1)
                    out_degree = degree(edge_index[0], num_nodes=num_nodes).view(-1, 1)

                    node_depths = torch.zeros(num_nodes, 1)
                    root_nodes = [n for n, d in g_nx.in_degree() if d == 0]
                    if not root_nodes: 
                        root_nodes = [0]
                    for root in root_nodes:
                        try:
                            depths = nx.shortest_path_length(g_nx, source=root)
                            for node, d in depths.items():
                                if d > node_depths[node][0]:
                                    node_depths[node][0] = d
                        except nx.NetworkXNoPath: 
                            pass
                    
                    betweenness = torch.tensor(list(nx.betweenness_centrality(g_nx).values()), dtype=torch.float).view(-1, 1)
                    closeness = torch.tensor(list(nx.closeness_centrality(g_nx).values()), dtype=torch.float).view(-1, 1)

                    structural_node_features = torch.cat([
                        in_degree / num_nodes, out_degree / num_nodes, node_depths / num_nodes, betweenness, closeness
                    ], dim=1)
                    
                    # 【关键修复】确保文本特征和结构特征都在CPU上
                    final_node_features = torch.cat([text_features, structural_node_features], dim=1)
                else:
                    final_node_features = torch.empty((0, 16 + 5), dtype=torch.float)

                # 【关键修复】确保所有张量都在CPU上
                graph_data = Data(
                    x=final_node_features.float().cpu(), 
                    edge_index=edge_index.cpu(), 
                    y=y.cpu()
                )
                
                # --- 【修改】计算全局图级别特征 (共9个)，暂不归一化 ---
                num_edges = g_nx.number_of_edges()
                log_num_nodes = np.log1p(num_nodes)
                degree_centrality = list(nx.degree_centrality(g_nx).values())
                degree_std = np.std(degree_centrality) if degree_centrality else 0.0
                num_leaves = sum(1 for _, d in g_nx.out_degree() if d == 0)
                leaf_ratio = num_leaves / num_nodes if num_nodes > 0 else 0.0
                density = nx.density(g_nx) if num_nodes > 1 else 0.0
                diameter = 0
                avg_shortest_path = 0
                if num_nodes > 1 and nx.is_weakly_connected(g_nx):
                    undirected_g = g_nx.to_undirected()
                    if undirected_g.number_of_nodes() > 0:
                        try:
                            largest_cc_nodes = max(nx.connected_components(undirected_g), key=len)
                            largest_cc = undirected_g.subgraph(largest_cc_nodes)
                            if largest_cc.number_of_nodes() > 1:
                                diameter = nx.diameter(largest_cc)
                                avg_shortest_path = nx.average_shortest_path_length(largest_cc)
                        except nx.NetworkXError:
                            diameter = 0
                            avg_shortest_path = 0
                linearity_ratio = diameter / num_nodes if num_nodes > 0 else 0.0
                num_roots = sum(1 for _, d in g_nx.in_degree() if d == 0)
                root_ratio = num_roots / num_nodes if num_nodes > 0 else 0.0
                graph_level_features_list = [
                    num_nodes, log_num_nodes,
                    degree_std, leaf_ratio, density,
                    diameter, avg_shortest_path,
                    linearity_ratio, root_ratio
                ]
                raw_graph_features_list.append(graph_level_features_list)
                
                data_list.append(graph_data)
            
            except Exception as e:
                print(f"\n警告：由于处理错误跳过行索引 {index} 的数据：{e}")
                continue

        # --- 【关键修复】检查data_list是否为空 ---
        if not data_list:
            print("所有数据解析失败，检查前几条失败的数据...")
            for index in range(min(10, len(self.raw_dataframe))):
                row = self.raw_dataframe.iloc[index]
                mermaid_code = row['Mermaid 代码']
                print(f"\n数据 {index}:")
                print(f"类型: {type(mermaid_code)}")
                print(f"内容: {repr(mermaid_code)}")
                nodes, edges = parse_mermaid(mermaid_code)
                print(f"解析结果 - 节点: {len(nodes)}, 边: {len(edges)}")
            
            raise RuntimeError("处理后没有任何有效的数据！请检查上述调试信息。")

        # --- 【新增】在所有图处理完后，进行全局特征归一化 ---
        if raw_graph_features_list:
            all_graph_features = torch.tensor(raw_graph_features_list, dtype=torch.float)
            
            # 计算 Min-Max 参数
            min_vals = torch.min(all_graph_features, dim=0).values
            max_vals = torch.max(all_graph_features, dim=0).values
            
            # 存储归一化参数，用于推理
            norm_params = {'min': min_vals, 'max': max_vals}
            torch.save(norm_params, self.processed_paths[1])
            
            # 应用归一化
            range_vals = max_vals - min_vals + 1e-8 
            
            for i, graph_data in enumerate(data_list):
                raw_features = torch.tensor([raw_graph_features_list[i]], dtype=torch.float)
                normalized_features = (raw_features - min_vals) / range_vals
                # 【关键修复】确保graph_features也在CPU上
                graph_data.graph_features = normalized_features.cpu()
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    


# ==============================================================================
# 2. GNN模型定义 (使用Set2Set和特征处理器)
# ==============================================================================

class GATScorer(torch.nn.Module):
    # --- num_graph_features 默认值更新为 9 ---
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_graph_features=9):
        super(GATScorer, self).__init__()
        
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.5)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.5)
        self.conv3 = GATConv(hidden_dim * num_heads, hidden_dim * 2, heads=1, concat=True, dropout=0.5)

        # --- 回归到 Set2Set 池化层 ---
        self.set2set = Set2Set(hidden_dim * 2, processing_steps=3)
        
        # --- 新增：一个简单的MLP来处理全局特征 ---
        self.graph_feature_processor = Sequential(
            Linear(num_graph_features, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim // 2)
        )

        # MLP输入维度 = 池化层输出 + 处理后的全局特征维度
        mlp_input_dim = 2 * (hidden_dim * 2) + (hidden_dim // 2)
        
        self.mlp = Sequential(
            Linear(mlp_input_dim, 128),
            ReLU(),
            Dropout(0.6),
            Linear(128, 4)
        )

    def forward(self, data):
        x, edge_index, batch, graph_features = data.x, data.edge_index, data.batch, data.graph_features
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        
        graph_embedding_pooled = self.set2set(x, batch)
        
        # --- 新增：处理全局特征 ---
        processed_graph_features = self.graph_feature_processor(graph_features)

        # 将GNN嵌入和处理后的全局特征拼接
        final_embedding = torch.cat([graph_embedding_pooled, processed_graph_features], dim=1)

        scores = self.mlp(final_embedding)
        return scores

# ==============================================================================
# 3. 训练和评估模块 (无变化)
# ==============================================================================

def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False)
    for data in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(loader.dataset)

class WeightedMAELoss(torch.nn.Module):
    def __init__(self, weights, device):
        super(WeightedMAELoss, self).__init__()
        self.weights = torch.tensor(weights, device=device)
        self.l1_loss = torch.nn.L1Loss(reduction='none')
        self.max_scores = torch.tensor([35, 35, 20, 10], device=device)

    def forward(self, pred_norm, target_norm):
        pred_real = pred_norm * self.max_scores
        target_real = target_norm * self.max_scores
        sub_losses = torch.mean(self.l1_loss(pred_real, target_real), dim=0)
        pred_total = torch.sum(pred_real, dim=1)
        target_total = torch.sum(target_real, dim=1)
        total_loss = torch.mean(self.l1_loss(pred_total, target_total))
        all_losses = torch.cat([sub_losses, total_loss.unsqueeze(0)])
        weighted_loss = torch.sum(all_losses * self.weights)
        return weighted_loss

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        all_preds.append(out)
        all_labels.append(data.y)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    mse_loss = F.mse_loss(all_preds, all_labels).item()
    max_scores_tensor = torch.tensor([35, 35, 20, 10], device=device)
    preds_real = all_preds * max_scores_tensor
    labels_real = all_labels * max_scores_tensor
    total_preds = torch.sum(preds_real, dim=1)
    total_labels = torch.sum(labels_real, dim=1)
    total_score_mae = F.l1_loss(total_preds, total_labels).item()
    return mse_loss, total_score_mae

# ==============================================================================
# 4. 主执行流程 (已修改)
# ==============================================================================

if __name__ == '__main__':
    # --- 配置参数 ---
    DATA_FILE_PATH = 'new_data.csv'
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005 # 恢复较小的学习率
    WEIGHT_DECAY = 5e-4 # 恢复较小的正则化
    EPOCHS = 200

    # --- 加载数据 ---
    if not os.path.exists(DATA_FILE_PATH):
        print(f"错误：数据文件 '{DATA_FILE_PATH}' 未找到！")
        exit()

    try:
        raw_dataframe = pd.read_csv(DATA_FILE_PATH)
        print(f"成功从 '{DATA_FILE_PATH}' 加载了 {len(raw_dataframe)} 行数据。")
    except Exception as e:
        print(f"加载CSV文件时出错：{e}")
        exit()
        
    # --- 创建数据集和数据加载器 ---
    # **重要提示**: 每次修改特征工程后，请务必手动删除旧的缓存文件夹。
    dataset = MermaidDataset(root='./data_processed_normalized_features', raw_dataframe=raw_dataframe)
    
    if len(dataset) < len(raw_dataframe) * 0.9:
        print(f"\n警告：处理了 {len(dataset)} 个图，原始数据有 {len(raw_dataframe)} 条记录。"
              f"许多记录被跳过。请检查CSV数据的格式错误。")

    if len(dataset) < 20:
        print("错误：有效数据点太少，无法继续训练。请检查数据文件。")
        exit()
        
    train_dataset, temp_dataset = train_test_split(dataset, test_size=0.3, random_state=42)
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=42)

    print(f"\n数据集划分：训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- 初始化模型、优化器和损失函数 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    input_dim = dataset.num_node_features
    print(f"最终的节点特征维度: {input_dim}") # 应该是 16 (text) + 5 (struct) = 21
    
    # --- 【修改】创建最终版模型实例 ---
    model = GATScorer(input_dim=input_dim, hidden_dim=128, num_heads=8, num_graph_features=9).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数总数（最终版模型）：{total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    loss_weights = [1.0, 1.0, 1.0, 1.0, 5.0]
    criterion = WeightedMAELoss(weights=loss_weights, device=device)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
        
    # --- 开始训练 ---
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 40
    print("\n--- 开始训练 ---")
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_normalized_features.pt') # <-- 使用新模型名
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        print(
            f"轮次：{epoch:03d} | "
            f"训练损失：{train_loss:.4f} | "
            f"验证损失：{val_loss:.4f} | "
            f"验证总分MAE：{val_mae:.2f} 分"
        )

        if early_stop_counter >= early_stop_patience:
            print(f"\n验证损失连续 {early_stop_patience} 个epoch没有改善，触发早停！")
            break

    # --- 在测试集上评估最终模型 ---
    print("\n--- 在测试集上进行最终评估 ---")
    model.load_state_dict(torch.load('best_model_normalized_features.pt'))
    test_loss, test_mae = evaluate(model, test_loader, device)
    
    print(f"最终测试MSE损失：{test_loss:.4f}")
    print(f"最终测试MAE（近似）：{test_mae:.2f} 分")
    print(f"(平均而言，模型的总分预测在0-100分制下偏差约 {test_mae:.2f} 分)")
    
    # --- 推理示例 ---
    print("\n--- 推理示例 ---")
    if len(test_dataset) > 0:
        model.eval()
        sample_graph = test_dataset[0].to(device)
        predicted_scores_normalized = model(sample_graph)
        max_scores_tensor = torch.tensor([35, 35, 20, 10], device=device) 
        predicted_sub_scores = predicted_scores_normalized * max_scores_tensor
        true_sub_scores = sample_graph.y * max_scores_tensor
        predicted_total_score = torch.sum(predicted_sub_scores)
        true_total_score = torch.sum(true_sub_scores)
        score_names = ["Structure_Logic", "Content_Completeness", "Hierarchy_Clarity", "Code_Syntax"]
        
        print("测试集中一个样本图的评分对比：")
        for i in range(4):
            print(f"  - {score_names[i]:<22}: 真实={true_sub_scores[0][i]:.2f}, 预测={predicted_sub_scores[0][i]:.2f}")
        
        print(f"  - {'总分':<22}: 真实={true_total_score:.2f}, 预测={predicted_total_score:.2f}")

    else:
        print("测试集为空，无法执行推理。")