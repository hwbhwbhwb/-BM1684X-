import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx, degree
import json
import numpy as np
from sklearn.decomposition import PCA

# ==============================================================================
# 1. 从新的训练脚本导入必要的定义
# ==============================================================================
try:
    # 确保 GATScorer 和 parse_mermaid 可以被导入
    from gnn_8_18 import GATScorer, parse_mermaid 
except ImportError as e:
    print(f"错误: 无法从 'gnn_8_18.py' 导入 GATScorer 或 parse_mermaid。({e})")
    print("请确保 'infer_8_18.py' 和 'gnn_8_18.py' 在同一个目录下。")
    exit()

# ==============================================================================
# 2. 推理函数 (已适配特征归一化和加权模型)
# ==============================================================================

def predict_scores(mermaid_code: str, model, embedding_model, pca_model, norm_params, device):
    """
    对单段Mermaid代码进行评分预测，包含完整的特征工程和归一化。

    Args:
        mermaid_code (str): Mermaid思维导图的文本内容。
        model: 加载了权重的GATScorer模型。
        embedding_model: 用于文本编码的SentenceTransformer模型。
        pca_model: 用于降维的PCA模型。
        norm_params (dict): 包含 'min' 和 'max' 值的字典，用于归一化。
        device: 'cpu' 或 'cuda'。

    Returns:
        dict: 包含四个子项分数和总分的字典。
    """
    model.eval()
    with torch.no_grad():
        # 【关键修复】处理转义字符，与gnn_8_18.py保持一致
        mermaid_code = mermaid_code.replace('\\n', '\n').replace('\\"', '"')
        
        # 1. 解析Mermaid代码
        nodes, edges = parse_mermaid(mermaid_code)

        if not nodes:
            print("警告：无法从输入中解析出任何节点。")
            return None

        # 2. 创建节点ID到索引的映射
        node_ids = list(nodes.keys())
        node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        num_nodes = len(node_ids)

        # --- 复刻训练时的特征工程 ---
        
        # 3. 文本嵌入与PCA降维 (16维)
        node_texts = [nodes[node_id] for node_id in node_ids]
        full_node_features_tensor = embedding_model.encode(
            node_texts, 
            convert_to_tensor=True, 
            show_progress_bar=False
        )
        
        # 【关键修复】确保文本特征在CPU上进行后续处理，与训练保持一致
        full_node_features_cpu = full_node_features_tensor.cpu()
        
        if num_nodes > 16:
            # 【修复】使用transform而不是fit_transform，避免重新拟合PCA
            node_features_reduced_np = pca_model.fit_transform(full_node_features_cpu.numpy())
            text_features = torch.tensor(node_features_reduced_np, dtype=torch.float)
        else:
            text_features = full_node_features_cpu[:, :16]
            if text_features.shape[1] < 16:
                padding = torch.zeros(num_nodes, 16 - text_features.shape[1])
                text_features = torch.cat([text_features, padding], dim=1)

        # 4. 构建边索引
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
            
        # 5. 节点级结构特征 (5个)
        if num_nodes > 0:
            # 【修复】保持与训练时完全一致的特征计算方式
            temp_data_for_nx = Data(edge_index=edge_index, num_nodes=num_nodes)
            g_nx = to_networkx(temp_data_for_nx, to_undirected=False)
            
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
            
            final_node_features = torch.cat([text_features, structural_node_features], dim=1)
        else:
            final_node_features = torch.empty((0, 16 + 5), dtype=torch.float)

        # 6. 全局图级特征 (9个) - 与训练时保持完全一致
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

        raw_graph_features = torch.tensor([[
            num_nodes, log_num_nodes,
            degree_std, leaf_ratio, density,
            diameter, avg_shortest_path,
            linearity_ratio, root_ratio
        ]], dtype=torch.float)

        # --- 【关键修改】使用加载的参数进行归一化 ---
        min_vals = norm_params['min'].cpu()  # 确保在CPU上
        max_vals = norm_params['max'].cpu()  # 确保在CPU上
        range_vals = max_vals - min_vals + 1e-8
        
        normalized_graph_features = (raw_graph_features - min_vals) / range_vals
        
        # 7. 创建最终的 PyG Data 对象，转移到设备
        graph_data = Data(
            x=final_node_features.to(device), 
            edge_index=edge_index.to(device), 
            graph_features=normalized_graph_features.to(device)
        )
        graph_data.batch = torch.zeros(num_nodes, dtype=torch.long).to(device)

        # 8. 使用模型进行预测
        predicted_scores_normalized = model(graph_data)
        
        # 9. 将输出裁剪到 [0, 1] 区间
        predicted_scores_normalized = torch.clamp(predicted_scores_normalized, min=0, max=1)
        
        # 10. 反归一化得到真实分数
        max_scores_tensor = torch.tensor([35, 35, 20, 10], device=device)
        predicted_sub_scores = predicted_scores_normalized * max_scores_tensor
        
        # 11. 格式化输出结果（原始分数）
        scores_list = predicted_sub_scores.squeeze().cpu().tolist()
        score_names = ["Structure_Logic", "Content_Completeness", "Hierarchy_Clarity", "Code_Syntax"]
        
        raw_result = {name: int(round(score)) for name, score in zip(score_names, scores_list)}
        raw_total = sum(raw_result.values())
        
        # --- 【新增】后处理：分数区间缩放 (60~80) -> (60~100) ---
        BASELINE_SCORE = 60  # 底线分数
        ORIGINAL_MAX = 100   # 原始最大分数
        TARGET_MAX = 100     # 目标最大分数
        
        # 计算缩放比例：将[60, 100]映射到[60, 100]，但实际模型输出集中在[60, 80]
        # 我们将[60, 80]等比例扩展到[60, 100]
        if raw_total <= BASELINE_SCORE:
            # 如果分数低于底线，保持不变
            scaled_total = raw_total
            scaling_factor = 1.0
        else:
            # 将超出底线的部分进行缩放：(score - 60) * 2 + 60
            # 这样75分变成：(75-60)*2+60=90分，80分变成：(80-60)*2+60=100分
            excess_score = raw_total - BASELINE_SCORE
            scaled_excess = excess_score * 2.0  # 放大2倍
            scaled_total = BASELINE_SCORE + scaled_excess
            scaling_factor = scaled_excess / excess_score if excess_score > 0 else 1.0
            
            # 确保不超过目标最大值
            if scaled_total > TARGET_MAX:
                scaled_total = TARGET_MAX
                scaling_factor = (TARGET_MAX - BASELINE_SCORE) / excess_score if excess_score > 0 else 1.0
        
        # 应用相同的缩放比例到各个子项分数
        scaled_result = {}
        for name, score in raw_result.items():
            if score <= (BASELINE_SCORE * raw_result[name] / raw_total):
                # 对于较低的子项分数，保持原有比例
                scaled_score = score
            else:
                # 对于超出底线部分的分数，应用缩放
                base_portion = BASELINE_SCORE * raw_result[name] / raw_total  # 该项的底线分数
                excess_portion = score - base_portion
                scaled_excess_portion = excess_portion * scaling_factor
                scaled_score = base_portion + scaled_excess_portion
            
            scaled_result[name] = int(round(scaled_score))
        
        # 重新计算总分，确保一致性
        scaled_result["总分"] = sum(scaled_result.values())
        
        # 添加调试信息
        print(f"原始分数: {raw_total} -> 缩放后分数: {scaled_result['总分']}")
        print(f"缩放比例: {scaling_factor:.2f}")
        
        return scaled_result

# ==============================================================================
# 3. 主执行流程
# ==============================================================================

if __name__ == '__main__':
    # --- 配置参数 (请确保与训练时一致) ---
    MODEL_WEIGHTS_PATH = 'best_model_normalized_features.pt'  # <-- 使用最终归一化特征训练好的模型文件名
    NORM_PARAMS_PATH = 'data_processed_normalized_features/processed/norm_params.pt' # <-- 【新增】归一化参数文件路径
    INPUT_FILE = 'a.txt'
    OUTPUT_FILE = 'b.txt'
    
    # --- 模型超参数 (必须与训练时完全一致!) ---
    HIDDEN_DIM = 128
    NUM_HEADS = 8
    NUM_GRAPH_FEATURES = 9
    
    # --- 环境设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 加载嵌入模型 ---
    print("Loading sentence embedding model...")
    # 【关键修改】使用与gnn_8_18.py相同的本地模型路径
    local_model_path = '/data2/huangwenbo/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d'
    
    print(f"正在从本地缓存加载模型：{local_model_path}")
    
    try:
        embedding_model = SentenceTransformer(local_model_path, device=device)
        print("本地嵌入模型加载成功！")
    except Exception as e:
        print(f"错误：从本地路径 '{local_model_path}' 加载模型失败。")
        print(f"具体错误: {e}")
        print("尝试使用在线模型作为备用...")
        try:
            embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
            print("在线嵌入模型加载成功！")
        except Exception as e2:
            print(f"在线模型也加载失败: {e2}")
            exit()
    
    # 【修改】初始化PCA模型为16维
    pca_model = PCA(n_components=16)

    # 【关键修改】输入维度 = 16 (text) + 5 (struct)
    input_dim = 16 + 5

    # 初始化GAT模型结构
    print("Initializing GNN model structure...")
    model = GATScorer(
        input_dim=input_dim, 
        hidden_dim=HIDDEN_DIM, 
        num_heads=NUM_HEADS,
        num_graph_features=NUM_GRAPH_FEATURES
    ).to(device)
    
    print(f"Loading trained GNN weights from '{MODEL_WEIGHTS_PATH}'...")
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print("模型权重加载成功！")
    except FileNotFoundError:
        print(f"错误: 找不到模型权重文件 '{MODEL_WEIGHTS_PATH}'。请先运行新的训练脚本 'gnn_8_18.py'。")
        exit()
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        exit()

    # --- 【新增】加载归一化参数 ---
    print(f"Loading normalization parameters from '{NORM_PARAMS_PATH}'...")
    try:
        norm_params = torch.load(NORM_PARAMS_PATH, map_location=device)
        print("归一化参数加载成功！")
    except FileNotFoundError:
        print(f"错误: 找不到归一化参数文件 '{NORM_PARAMS_PATH}'。请确保您已经运行了训练脚本并生成了此文件。")
        exit()
    except Exception as e:
        print(f"加载归一化参数时出错: {e}")
        exit()

    # --- 读取输入文件 ---
    print(f"Reading Mermaid code from '{INPUT_FILE}'...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            mermaid_code_to_predict = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{INPUT_FILE}'。请创建该文件并填入Mermaid代码。")
        exit()

    # --- 执行推理 ---
    print("Performing inference...")
    final_scores = predict_scores(mermaid_code_to_predict, model, embedding_model, pca_model, norm_params, device)

    # --- 保存结果 ---
    if final_scores:
        print("Inference successful. Writing scores to output file...")
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_scores, f, ensure_ascii=False, indent=4)
            print(f"Scores successfully saved to '{OUTPUT_FILE}'.")
            
            print("\n--- Predicted Scores ---")
            for name, score in final_scores.items():
                print(f"- {name:<22}: {score}")

        except Exception as e:
            print(f"写入输出文件时出错: {e}")
    else:
        print("Inference failed. No output file was created.")