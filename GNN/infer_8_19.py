import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx, degree
import json
import numpy as np
import random  # <-- 添加导入

# ==============================================================================
# 1. 从新的训练脚本导入必要的定义
# ==============================================================================
try:
    from gnn_8_19 import GCNScorer, parse_mermaid
except ImportError as e:
    print(f"错误: 无法从 'gnn_8_19.py' 导入 GCNScorer 或 parse_mermaid。({e})")
    print("请确保 'infer_8_19.py' 和 'gnn_8_19.py' 在同一个目录下。")
    exit()

# ==============================================================================
# 2. 推理函数 (混合规则+GNN)
# ==============================================================================

def rule_based_scorer_for_outline():
    """
    为"简略大纲型"生成一个在60-69之间的随机分数，并按比例分配子项。
    """
    total_score = random.randint(60, 69)
    weights = {
        "Structure_Logic": 25,
        "Content_Completeness": 25,
        "Hierarchy_Clarity": 15,
        "Code_Syntax": 8,
    }
    total_weight = sum(weights.values())
    sub_scores = {}
    for name, weight in weights.items():
        sub_scores[name] = int(round(total_score * (weight / total_weight)))
    current_sum = sum(sub_scores.values())
    diff = total_score - current_sum
    if diff != 0:
        highest_weight_dim = max(weights, key=weights.get)
        sub_scores[highest_weight_dim] += diff
    result = sub_scores
    result["总分"] = total_score
    result["预测类型"] = "简略大纲型 (规则判断)"
    return result

def predict_scores(mermaid_code: str, model, embedding_model, device):
    """
    对单段Mermaid代码进行评分预测，使用语义图构建。
    """
    # 【关键修复】处理转义字符，与gnn_8_19.py保持一致
    mermaid_code = mermaid_code.replace('\\n', '\n').replace('\\"', '"')
    
    nodes, explicit_edges = parse_mermaid(mermaid_code)
    if not nodes:
        print("警告：无法从输入中解析出任何节点。")
        return None

    node_ids = list(nodes.keys())
    num_nodes = len(node_ids)

    # 规则判断：如果最大深度为1，则直接规则打分
    if num_nodes > 1:
        g_nx_struct = nx.DiGraph()
        g_nx_struct.add_nodes_from(node_ids)
        g_nx_struct.add_edges_from(explicit_edges)
        max_depth = 0
        root_nodes = [n for n, d in g_nx_struct.in_degree() if d == 0]
        if not root_nodes and num_nodes > 0:
            root_nodes = [node_ids[0]]
        all_depths = []
        for root in root_nodes:
            try:
                depths = nx.shortest_path_length(g_nx_struct, source=root)
                all_depths.extend(depths.values())
            except nx.NetworkXNoPath:
                pass
        if all_depths:
            max_depth = max(all_depths)
        # if max_depth == 1:
        #     print("检测到图只有两层，使用规则进行打分...")
        #     print("节点列表:", node_ids)
        #     print("边列表:", explicit_edges)
        #     print("所有深度:", all_depths)
        #     print("最大深度:", max_depth)
        #     return rule_based_scorer_for_outline()

    # GNN预测流程
    print("图结构复杂，使用GNN进行预测...")
    model.eval()
    with torch.no_grad():
        node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        node_texts = [nodes[node_id] for node_id in node_ids]
        text_embeddings = embedding_model.encode(
            node_texts, convert_to_tensor=True, show_progress_bar=False
        ).to(device)
        cosine_scores = util.cos_sim(text_embeddings, text_embeddings)
        similarity_threshold = 0.5
        source_indices, target_indices, edge_weights = [], [], []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                similarity = cosine_scores[i, j].item()
                if similarity > similarity_threshold:
                    source_indices.extend([i, j])
                    target_indices.extend([j, i])
                    edge_weights.extend([similarity, similarity])
        if source_indices:
            semantic_edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long).to(device)
            semantic_edge_attr = torch.tensor(edge_weights, dtype=torch.float).to(device)
        else:
            semantic_edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
            semantic_edge_attr = torch.empty((0,), dtype=torch.float).to(device)
        if num_nodes > 0:
            if explicit_edges:
                valid_edges = [e for e in explicit_edges if e[0] in node_id_to_idx and e[1] in node_id_to_idx]
                if valid_edges:
                    src_struct = [node_id_to_idx[s] for s, t in valid_edges]
                    tgt_struct = [node_id_to_idx[t] for s, t in valid_edges]
                    edge_index_struct = torch.tensor([src_struct, tgt_struct], dtype=torch.long).to(device)
                else:
                    edge_index_struct = torch.empty((2, 0), dtype=torch.long).to(device)
            else:
                edge_index_struct = torch.empty((2, 0), dtype=torch.long).to(device)
            g_nx_struct_gnn = to_networkx(Data(edge_index=edge_index_struct.cpu(), num_nodes=num_nodes), to_undirected=False)
            in_degree = degree(edge_index_struct[1], num_nodes=num_nodes).view(-1, 1)
            out_degree = degree(edge_index_struct[0], num_nodes=num_nodes).view(-1, 1)
            node_depths = torch.zeros(num_nodes, 1, device=device)
            root_nodes_gnn = [n for n, d in g_nx_struct_gnn.in_degree() if d == 0]
            if not root_nodes_gnn and num_nodes > 0:
                root_nodes_gnn = [0]
            for root in root_nodes_gnn:
                try:
                    depths = nx.shortest_path_length(g_nx_struct_gnn, source=root)
                    for node, d in depths.items():
                        if d > node_depths[node][0]:
                            node_depths[node][0] = d
                except nx.NetworkXNoPath:
                    pass
            betweenness = torch.tensor(list(nx.betweenness_centrality(g_nx_struct_gnn).values()), dtype=torch.float).view(-1, 1).to(device)
            closeness = torch.tensor(list(nx.closeness_centrality(g_nx_struct_gnn).values()), dtype=torch.float).view(-1, 1).to(device)
            node_features = torch.cat([
                in_degree / num_nodes, out_degree / num_nodes, node_depths / num_nodes, betweenness, closeness
            ], dim=1)
        else:
            node_features = torch.empty((0, 5), dtype=torch.float).to(device)
        num_edges_struct = g_nx_struct.number_of_edges() if num_nodes > 1 else 0
        density = nx.density(g_nx_struct) if num_nodes > 1 else 0.0
        diameter = 0
        if num_nodes > 1 and nx.is_weakly_connected(g_nx_struct):
            undirected_g = g_nx_struct.to_undirected()
            if undirected_g.number_of_nodes() > 0:
                largest_cc_nodes = max(nx.connected_components(undirected_g), key=len)
                largest_cc = undirected_g.subgraph(largest_cc_nodes)
                if largest_cc.number_of_nodes() > 1:
                    diameter = nx.diameter(largest_cc)
        degree_centrality = list(nx.degree_centrality(g_nx_struct).values()) if num_nodes > 1 else []
        degree_std = np.std(degree_centrality) if degree_centrality else 0.0
        num_leaves = sum(1 for _, d in g_nx_struct.out_degree() if d == 0) if num_nodes > 1 else 0
        leaf_ratio = num_leaves / num_nodes if num_nodes > 0 else 0.0
        linearity_ratio = diameter / num_nodes if num_nodes > 0 else 0.0
        graph_level_features = torch.tensor([[
            num_nodes, np.log1p(num_nodes),
            degree_std, leaf_ratio, density,
            diameter, linearity_ratio
        ]], dtype=torch.float).to(device)
        graph_data = Data(
            x=node_features,
            edge_index=semantic_edge_index,
            edge_attr=semantic_edge_attr,
            graph_features=graph_level_features
        )
        graph_data.batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
        predicted_scores_normalized = model(graph_data)
        predicted_scores_normalized = torch.clamp(predicted_scores_normalized, min=0, max=1)
        max_scores_tensor = torch.tensor([35, 35, 20, 10], device=device)
        predicted_sub_scores = predicted_scores_normalized * max_scores_tensor
        scores_list = predicted_sub_scores.squeeze().cpu().tolist()
        score_names = ["Structure_Logic", "Content_Completeness", "Hierarchy_Clarity", "Code_Syntax"]
        result = {name: int(round(score)) for name, score in zip(score_names, scores_list)}
        result["总分"] = sum(result.values())
        result["预测类型"] = "GNN预测"
        return result

# ==============================================================================
# 3. 主执行流程
# ==============================================================================

if __name__ == '__main__':
    MODEL_WEIGHTS_PATH = 'best_model_semantic_graph.pt'
    INPUT_FILE = 'a.txt'
    OUTPUT_FILE = 'b.txt'
    HIDDEN_DIM = 128
    NUM_GRAPH_FEATURES = 7
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading sentence embedding model...")
    # 【关键修改】使用与gnn_8_19.py相同的本地模型路径
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
    
    input_dim = 5  # 节点特征维度：[归一化入度, 归一化出度, 归一化深度, 中介中心性, 接近中心性]
    
    print("Initializing GNN model structure (GCNScorer)...")
    model = GCNScorer(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_graph_features=NUM_GRAPH_FEATURES
    ).to(device)
    
    print(f"Loading trained GNN weights from '{MODEL_WEIGHTS_PATH}'...")
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print("模型权重加载成功！")
    except FileNotFoundError:
        print(f"错误: 找不到模型权重文件 '{MODEL_WEIGHTS_PATH}'。请先运行新的训练脚本 'gnn_8_19.py'。")
        exit()
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        exit()
    
    print(f"Reading Mermaid code from '{INPUT_FILE}'...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            mermaid_code_to_predict = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{INPUT_FILE}'。请创建该文件并填入Mermaid代码。")
        exit()
    
    print("Performing inference...")
    final_scores = predict_scores(mermaid_code_to_predict, model, embedding_model, device)
    
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