import os
import re
import json
import networkx as nx
import numpy as np

# --- 配置部分 ---
# 输入输出文件
INPUT_FILE = 'a.txt'
OUTPUT_FILE = 'b.txt'

# 定义评分标准和分数范围
SCORE_RANGES = {
    "详细型": (90, 100),
    "逐渐展开型": (80, 89),
    "简略大纲型": (70, 79),
    "直线型": (60, 69),
}

# 定义四个维度在总分中的参考权重（用于按比例分配）
# 这些权重是根据您的示例大致估算的
DIMENSION_WEIGHTS = {
    "Structure_Logic": 35,
    "Content_Completeness": 35,
    "Hierarchy_Clarity": 20,
    "Code_Syntax": 10,
}
TOTAL_WEIGHT = sum(DIMENSION_WEIGHTS.values())

# 定义“逐渐展开型”的节点数上限
GRADUAL_EXPANSION_NODE_LIMIT = 12

# --- 辅助函数 ---

def parse_mermaid(mermaid_code):
    """
    解析Mermaid代码，返回节点和边的列表。
    """
    nodes = set()
    edges = []
    
    # 匹配节点（无论是否有文本）和边
    pattern = re.compile(r'(\w+)\s*(?:\[.*?\]|\(.*?\))?\s*-->\s*(\w+)\s*(?:\[.*?\]|\(.*?\))?')
    
    lines = mermaid_code.strip().split('\n')
    
    for line in lines:
        match = pattern.search(line)
        if match:
            source, target = match.groups()
            nodes.add(source)
            nodes.add(target)
            edges.append((source, target))
            
    # 处理只有一行定义的节点，例如 A["..."]
    if not edges and len(lines) == 1:
        node_match = re.match(r'(\w+)', lines[0].strip())
        if node_match:
            nodes.add(node_match.group(1))

    return list(nodes), edges

def get_graph_properties(nodes, edges):
    """
    根据节点和边构建NetworkX图，并计算所需属性。
    """
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    
    num_nodes = g.number_of_nodes()
    
    # 计算最大深度
    max_depth = 0
    if num_nodes > 0:
        root_nodes = [n for n, d in g.in_degree() if d == 0]
        if not root_nodes: # 如果有环，选择一个伪根
            root_nodes = [nodes[0]] if nodes else []
        
        all_depths = []
        for root in root_nodes:
            try:
                # 获取从根节点到所有可达节点的最短路径长度
                depths = nx.shortest_path_length(g, source=root)
                all_depths.extend(depths.values())
            except nx.NetworkXNoPath:
                pass
        
        if all_depths:
            max_depth = max(all_depths)
            
    return g, num_nodes, max_depth

def is_linear_graph(g, num_nodes):
    """
    判断一个图是否为“一条直线”。
    依据：
    1. 每个节点的入度和出度最多为1。
    2. 有且仅有一个根节点（入度为0）和一个叶节点（出度为0）。
    3. 其他所有中间节点的入度和出度都为1。
    4. 边数 = 节点数 - 1
    """
    if num_nodes <= 1:
        return True # 单个节点或空图也视为线性
    
    if g.number_of_edges() != num_nodes - 1:
        return False
        
    in_degrees = dict(g.in_degree())
    out_degrees = dict(g.out_degree())
    
    root_count = 0
    leaf_count = 0
    middle_count = 0
    
    for node in g.nodes():
        in_d = in_degrees.get(node, 0)
        out_d = out_degrees.get(node, 0)
        
        if in_d > 1 or out_d > 1:
            return False # 任何节点的分支大于1，则不是直线
        if in_d == 0 and out_d == 1:
            root_count += 1
        elif in_d == 1 and out_d == 0:
            leaf_count += 1
        elif in_d == 1 and out_d == 1:
            middle_count += 1
        else:
            return False # 存在孤立节点或不符合直线结构
            
    return root_count == 1 and leaf_count == 1

def classify_graph(g, num_nodes, max_depth):
    """
    根据规则对图进行分类。
    """
    if is_linear_graph(g, num_nodes):
        return "直线型"
    
    if num_nodes > GRADUAL_EXPANSION_NODE_LIMIT:
        return "详细型"
        
    if max_depth <= 1 and num_nodes > 1: # 只有根节点和一层子节点
        return "简略大纲型"
        
    # 剩余的节点数在12以内，且层数不为1（即层数大于等于2）的图
    # 您的定义中“层数不一致”和“非完全图”比较模糊，这里简化为
    # 只要不是直线型、简略大纲型，且节点数在12以内，就归为逐渐展开型
    return "逐渐展开型"

def calculate_total_score(map_type, num_nodes):
    """
    根据类型和节点数在分数区间内进行线性插值计算总分。
    同一类型内，节点数越多，分数越高。
    """
    min_score, max_score = SCORE_RANGES[map_type]
    
    # 为每种类型定义一个合理的节点数参考范围，用于插值
    # 这些范围可以根据经验调整
    node_range = {
        "直线型": (2, 20),      # 假设直线图最长20个节点
        "简略大纲型": (2, 10),  # 假设大纲图最多10个节点
        "逐渐展开型": (3, GRADUAL_EXPANSION_NODE_LIMIT), # 3到12个节点
        "详细型": (GRADUAL_EXPANSION_NODE_LIMIT + 1, 50) # 13到50个节点
    }
    
    min_nodes, max_nodes = node_range[map_type]
    
    # 确保当前节点数在定义的范围内，避免超出
    clamped_num_nodes = max(min_nodes, min(num_nodes, max_nodes))
    
    # 线性插值
    if max_nodes == min_nodes:
        percentage = 0.5 # 如果范围只有一个点，取中间分数
    else:
        percentage = (clamped_num_nodes - min_nodes) / (max_nodes - min_nodes)
        
    score_range_size = max_score - min_score
    total_score = min_score + (score_range_size * percentage)
    
    return int(round(total_score))

def distribute_sub_scores(total_score):
    """
    根据总分和预设的维度权重，按比例分配四个维度的分数。
    """
    sub_scores = {}
    
    for dim, weight in DIMENSION_WEIGHTS.items():
        sub_scores[dim] = int(round(total_score * (weight / TOTAL_WEIGHT)))
        
    # 由于四舍五入，子分数之和可能不等于总分，进行修正
    current_sum = sum(sub_scores.values())
    diff = total_score - current_sum
    
    # 将差值加到权重最高的维度上
    if diff != 0:
        # 找到权重最高的维度
        highest_weight_dim = max(DIMENSION_WEIGHTS, key=DIMENSION_WEIGHTS.get)
        sub_scores[highest_weight_dim] += diff
        
    return sub_scores

# --- 主执行流程 ---
def main():
    print(f"Reading Mermaid code from '{INPUT_FILE}'...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            mermaid_code = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{INPUT_FILE}'。请创建该文件并填入Mermaid代码。")
        return

    # 1. 解析Mermaid代码并获取图属性
    nodes, edges = parse_mermaid(mermaid_code)
    g, num_nodes, max_depth = get_graph_properties(nodes, edges)
    
    if num_nodes == 0:
        print("错误：无法从Mermaid代码中解析出任何有效节点。")
        return
        
    print(f"Graph properties: Nodes={num_nodes}, Edges={len(edges)}, Max Depth={max_depth}")

    # 2. 根据规则对图进行分类
    map_type = classify_graph(g, num_nodes, max_depth)
    print(f"Classified as: {map_type}")

    # 3. 计算总分
    total_score = calculate_total_score(map_type, num_nodes)
    print(f"Calculated Total Score: {total_score}")

    # 4. 按比例分配子分数
    sub_scores = distribute_sub_scores(total_score)
    
    # 5. 整合最终结果
    final_result = {
        "Structure_Logic": sub_scores["Structure_Logic"],
        "Content_Completeness": sub_scores["Content_Completeness"],
        "Hierarchy_Clarity": sub_scores["Hierarchy_Clarity"],
        "Code_Syntax": sub_scores["Code_Syntax"],
        "Total_Score": total_score
    }
    
    # 6. 写入文件
    print(f"Writing scores to '{OUTPUT_FILE}'...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)
        print("Scores successfully saved.")
        
        print("\n--- Final Scores ---")
        for name, score in final_result.items():
            print(f"- {name:<22}: {score}")

    except Exception as e:
        print(f"写入输出文件时出错: {e}")


if __name__ == '__main__':
    main()