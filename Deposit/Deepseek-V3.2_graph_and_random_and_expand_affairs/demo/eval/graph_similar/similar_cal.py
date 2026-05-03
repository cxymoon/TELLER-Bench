import networkx as nx
import os
import re
from typing import Tuple, Dict, Any, Set

def parse_api_graph(txt_path: str) -> nx.DiGraph:
    """
    从一个 TXT 格式（每个节点用类似 YAML-block 风格，但我们不用 yaml 库）读取，
    构建有向图：
    - 节点：api_name
    - 边：direct_dependencies dependent_api → this api
    假设每个节点块以 "- api_name:" 开头，属性缩进且 key: "value" 格式。
    """
    G = nx.DiGraph()
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    node_blocks = []
    current_block = None
    for line in lines:
        if re.match(r'^\s*-\s*api_name\s*:', line):
            # 开始新节点块
            if current_block is not None:
                node_blocks.append(current_block)
            current_block = [line]
        elif current_block is not None:
            current_block.append(line)
    if current_block is not None:
        node_blocks.append(current_block)

    for block in node_blocks:
        block_text = "".join(block)
        # 提取 api_name
        m_api = re.search(r'api_name\s*:\s*["\']?([^"\',\s]+?)["\']?(?:\s*[,}\s]|$)', block_text)
        if not m_api:
            continue
        api = m_api.group(1).strip()
        G.add_node(api)
        # 提取 direct_dependencies 列表中的 dependent_api
        # 我们假设 direct_dependencies: 下每个条目以 '- dependent_api:' 开头
        deps = re.findall(r'dependent_api\s*:\s*["\']?([^"\',\s}]+?)["\']?(?:\s*[,}\s]|$)', block_text)
        for dep_api in deps:
            dep_api = dep_api.strip()
            G.add_edge(dep_api, api)
    return G

def compute_graph_metrics(G1: nx.DiGraph, G2: nx.DiGraph) -> Dict[str, float]:
    """
    计算图的三个核心指标：精确率、召回率、F1
    """
    V1, V2 = set(G1.nodes()), set(G2.nodes())
    E1, E2 = set(G1.edges()), set(G2.edges())
    
    # 节点级别的指标
    node_true_positives = len(V1 & V2)  # 两个图共有的节点
    node_false_positives = len(V1 - V2)  # G1有但G2没有的节点
    node_false_negatives = len(V2 - V1)  # G2有但G1没有的节点
    
    # 节点精确率、召回率、F1
    node_precision = node_true_positives / len(V1) if len(V1) > 0 else 0
    node_recall = node_true_positives / len(V2) if len(V2) > 0 else 0
    if node_precision + node_recall > 0:
        node_f1 = 2 * (node_precision * node_recall) / (node_precision + node_recall)
    else:
        node_f1 = 0
    
    # 边级别的指标
    edge_true_positives = len(E1 & E2)  # 两个图共有的边
    edge_false_positives = len(E1 - E2)  # G1有但G2没有的边
    edge_false_negatives = len(E2 - E1)  # G2有但G1没有的边
    
    # 边精确率、召回率、F1
    edge_precision = edge_true_positives / len(E1) if len(E1) > 0 else 0
    edge_recall = edge_true_positives / len(E2) if len(E2) > 0 else 0
    if edge_precision + edge_recall > 0:
        edge_f1 = 2 * (edge_precision * edge_recall) / (edge_precision + edge_recall)
    else:
        edge_f1 = 0
    
    return {
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
        "node_true_positives": node_true_positives,
        "node_false_positives": node_false_positives,
        "node_false_negatives": node_false_negatives,
        "edge_true_positives": edge_true_positives,
        "edge_false_positives": edge_false_positives,
        "edge_false_negatives": edge_false_negatives
    }

def compute_similarity(G1: nx.DiGraph, G2: nx.DiGraph,
                       alpha: float=0.4, beta: float=0.4, gamma: float=0.2
                      ) -> Tuple[float, Dict[str, float]]:
    V1, V2 = set(G1.nodes()), set(G2.nodes())
    E1, E2 = set(G1.edges()), set(G2.edges())
    
    # 计算图指标
    metrics = compute_graph_metrics(G1, G2)
    
    # 原有的重叠度计算（作为召回率）
    node_overlap = len(V1 & V2) / len(V1 | V2) if (V1 | V2) else 0.0
    edge_overlap = len(E1 & E2) / len(E1 | E2) if (E1 | E2) else 0.0

    paths1 = set()
    for u in G1.nodes():
        for v in G1.nodes():
            if u != v:
                for p in nx.all_simple_paths(G1, u, v, cutoff=4):
                    if len(p) >= 2:
                        paths1.add(tuple(p))
    paths2 = set()
    for u in G2.nodes():
        for v in G2.nodes():
            if u != v:
                for p in nx.all_simple_paths(G2, u, v, cutoff=4):
                    if len(p) >= 2:
                        paths2.add(tuple(p))
    if paths1 or paths2:
        path_overlap = len(paths1 & paths2) / len(paths1 | paths2)
    else:
        path_overlap = 0.0

    sim_score = alpha * node_overlap + beta * edge_overlap + gamma * path_overlap

    # 合并所有指标
    all_details = {
        "node_overlap": node_overlap,
        "edge_overlap": edge_overlap,
        "path_overlap": path_overlap,
        **metrics  # 包含所有精确率、召回率、F1指标
    }

    return sim_score, all_details


if __name__ == "__main__":
    file2 = "/Deposit/Deepseek-V3.2_graph_and_random_and_expand_affairs/demo/direct_graph_generation/response_output.txt"
    file1 = "/Deposit/Deepseek-V3.2_graph_and_random_and_expand_affairs/demo/eval/SOP_eval/response_output.txt"
    G1 = parse_api_graph(file1)
    G2 = parse_api_graph(file2)
    sim, details = compute_similarity(G1, G2)
    
    print(f"Similarity score: {sim:.4f}")
    print("\n=== 重叠度指标 ===")
    print(f"节点重叠度: {details['node_overlap']:.4f}")
    print(f"边重叠度: {details['edge_overlap']:.4f}")
    print(f"路径重叠度: {details['path_overlap']:.4f}")
    
    print("\n=== 节点级别指标 ===")
    print(f"节点精确率: {details['node_precision']:.4f}")
    print(f"节点召回率: {details['node_recall']:.4f}")
    print(f"节点F1: {details['node_f1']:.4f}")
    print(f"节点统计 - 真正例: {details['node_true_positives']}, 假正例: {details['node_false_positives']}, 假反例: {details['node_false_negatives']}")
    
    print("\n=== 边级别指标 ===")
    print(f"边精确率: {details['edge_precision']:.4f}")
    print(f"边召回率: {details['edge_recall']:.4f}")
    print(f"边F1: {details['edge_f1']:.4f}")
    print(f"边统计 - 真正例: {details['edge_true_positives']}, 假正例: {details['edge_false_positives']}, 假反例: {details['edge_false_negatives']}")