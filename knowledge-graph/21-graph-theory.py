"""
模块 21: 图谱理论基础
===================

本模块包含以下示例:
1. 使用 NetworkX 创建不同类型的图
2. 计算图的基本属性
3. 分析真实网络的拓扑特征
4. 实现 BFS 和 DFS 遍历

安装依赖:
pip install networkx matplotlib

运行示例:
python 21-graph-theory.py
"""

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import numpy as np


# ============================================================================
# 示例 1: 使用 NetworkX 创建不同类型的图
# ============================================================================

def example_1_create_graphs():
    """示例 1: 创建不同类型的图"""
    print("=" * 60)
    print("示例 1: 使用 NetworkX 创建不同类型的图")
    print("=" * 60)

    # 1. 无向图 (Undirected Graph)
    print("\n1. 无向图 (Undirected Graph)")
    G_undirected = nx.Graph()
    G_undirected.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C'), ('C', 'D')])
    print(f"   节点数: {G_undirected.number_of_nodes()}")
    print(f"   边数: {G_undirected.number_of_edges()}")
    print(f"   节点: {list(G_undirected.nodes())}")
    print(f"   边: {list(G_undirected.edges())}")

    # 2. 有向图 (Directed Graph)
    print("\n2. 有向图 (Directed Graph)")
    G_directed = nx.DiGraph()
    G_directed.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D')])
    print(f"   节点数: {G_directed.number_of_nodes()}")
    print(f"   边数: {G_directed.number_of_edges()}")
    print(f"   节点: {list(G_directed.nodes())}")
    print(f"   边: {list(G_directed.edges())}")

    # 3. 加权图 (Weighted Graph)
    print("\n3. 加权图 (Weighted Graph)")
    G_weighted = nx.Graph()
    G_weighted.add_weighted_edges_from([('A', 'B', 0.5), ('B', 'C', 0.8),
                                        ('A', 'C', 0.3), ('C', 'D', 0.6)])
    print(f"   节点数: {G_weighted.number_of_nodes()}")
    print(f"   边数: {G_weighted.number_of_edges()}")
    print(f"   边 A-B 的权重: {G_weighted['A']['B']['weight']}")

    # 4. 多重图 (MultiGraph)
    print("\n4. 多重图 (MultiGraph)")
    G_multi = nx.MultiGraph()
    G_multi.add_edges_from([('A', 'B'), ('A', 'B'), ('B', 'C')])
    print(f"   节点数: {G_multi.number_of_nodes()}")
    print(f"   边数: {G_multi.number_of_edges()}")
    print(f"   A 和 B 之间的边数: {G_multi.number_of_edges('A', 'B')}")

    # 可视化（可选）
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('不同类型的图', fontsize=16, fontweight='bold')

        # 无向图
        pos1 = nx.spring_layout(G_undirected, seed=42)
        nx.draw(G_undirected, pos1, ax=axes[0, 0], with_labels=True,
                node_color='lightblue', node_size=500, font_weight='bold')
        axes[0, 0].set_title('无向图')

        # 有向图
        pos2 = nx.spring_layout(G_directed, seed=42)
        nx.draw(G_directed, pos2, ax=axes[0, 1], with_labels=True,
                node_color='lightgreen', node_size=500, font_weight='bold',
                arrowsize=20, edge_color='red')
        axes[0, 1].set_title('有向图')

        # 加权图
        pos3 = nx.spring_layout(G_weighted, seed=42)
        nx.draw(G_weighted, pos3, ax=axes[1, 0], with_labels=True,
                node_color='lightyellow', node_size=500, font_weight='bold')
        nx.draw_networkx_edge_labels(G_weighted, pos3, ax=axes[1, 0],
                                     edge_labels=nx.get_edge_attributes(G_weighted, 'weight'))
        axes[1, 0].set_title('加权图')

        # 多重图
        pos4 = nx.spring_layout(G_multi, seed=42)
        nx.draw(G_multi, pos4, ax=axes[1, 1], with_labels=True,
                node_color='lightpink', node_size=500, font_weight='bold')
        axes[1, 1].set_title('多重图')

        plt.tight_layout()
        plt.savefig('21-example1-graph-types.png', dpi=150, bbox_inches='tight')
        print("\n   图像已保存为: 21-example1-graph-types.png")
    except Exception as e:
        print(f"\n   可视化失败: {e}")


# ============================================================================
# 示例 2: 计算图的基本属性
# ============================================================================

def example_2_graph_properties():
    """示例 2: 计算图的基本属性"""
    print("\n" + "=" * 60)
    print("示例 2: 计算图的基本属性")
    print("=" * 60)

    # 使用 Karate Club 图作为示例
    G = nx.karate_club_graph()
    print(f"\n使用 Karate Club 图: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

    # 1. 度 (Degree)
    print("\n1. 度 (Degree)")
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / len(degrees)
    print(f"   平均度: {avg_degree:.2f}")
    print(f"   最大度: {max(degrees.values())} (节点 {max(degrees, key=degrees.get)})")
    print(f"   最小度: {min(degrees.values())} (节点 {min(degrees, key=degrees.get)})")

    # 度分布
    degree_values = list(degrees.values())
    print(f"   度分布前5个: {sorted(degree_values, reverse=True)[:5]}")

    # 2. 最短路径
    print("\n2. 最短路径 (Shortest Path)")
    shortest_path = nx.shortest_path(G, 0, 33)
    print(f"   节点 0 到 33 的最短路径: {shortest_path}")
    print(f"   路径长度: {len(shortest_path) - 1}")

    # 所有节点对之间的平均最短路径长度
    avg_path_length = nx.average_shortest_path_length(G)
    print(f"   平均最短路径长度: {avg_path_length:.2f}")

    # 3. 图的密度
    print("\n3. 图的密度 (Density)")
    density = nx.density(G)
    print(f"   图密度: {density:.3f}")
    print(f"   说明: {'这是稀疏图' if density < 0.1 else '这是稠密图'}")

    # 4. 图的直径
    print("\n4. 图的直径 (Diameter)")
    diameter = nx.diameter(G)
    print(f"   图直径: {diameter}")
    print(f"   说明: 任意两个节点之间最短路径的最大值")

    # 5. 聚类系数
    print("\n5. 聚类系数 (Clustering Coefficient)")
    clustering = nx.average_clustering(G)
    print(f"   平均聚类系数: {clustering:.3f}")
    print(f"   说明: {'节点之间连接紧密' if clustering > 0.5 else '节点之间连接稀疏'}")


# ============================================================================
# 示例 3: 分析真实网络的拓扑特征
# ============================================================================

def example_3_network_topology():
    """示例 3: 分析真实网络的拓扑特征"""
    print("\n" + "=" * 60)
    print("示例 3: 分析真实网络的拓扑特征")
    print("=" * 60)

    # 使用 Karate Club 图
    G = nx.karate_club_graph()

    # 1. 度分布分析
    print("\n1. 度分布分析")
    degrees = [d for n, d in G.degree()]
    print(f"   最大度: {max(degrees)}")
    print(f"   平均度: {sum(degrees) / len(degrees):.2f}")
    print(f"   度标准差: {np.std(degrees):.2f}")

    # 检查是否为幂律分布（无标度网络特征）
    sorted_degrees = sorted(degrees, reverse=True)
    print(f"   前5个最大度: {sorted_degrees[:5]}")
    print(f"   说明: 存在{'Hub节点' if max(degrees) > sum(degrees)/len(degrees)*3 else '无明显Hub节点'}")

    # 2. 聚类系数
    print("\n2. 聚类系数分析")
    clustering = nx.average_clustering(G)
    print(f"   平均聚类系数: {clustering:.3f}")
    print(f"   说明: {'高聚类系数' if clustering > 0.3 else '低聚类系数'}")

    # 3. 小世界网络检测
    print("\n3. 小世界网络检测")

    def is_small_world(G):
        """判断是否为小世界网络"""
        clustering = nx.average_clustering(G)
        avg_path = nx.average_shortest_path_length(G)

        # 小世界网络特征: 高聚类系数 + 短路径长度
        return clustering > 0.3 and avg_path < 6

    sw_result = is_small_world(G)
    print(f"   是否为小世界网络: {sw_result}")

    if sw_result:
        print(f"   特征: 高聚类系数 ({clustering:.3f}) + 短路径长度 ({nx.average_shortest_path_length(G):.2f})")

    # 4. 连通性分析
    print("\n4. 连通性分析")
    is_connected = nx.is_connected(G)
    print(f"   图是否连通: {is_connected}")

    if is_connected:
        # 连通分量
        num_components = nx.number_connected_components(G)
        print(f"   连通分量数: {num_components}")

        # 找到最大连通分量
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"   最大连通分量大小: {len(largest_cc)}")

    # 5. 中心性分析
    print("\n5. 中心性分析 (前3个节点)")

    # 度中心性
    degree_centrality = nx.degree_centrality(G)
    top3_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"   度中心性 Top 3: {[(n, f"{c:.3f}") for n, c in top3_degree]}")

    # 中介中心性
    betweenness_centrality = nx.betweenness_centrality(G)
    top3_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"   中介中心性 Top 3: {[(n, f"{c:.3f}") for n, c in top3_betweenness]}")

    # 接近中心性
    closeness_centrality = nx.closeness_centrality(G)
    top3_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"   接近中心性 Top 3: {[(n, f"{c:.3f}") for n, c in top3_closeness]}")


# ============================================================================
# 示例 4: 实现 BFS 和 DFS 遍历
# ============================================================================

def bfs_custom(G, start):
    """自定义 BFS 实现"""
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in sorted(G.neighbors(node)):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


def dfs_custom(G, start, visited=None):
    """自定义 DFS 实现（递归）"""
    if visited is None:
        visited = set()

    visited.add(start)
    order = [start]

    for neighbor in sorted(G.neighbors(start)):
        if neighbor not in visited:
            order.extend(dfs_custom(G, neighbor, visited))

    return order


def example_4_graph_traversal():
    """示例 4: 实现 BFS 和 DFS 遍历"""
    print("\n" + "=" * 60)
    print("示例 4: 实现 BFS 和 DFS 遍历")
    print("=" * 60)

    # 创建一个简单的树状图用于演示
    G = nx.balanced_tree(2, 3)  # 2叉树，深度3
    print(f"\n使用平衡二叉树: {G.number_of_nodes()} 个节点")

    start_node = 0

    # 1. BFS 遍历
    print("\n1. BFS (广度优先搜索) 遍历")
    bfs_order = bfs_custom(G, start_node)
    print(f"   自定义 BFS 遍历顺序 (前20个): {bfs_order[:20]}")

    # NetworkX 内置 BFS
    bfs_edges = list(nx.bfs_edges(G, start_node))
    bfs_nx_order = [start_node] + [v for u, v in bfs_edges]
    print(f"   NetworkX BFS (前20个): {bfs_nx_order[:20]}")

    # 2. DFS 遍历
    print("\n2. DFS (深度优先搜索) 遍历")
    dfs_order = dfs_custom(G, start_node)
    print(f"   自定义 DFS 遍历顺序 (前20个): {dfs_order[:20]}")

    # NetworkX 内置 DFS
    dfs_edges = list(nx.dfs_edges(G, start_node))
    dfs_nx_order = [start_node] + [v for u, v in dfs_edges]
    print(f"   NetworkX DFS (前20个): {dfs_nx_order[:20]}")

    # 3. BFS vs DFS 对比
    print("\n3. BFS vs DFS 对比")
    print(f"   BFS 特点: 层次遍历，逐层访问，保证最短路径")
    print(f"   DFS 特点: 深入探索，沿路径走到底，再回溯")

    # 4. 应用: 最短路径查找
    print("\n4. 应用: 使用 BFS 查找最短路径")
    target_node = max(G.nodes())

    # NetworkX 最短路径
    shortest_path = nx.shortest_path(G, start_node, target_node)
    print(f"   从节点 {start_node} 到 {target_node} 的最短路径:")
    print(f"   {shortest_path}")
    print(f"   路径长度: {len(shortest_path) - 1}")


# ============================================================================
# 额外示例: 知识图谱三元组表示
# ============================================================================

def example_knowledge_graph_triples():
    """额外示例: 知识图谱的三元组表示"""
    print("\n" + "=" * 60)
    print("额外示例: 知识图谱的三元组表示")
    print("=" * 60)

    # 定义知识图谱的三元组
    knowledge_graph = [
        ("埃隆·马斯克", "CEO_of", "特斯拉"),
        ("埃隆·马斯克", "founder_of", "SpaceX"),
        ("埃隆·马斯克", "owner_of", "X (Twitter)"),
        ("特斯拉", "industry", "电动汽车"),
        ("特斯拉", "founded_in", "2003"),
        ("SpaceX", "industry", "航天"),
        ("SpaceX", "launched", "星舰"),
        ("星舰", "status", "测试中"),
    ]

    print("\n知识图谱三元组:")
    for i, (subj, rel, obj) in enumerate(knowledge_graph, 1):
        print(f"   {i}. {subj} --[{rel}]--> {obj}")

    # 创建图
    G = nx.MultiDiGraph()
    for subj, rel, obj in knowledge_graph:
        G.add_edge(subj, obj, relation=rel)

    print(f"\n图的统计:")
    print(f"   节点数: {G.number_of_nodes()}")
    print(f"   边数: {G.number_of_edges()}")

    # 查询示例
    print("\n查询示例:")

    # 1. 埃隆·马斯克的直接关系
    print("\n1. 埃隆·马斯克的直接关系:")
    if G.has_node("埃隆·马斯克"):
        neighbors = list(G.neighbors("埃隆·马斯克"))
        for neighbor in neighbors:
            edges_data = G.get_edge_data("埃隆·马斯克", neighbor)
            for edge in edges_data.values():
                print(f"   埃隆·马斯克 --[{edge['relation']}]--> {neighbor}")

    # 2. 多跳查询: 埃隆·马斯克 -> ??? -> ???
    print("\n2. 多跳查询 (埃隆·马斯克 -> 1步 -> 2步):")
    if G.has_node("埃隆·马斯克"):
        for node1 in G.neighbors("埃隆·马斯克"):
            for node2 in G.neighbors(node1):
                if node2 != "埃隆·马斯克":
                    edge1 = G.get_edge_data("埃隆·马斯克", node1)[0]['relation']
                    edge2 = G.get_edge_data(node1, node2)[0]['relation']
                    print(f"   埃隆·马斯克 --[{edge1}]--> {node1} --[{edge2}]--> {node2}")

    # 3. 路径查找
    print("\n3. 路径查找 (埃隆·马斯克 -> 星舰):")
    try:
        path = nx.shortest_path(G, "埃隆·马斯克", "星舰")
        print(f"   路径: {' -> '.join(path)}")

        # 显示关系
        path_with_relations = []
        for i in range(len(path) - 1):
            edge_data = G.get_edge_data(path[i], path[i+1])[0]
            path_with_relations.append(f"{path[i]} --[{edge_data['relation']}]--> {path[i+1]}")
        print(f"   完整路径:")
        for step in path_with_relations:
            print(f"     {step}")
    except nx.NetworkXNoPath:
        print("   未找到路径")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("模块 21: 图谱理论基础 - 实践示例")
    print("=" * 60)

    # 运行示例
    example_1_create_graphs()
    example_2_graph_properties()
    example_3_network_topology()
    example_4_graph_traversal()
    example_knowledge_graph_triples()

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
