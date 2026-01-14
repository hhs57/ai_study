"""
模块 22: 知识表示理论
=====================

本模块包含以下示例:
1. 关系型数据库 vs 图数据库查询对比
2. 知识图谱三元组表示和推理
3. DIKW 模型示例

安装依赖:
pip install networkx matplotlib

运行示例:
python 22-knowledge-representation.py
"""

import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================================
# 示例 1: 关系型数据库 vs 图数据库查询对比
# ============================================================================

def example_1_relational_vs_graph():
    """示例 1: 关系型数据库 vs 图数据库查询对比"""
    print("=" * 70)
    print("示例 1: 关系型数据库 vs 图数据库查询对比")
    print("=" * 70)

    # 场景: 查找"张三"的朋友的朋友
    print("\n场景: 查找'张三'的朋友的朋友（2跳关系）\n")

    # 1. 关系型数据库方式（概念展示）
    print("1. 关系型数据库（SQL）")
    print("-" * 70)
    print("-- 数据表结构:")
    print("-- CREATE TABLE users (id INT, name VARCHAR(50));")
    print("-- CREATE TABLE friendships (user_id INT, friend_id INT);")
    print()
    sql_query = """
-- SQL 查询（需要多次 JOIN）
SELECT DISTINCT u2.name
FROM users u1
JOIN friendships f1 ON u1.id = f1.user_id
JOIN friendships f2 ON f1.friend_id = f2.user_id
JOIN users u2 ON f2.friend_id = u2.id
WHERE u1.name = '张三'
  AND u2.name != '张三';
    """
    print(sql_query)
    print("\n问题:")
    print("  - 需要 3 次 JOIN 操作")
    print("  - 查询复杂度高")
    print("  - 跳数增加时复杂度呈指数增长")
    print("  - 难以处理变长路径查询")

    # 2. 图数据库方式（概念展示）
    print("\n2. 图数据库（Cypher）")
    print("-" * 70)
    print("-- 数据模型: (人物:张三)-[:朋友]->(人物:李四)")
    print()
    cypher_query = """
-- Cypher 查询（直观的路径匹配）
MATCH (p:Person {name: '张三'})-[:朋友]->()-[:朋友]->(fof:Person)
RETURN DISTINCT fof.name
    """
    print(cypher_query)
    print("\n优势:")
    print("  - 查询语言直观，接近图思维")
    print("  - 性能优化（图遍历 vs 表连接）")
    print("  - 易于扩展：查询 3 跳、4 跳关系")
    print("  - 支持变长路径：[:朋友*1..3]")

    # 3. 实际代码对比（使用 NetworkX 模拟）
    print("\n3. 实际代码对比")
    print("-" * 70)

    # 创建社交网络图
    G = nx.Graph()
    friendships = [
        ("张三", "李四"),
        ("张三", "王五"),
        ("李四", "赵六"),
        ("李四", "孙七"),
        ("王五", "赵六"),
        ("赵六", "周八"),
        ("孙七", "周八"),
    ]
    G.add_edges_from(friendships)

    print("\n社交网络数据:")
    for i, (p1, p2) in enumerate(friendships, 1):
        print(f"  {i}. {p1} - 朋友 - {p2}")

    # Python 字典模拟关系型数据库查询
    print("\n[方式1] Python 字典（模拟关系型数据库）:")
    print("-" * 70)

    # 构建邻接表
    friends_dict = defaultdict(list)
    for p1, p2 in friendships:
        friends_dict[p1].append(p2)
        friends_dict[p2].append(p1)

    print("数据结构:", dict(friends_dict))
    print("\n查询'张三'的朋友的朋友:")
    print("friends_of_friends = set()")
    print("for friend1 in friends_dict['张三']:")
    print("    for friend2 in friends_dict.get(friend1, []):")
    print("        if friend2 != '张三':")
    print("            friends_of_friends.add(friend2)")

    friends_of_friends = set()
    for friend1 in friends_dict['张三']:
        for friend2 in friends_dict.get(friend1, []):
            if friend2 != '张三':
                friends_of_friends.add(friend2)

    print(f"\n结果: {sorted(friends_of_friends)}")
    print("\n代码特点: 需要嵌套循环，跳数增加时代码复杂度增加")

    # NetworkX 模拟图数据库查询
    print("\n[方式2] NetworkX（模拟图数据库）:")
    print("-" * 70)

    print("查询'张三'的朋友的朋友:")
    print("fof_nodes = []")
    print("for node in G.nodes():")
    print("    if nx.has_path(G, '张三', node):")
    print("        path = nx.shortest_path(G, '张三', node)")
    print("        if len(path) == 3:")
    print("            fof_nodes.append(path[-1])")

    fof_nodes = []
    for node in G.nodes():
        if nx.has_path(G, '张三', node):
            path = nx.shortest_path(G, '张三', node)
            if len(path) == 3:
                fof_nodes.append(path[-1])

    print(f"\n结果: {sorted(set(fof_nodes))}")
    print("\n代码特点: 利用图算法，代码更简洁，语义更清晰")

    # 可视化
    try:
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10, 6))

        # 高亮张三的朋友的朋友
        colors = []
        sizes = []
        for node in G.nodes():
            if node == '张三':
                colors.append('#ff6b6b')
                sizes.append(800)
            elif node in friends_dict['张三']:
                colors.append('#4ecdc4')
                sizes.append(600)
            elif node in friends_of_friends:
                colors.append('#ffe66d')
                sizes.append(600)
            else:
                colors.append('#95a5a6')
                sizes.append(400)

        nx.draw(G, pos, with_labels=True, node_color=colors,
                node_size=sizes, font_size=12, font_weight='bold',
                edge_color='gray', width=2, alpha=0.7)

        plt.title("社交网络可视化\n红色:张三 | 青色:1跳朋友 | 黄色:2跳朋友",
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('22-example1-social-network.png', dpi=150, bbox_inches='tight')
        print("\n可视化已保存为: 22-example1-social-network.png")
    except Exception as e:
        print(f"\n可视化失败: {e}")

    print("\n总结:")
    print("  - 关系型数据库: 适合结构化查询，多表连接复杂")
    print("  - 图数据库: 天然支持关系查询，路径查询简洁高效")


# ============================================================================
# 示例 2: 知识图谱三元组表示和推理
# ============================================================================

def example_2_knowledge_graph_triples():
    """示例 2: 知识图谱的三元组表示和推理"""
    print("\n" + "=" * 70)
    print("示例 2: 知识图谱的三元组表示和推理")
    print("=" * 70)

    # 1. 三元组表示
    print("\n1. 三元组表示 (主语 - 谓语 - 宾语)")
    print("-" * 70)

    knowledge_graph = [
        ("苏格拉底", "is_a", "人"),
        ("人", "is_mortal", "凡人"),
        ("柏拉图", "is_a", "人"),
        ("柏拉图", "teacher_of", "亚里士多德"),
        ("亚里士多德", "is_a", "人"),
        ("苏格拉底", "teacher_of", "柏拉图"),
    ]

    print("知识图谱三元组:")
    for i, (subj, pred, obj) in enumerate(knowledge_graph, 1):
        print(f"  {i}. {subj} --[{pred}]--> {obj}")

    # 2. 创建图结构
    print("\n2. 构建图结构")
    print("-" * 70)

    G = nx.MultiDiGraph()
    for subj, pred, obj in knowledge_graph:
        G.add_edge(subj, obj, relation=pred)

    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")
    print(f"节点: {list(G.nodes())}")

    # 3. 简单推理示例
    print("\n3. 简单推理（演绎推理）")
    print("-" * 70)

    print("推理规则:")
    print("  IF X is_a 人 AND 人 is_mortal 凡人")
    print("  THEN X is_mortal 凡人")

    print("\n推理过程:")
    for node in G.nodes():
        # 检查是否是"人"
        is_human = False
        for _, _, data in G.out_edges(node, data=True):
            if data['relation'] == 'is_a':
                target = _
                if target == '人':
                    is_human = True
                    break

        # 如果是"人"，推导出"is_mortal 凡人"
        if is_human:
            mortal_edge_exists = False
            for _, _, data in G.out_edges(node, data=True):
                if data['relation'] == 'is_mortal':
                    mortal_edge_exists = True
                    break

            if not mortal_edge_exists:
                print(f"  推导: {node} is_mortal 凡人 (从 '人 is_mortal 凡人' 继承)")

    # 4. 路径查询
    print("\n4. 路径查询 (苏格拉底 -> 亚里士多德)")
    print("-" * 70)

    try:
        path = nx.shortest_path(G, '苏格拉底', '亚里士多德')
        print(f"最短路径: {' -> '.join(path)}")

        # 显示完整关系
        print("\n详细路径:")
        for i in range(len(path) - 1):
            edge_data = G.get_edge_data(path[i], path[i+1])
            relation = edge_data[0]['relation']
            print(f"  {path[i]} --[{relation}]--> {path[i+1]}")

        print(f"\n路径长度: {len(path) - 1} 跳")

    except nx.NetworkXNoPath:
        print("未找到路径")

    # 5. 关系查询
    print("\n5. 关系查询")
    print("-" * 70)

    # 查询柏拉图的所有关系
    print("柏拉图的所有关系:")
    if G.has_node('柏拉图'):
        # 出边（柏拉图 -> ?）
        print("  出边:")
        for _, target, data in G.out_edges('柏拉图', data=True):
            print(f"    柏拉图 --[{data['relation']}]--> {target}")

        # 入边（? -> 柏拉图）
        print("  入边:")
        for source, _, data in G.in_edges('柏拉图', data=True):
            print(f"    {source} --[{data['relation']}]--> 柏拉图")

    # 可视化
    try:
        pos = nx.spring_layout(G, seed=42, k=2)
        plt.figure(figsize=(12, 8))

        # 绘制节点
        node_colors = []
        for node in G.nodes():
            if node == '苏格拉底':
                node_colors.append('#ff6b6b')
            elif node == '柏拉图':
                node_colors.append('#4ecdc4')
            elif node == '亚里士多德':
                node_colors.append('#ffe66d')
            else:
                node_colors.append('#95e1d3')

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=1500, alpha=0.9)

        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                              width=2, alpha=0.6, arrows=True,
                              arrowsize=20, arrowstyle='->')

        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=10,
                               font_weight='bold')

        # 绘制边标签
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            edge_labels[(u, v)] = data['relation']
        nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                     font_size=8)

        plt.title("知识图谱三元组可视化",
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('22-example2-knowledge-graph.png', dpi=150, bbox_inches='tight')
        print("\n可视化已保存为: 22-example2-knowledge-graph.png")
    except Exception as e:
        print(f"\n可视化失败: {e}")


# ============================================================================
# 示例 3: DIKW 模型演示
# ============================================================================

def example_3_dikw_model():
    """示例 3: DIKW 模型（数据-信息-知识-智慧）"""
    print("\n" + "=" * 70)
    print("示例 3: DIKW 模型演示")
    print("=" * 70)

    print("\nDIKW 层次结构:")
    print("-" * 70)

    # 数据
    data = [25, 30, 35, 28, 32, 27, 33]
    print(f"\n1. 数据 (Data)")
    print(f"   原始数值: {data}")
    print(f"   说明: 未处理的原始事实和观察结果")

    # 信息
    print(f"\n2. 信息 (Information)")
    avg = sum(data) / len(data)
    max_val = max(data)
    min_val = min(data)
    print(f"   平均值: {avg:.1f}")
    print(f"   最大值: {max_val}")
    print(f"   最小值: {min_val}")
    print(f"   说明: 经过处理、有组织的数据")

    # 知识
    print(f"\n3. 知识 (Knowledge)")
    print(f"   规律: 温度范围在 25-35°C 之间")
    print(f"   规律: 平均温度约 30°C")
    print(f"   规律: 存在日间波动模式")
    print(f"   说明: 对信息的理解和模式识别")

    # 智慧
    print(f"\n4. 智慧 (Wisdom)")
    print(f"   判断: 该环境适合人类居住")
    print(f"   决策: 无需额外降温或加热设备")
    print(f"   洞察: 基于知识做出的明智判断")
    print(f"   说明: 运用知识做出决策和判断")

    print("\nDIKW 转化过程:")
    print("-" * 70)
    print("数据 → 信息: 通过整理和计算（求平均值、最大/最小值）")
    print("信息 → 知识: 通过模式识别和规律总结")
    print("知识 → 智慧: 通过应用和决策")

    # 知识图谱中的 DIKW
    print("\n知识图谱中的 DIKW:")
    print("-" * 70)
    print("数据层: RDF 三元组（主语-谓语-宾语）")
    print("信息层: 结构化的本体和关系")
    print("知识层: 推理得出的新关系和规则")
    print("智慧层: 基于知识的智能决策和推荐")

    print("\n示例:")
    print("  数据: (巴黎, capital_of, 法国)")
    print("  信息: 法国首都人口 200 万")
    print("  知识: 法国 → 欧盟成员国 → 使用欧元")
    print("  智慧: 去巴黎旅游需要准备欧元货币")


# ============================================================================
# 额外示例: 知识图谱类型对比
# ============================================================================

def example_knowledge_graph_types():
    """额外示例: 不同类型的知识图谱对比"""
    print("\n" + "=" * 70)
    print("额外示例: 知识图谱类型对比")
    print("=" * 70)

    kg_types = {
        "开放知识图谱": {
            "特点": ["公开可访问", "社区维护", "大规模"],
            "代表": ["Wikidata", "DBpedia", "YAGO"],
            "规模": "百万至十亿级实体",
            "应用": "通用知识查询、链接数据"
        },
        "企业知识图谱": {
            "特点": ["私有化部署", "领域特定", "高质量"],
            "代表": ["Google KG", "百度KG", "阿里巴巴KG"],
            "规模": "千万级实体",
            "应用": "搜索增强、推荐系统"
        },
        "领域知识图谱": {
            "特点": ["垂直领域", "深度专业", "精确建模"],
            "代表": ["医疗KG", "法律KG", "金融KG"],
            "规模": "百万级实体",
            "应用": "专业问答、决策支持"
        },
        "多模态知识图谱": {
            "特点": ["文本+图像", "跨模态关联", "语义丰富"],
            "代表": ["Visual Genome", "MS-COCO KG"],
            "规模": "百万级实体",
            "应用": "图像理解、跨模态检索"
        }
    }

    for kg_type, info in kg_types.items():
        print(f"\n{kg_type}")
        print("-" * 70)
        print(f"  特点: {', '.join(info['特点'])}")
        print(f"  代表: {', '.join(info['代表'])}")
        print(f"  规模: {info['规模']}")
        print(f"  应用: {info['应用']}")

    print("\n选择建议:")
    print("-" * 70)
    print("  学习和实验: 使用开放知识图谱（Wikidata、DBpedia）")
    print("  企业应用: 构建企业知识图谱")
    print("  专业领域: 构建领域知识图谱")
    print("  前沿研究: 探索多模态知识图谱")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("模块 22: 知识表示理论 - 实践示例")
    print("=" * 70)

    # 运行示例
    example_1_relational_vs_graph()
    example_2_knowledge_graph_triples()
    example_3_dikw_model()
    example_knowledge_graph_types()

    print("\n" + "=" * 70)
    print("所有示例运行完成!")
    print("=" * 70)
    print("\n核心要点:")
    print("  1. 知识图谱用图结构表示知识（节点=实体，边=关系）")
    print("  2. 图数据库在关系查询上比关系型数据库更高效")
    print("  3. 三元组是知识图谱的基本单元（主语-谓语-宾语）")
    print("  4. DIKW 模型描述了从数据到智慧的层次结构")
    print("  5. 不同类型的知识图谱适用于不同场景")
    print("=" * 70)


if __name__ == "__main__":
    main()
