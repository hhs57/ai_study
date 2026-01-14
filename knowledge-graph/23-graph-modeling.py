"""
模块 23: 图数据建模原则
===================

本模块包含以下示例:
1. Property Graph 建模示例 (使用 NetworkX)
2. RDF Graph 建模示例 (使用 RDFlib)
3. 电商知识图谱建模案例

安装依赖:
pip install networkx matplotlib rdflib

运行示例:
python 23-graph-modeling.py
"""

import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph, Namespace, Literal, RDF, RDFS, XSD
from rdflib.namespace import FOAF
import json

# ============================================================================
# 示例 1: Property Graph 建模 - 社交网络
# ============================================================================

def example_1_property_graph_modeling():
    """示例 1: Property Graph 建模 - 社交网络"""
    print("=" * 70)
    print("示例 1: Property Graph 建模 - 社交网络")
    print("=" * 70)

    # 创建Property Graph (使用 NetworkX 模拟)
    G = nx.DiGraph()

    print("\n1. 定义节点和属性")
    print("-" * 70)

    # 用户节点 (带属性)
    users = [
        ("user1", {
            "label": "Person",
            "name": "张三",
            "age": 28,
            "city": "北京",
            "email": "zhangsan@example.com"
        }),
        ("user2", {
            "label": "Person",
            "name": "李四",
            "age": 32,
            "city": "上海",
            "email": "lisi@example.com"
        }),
        ("user3", {
            "label": "Person",
            "name": "王五",
            "age": 25,
            "city": "深圳",
            "email": "wangwu@example.com"
        })
    ]

    for user_id, attrs in users:
        G.add_node(user_id, **attrs)
        print(f"添加节点: {user_id} - {attrs['name']} ({attrs['city']})")

    print("\n2. 定义关系和属性")
    print("-" * 70)

    # 关系 (带属性)
    relationships = [
        ("user1", "user2", {
            "label": "KNOWS",
            "since": 2020,
            "strength": 0.8
        }),
        ("user2", "user3", {
            "label": "KNOWS",
            "since": 2021,
            "strength": 0.6
        }),
        ("user1", "user3", {
            "label": "FOLLOWS",
            "since": 2022
        })
    ]

    for source, target, attrs in relationships:
        G.add_edge(source, target, **attrs)
        print(f"添加关系: {G.nodes[source]['name']} " +
              f"--[{attrs['label']}]--> {G.nodes[target]['name']}")

    print("\n3. 查询示例")
    print("-" * 70)

    # 查询1: 找出张三的所有朋友
    print("\n查询1: 找出张三的所有朋友")
    zhangsan_friends = []
    for neighbor in G.successors("user1"):
        friend_name = G.nodes[neighbor]["name"]
        relation = G.edges["user1", neighbor]["label"]
        strength = G.edges["user1", neighbor].get("strength", "N/A")
        zhangsan_friends.append(f"{friend_name} (关系:{relation}, 强度:{strength})")

    print(f"张三的朋友: {', '.join(zhangsan_friends)}")

    # 查询2: 找出二度人脉 (朋友的朋友)
    print("\n查询2: 找出张三的二度人脉")
    friends_of_friends = set()
    for friend in G.successors("user1"):
        for fof in G.successors(friend):
            if fof != "user1":
                fof_name = G.nodes[fof]["name"]
                friends_of_friends.add(fof_name)

    print(f"张三的二度人脉: {', '.join(friends_of_friends)}")

    # 查询3: 找出最年长的用户
    print("\n查询3: 找出最年长的用户")
    oldest_user = max(G.nodes(data=True),
                     key=lambda x: x[1].get("age", 0))
    print(f"最年长的用户: {oldest_user[1]['name']}, {oldest_user[1]['age']}岁")

    # 可视化
    try:
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(12, 8))

        # 绘制节点
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            data = G.nodes[node]
            node_colors.append(data.get("city", "Unknown"))
            node_sizes.append(data.get("age", 25) * 20)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.7,
                              cmap=plt.cm.Set3)

        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color="gray",
                              width=2, alpha=0.6,
                              arrows=True, arrowsize=20,
                              arrowstyle='->')

        # 绘制节点标签
        node_labels = {node: G.nodes[node]["name"]
                      for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, node_labels,
                               font_size=12, font_weight='bold')

        # 绘制边标签
        edge_labels = {(u, v): G.edges[u, v]["label"]
                      for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                    font_size=9)

        plt.title("Property Graph 示例: 社交网络\n(节点大小=年龄, 节点颜色=城市)",
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('23-example1-property-graph.png', dpi=150, bbox_inches='tight')
        print("\n可视化已保存为: 23-example1-property-graph.png")
    except Exception as e:
        print(f"\n可视化失败: {e}")

    print("\n4. Property Graph 建模原则总结")
    print("-" * 70)
    print("✅ 原则1: 节点可以灵活定义属性")
    print("✅ 原则2: 边也可以有属性(如关系强度)")
    print("✅ 原则3: Schema灵活,无需预定义")
    print("✅ 原则4: 适合应用导向的快速开发")


# ============================================================================
# 示例 2: RDF Graph 建模 - 学术论文引用网络
# ============================================================================

def example_2_rdf_graph_modeling():
    """示例 2: RDF Graph 建模 - 学术论文引用网络"""
    print("\n" + "=" * 70)
    print("示例 2: RDF Graph 建模 - 学术论文引用网络")
    print("=" * 70)

    # 创建 RDF 图
    g = Graph()

    # 定义命名空间
    EX = Namespace("http://example.org/")
    g.bind("ex", EX)
    g.bind("foaf", FOAF)
    g.bind("rdfs", RDFS)

    print("\n1. 定义本体 (Schema)")
    print("-" * 70)

    # 定义类
    g.add((EX.Paper, RDF.type, RDFS.Class))
    g.add((EX.Author, RDF.type, RDFS.Class))
    g.add((EX.Conference, RDF.type, RDFS.Class))

    print("定义类: Paper, Author, Conference")

    # 定义属性
    g.add((EX.title, RDF.type, RDF.Property))
    g.add((EX.title, RDFS.domain, EX.Paper))
    g.add((EX.title, RDFS.range, XSD.string))

    g.add((EX.year, RDF.type, RDF.Property))
    g.add((EX.year, RDFS.domain, EX.Paper))
    g.add((EX.year, RDFS.range, XSD.integer))

    g.add((EX.cites, RDF.type, RDF.Property))
    g.add((EX.cites, RDFS.domain, EX.Paper))
    g.add((EX.cites, RDFS.range, EX.Paper))

    g.add((EX.writtenBy, RDF.type, RDF.Property))
    g.add((EX.writtenBy, RDFS.domain, EX.Paper))
    g.add((EX.writtenBy, RDFS.range, EX.Author))

    print("定义属性: title, year, cites, writtenBy")

    print("\n2. 添加实例数据")
    print("-" * 70)

    # 作者
    g.add((EX.author1, RDF.type, EX.Author))
    g.add((EX.author1, FOAF.name, Literal("张伟")))
    g.add((EX.author2, RDF.type, EX.Author))
    g.add((EX.author2, FOAF.name, Literal("李娜")))
    g.add((EX.author3, RDF.type, EX.Author))
    g.add((EX.author3, FOAF.name, Literal("王强")))

    # 论文
    g.add((EX.paper1, RDF.type, EX.Paper))
    g.add((EX.paper1, EX.title, Literal("深度学习在知识图谱中的应用")))
    g.add((EX.paper1, EX.year, Literal(2020, datatype=XSD.integer)))
    g.add((EX.paper1, EX.writtenBy, EX.author1))

    g.add((EX.paper2, RDF.type, EX.Paper))
    g.add((EX.paper2, EX.title, Literal("图神经网络综述")))
    g.add((EX.paper2, EX.year, Literal(2021, datatype=XSD.integer)))
    g.add((EX.paper2, EX.writtenBy, EX.author2))

    g.add((EX.paper3, RDF.type, EX.Paper))
    g.add((EX.paper3, EX.title, Literal("知识图谱推理技术")))
    g.add((EX.paper3, EX.year, Literal(2022, datatype=XSD.integer)))
    g.add((EX.paper3, EX.writtenBy, EX.author3))

    # 引用关系
    g.add((EX.paper2, EX.cites, EX.paper1))
    g.add((EX.paper3, EX.cites, EX.paper1))
    g.add((EX.paper3, EX.cites, EX.paper2))

    print("添加了3篇论文和3位作者")
    print("添加了引用关系: paper2→paper1, paper3→paper1, paper3→paper2")

    print("\n3. SPARQL 查询示例")
    print("-" * 70)

    # 查询1: 列出所有论文
    print("\n查询1: 列出所有论文")
    query1 = """
    PREFIX ex: <http://example.org/>

    SELECT ?paper ?title ?year
    WHERE {
        ?paper a ex:Paper ;
               ex:title ?title ;
               ex:year ?year .
    }
    ORDER BY DESC(?year)
    """

    results1 = g.query(query1)
    for row in results1:
        paper_id = row["paper"].split("/")[-1]
        print(f"  {paper_id}: {row['title']} ({row['year']})")

    # 查询2: 找出被引用最多的论文
    print("\n查询2: 找出被引用最多的论文")
    query2 = """
    PREFIX ex: <http://example.org/>

    SELECT ?paper ?title (COUNT(?citing) as ?citationCount)
    WHERE {
        ?paper a ex:Paper ;
               ex:title ?title .
        OPTIONAL { ?citing ex:cites ?paper . }
    }
    GROUP BY ?paper ?title
    ORDER BY DESC(?citationCount)
    """

    results2 = g.query(query2)
    for row in results2:
        paper_id = row["paper"].split("/")[-1]
        print(f"  {paper_id}: {row['title']} - 被引用{row['citationCount']}次")

    # 查询3: 找出作者的所有论文
    print("\n查询3: 找出张伟的所有论文")
    query3 = """
    PREFIX ex: <http://example.org/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?paper ?title ?year
    WHERE {
        ?author foaf:name "张伟" .
        ?paper ex:writtenBy ?author ;
               ex:title ?title ;
               ex:year ?year .
    }
    """

    results3 = g.query(query3)
    for row in results3:
        print(f"  {row['title']} ({row['year']})")

    # 序列化为不同格式
    print("\n4. RDF 序列化格式对比")
    print("-" * 70)

    print("\nTurtle 格式:")
    print(g.serialize(format='turtle').decode('utf-8')[:500] + "...")

    print("\nN-Triples 格式:")
    triples = list(g.triples((None, None, None)))[:3]
    for s, p, o in triples:
        print(f"  {s.n3()} {p.n3()} {o.n3()} .")

    print("\n5. RDF Graph 建模原则总结")
    print("-" * 70)
    print("✅ 原则1: 使用URI唯一标识资源")
    print("✅ 原则2: 基于标准本体 (RDFS, OWL)")
    print("✅ 原则3: 强类型,需要显式定义数据类型")
    print("✅ 原则4: 适合数据交换和互操作")


# ============================================================================
# 示例 3: 电商知识图谱建模
# ============================================================================

def example_3_ecommerce_kg_modeling():
    """示例 3: 电商知识图谱建模案例"""
    print("\n" + "=" * 70)
    print("示例 3: 电商知识图谱建模案例")
    print("=" * 70)

    # 创建多部图 (MultiDiGraph) 支持多重关系
    G = nx.MultiDiGraph()

    print("\n1. 建模设计")
    print("-" * 70)
    print("节点类型: User, Product, Category, Brand, Order, Review")
    print("关系类型:")
    print("  - BOUGHT: 用户购买商品")
    print("  - REVIEWED: 用户评价商品")
    print("  - BELONGS_TO: 商品属于分类")
    print("  - PRODUCED_BY: 商品由品牌生产")
    print("  - CONTAINS: 订单包含商品")

    print("\n2. 创建节点")
    print("-" * 70)

    # 用户
    users = [
        ("user1", {"label": "User", "name": "张三", "level": "VIP"}),
        ("user2", {"label": "User", "name": "李四", "level": "普通"}),
    ]

    # 商品
    products = [
        ("prod1", {"label": "Product", "name": "iPhone 15", "price": 6999}),
        ("prod2", {"label": "Product", "name": "MacBook Pro", "price": 14999}),
        ("prod3", {"label": "Product", "name": "AirPods", "price": 1299}),
    ]

    # 分类
    categories = [
        ("cat1", {"label": "Category", "name": "手机"}),
        ("cat2", {"label": "Category", "name": "电脑"}),
        ("cat3", {"label": "Category", "name": "耳机"}),
    ]

    # 品牌
    brands = [
        ("brand1", {"label": "Brand", "name": "Apple"}),
        ("brand2", {"label": "Brand", "name": "华为"}),
    ]

    # 订单
    orders = [
        ("order1", {"label": "Order", "id": "1001", "date": "2024-01-15"}),
        ("order2", {"label": "Order", "id": "1002", "date": "2024-01-20"}),
    ]

    # 评价
    reviews = [
        ("review1", {"label": "Review", "rating": 5, "comment": "非常好!"}),
        ("review2", {"label": "Review", "rating": 4, "comment": "不错"}),
    ]

    all_nodes = users + products + categories + brands + orders + reviews
    for node_id, attrs in all_nodes:
        G.add_node(node_id, **attrs)

    print(f"添加了 {len(G.nodes())} 个节点")

    print("\n3. 创建关系")
    print("-" * 70)

    relationships = [
        # 用户购买
        ("user1", "prod1", "BOUGHT", {"date": "2024-01-15"}),
        ("user1", "prod2", "BOUGHT", {"date": "2024-01-20"}),
        ("user2", "prod3", "BOUGHT", {"date": "2024-01-18"}),

        # 商品分类
        ("prod1", "cat1", "BELONGS_TO", {}),
        ("prod2", "cat2", "BELONGS_TO", {}),
        ("prod3", "cat3", "BELONGS_TO", {}),

        # 商品品牌
        ("prod1", "brand1", "PRODUCED_BY", {}),
        ("prod2", "brand1", "PRODUCED_BY", {}),

        # 订单包含
        ("order1", "prod1", "CONTAINS", {"quantity": 1}),
        ("order2", "prod2", "CONTAINS", {"quantity": 1}),

        # 用户评价
        ("user1", "review1", "WROTE", {}),
        ("review1", "prod1", "REVIEWS", {}),
        ("user2", "review2", "WROTE", {}),
        ("review2", "prod3", "REVIEWS", {}),
    ]

    for source, target, rel_type, attrs in relationships:
        G.add_edge(source, target, relation=rel_type, **attrs)
        source_name = G.nodes[source]["name"] if "name" in G.nodes[source] else source
        target_name = G.nodes[target]["name"] if "name" in G.nodes[target] else target
        print(f"  {source_name} --[{rel_type}]--> {target_name}")

    print(f"\n添加了 {len(G.edges())} 条关系")

    print("\n4. 典型查询场景")
    print("-" * 70)

    # 场景1: 个性化推荐 - 找相似商品
    print("\n场景1: 找与iPhone 15相关的商品")
    related_products = set()
    for neighbor in G.successors("prod1"):
        # 找到同一品牌或同一分类的商品
        for pred in G.predecessors(neighbor):
            if pred != "prod1" and G.nodes[pred]["label"] == "Product":
                related_products.add(G.nodes[pred]["name"])

    print(f"与iPhone 15相关的商品: {', '.join(related_products) if related_products else '无'}")

    # 场景2: 用户画像 - 分析用户偏好
    print("\n场景2: 张三的购买偏好分析")
    user1_purchases = list(G.successors("user1"))
    categories_bought = set()
    total_spent = 0

    for prod in user1_purchases:
        if G.nodes[prod]["label"] == "Product":
            # 找商品所属分类
            for cat in G.successors(prod):
                if G.edges[prod, cat]["relation"] == "BELONGS_TO":
                    categories_bought.add(G.nodes[cat]["name"])
            # 计算花费
            total_spent += G.nodes[prod]["price"]

    print(f"  张三购买过的分类: {', '.join(categories_bought)}")
    print(f"  总消费: ¥{total_spent}")

    # 场景3: 关联销售 - 购买iPhone的用户还买了什么
    print("\n场景3: 购买iPhone 15的用户还购买了什么")
    iphone_buyers = set()
    for user in G.predecessors("prod1"):
        if G.edges[user, prod1]["relation"] == "BOUGHT":
            iphone_buyers.add(user)

    other_products = {}
    for user in iphone_buyers:
        for prod in G.successors(user):
            if G.edges[user, prod]["relation"] == "BOUGHT" and prod != "prod1":
                prod_name = G.nodes[prod]["name"]
                other_products[prod_name] = other_products.get(prod_name, 0) + 1

    print("  购买iPhone 15的用户还购买了:")
    for prod, count in sorted(other_products.items(), key=lambda x: -x[1]):
        print(f"    {prod}: {count}人")

    # 可视化
    try:
        pos = nx.spring_layout(G, seed=42, k=1.5)
        plt.figure(figsize=(14, 10))

        # 按类型绘制节点
        type_colors = {
            "User": "#3b82f6",
            "Product": "#8b5cf6",
            "Category": "#22c55e",
            "Brand": "#f59e0b",
            "Order": "#ec4899",
            "Review": "#06b6d4"
        }

        for node_type, color in type_colors.items():
            nodes_of_type = [n for n, d in G.nodes(data=True)
                           if d["label"] == node_type]
            if nodes_of_type:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type,
                                     node_color=color, label=node_type,
                                     node_size=500, alpha=0.7)

        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color="gray",
                              width=1, alpha=0.5,
                              arrows=True, arrowsize=15)

        # 绘制节点标签
        node_labels = {n: d.get("name", n) for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, node_labels,
                               font_size=8, font_weight='bold')

        plt.legend(loc='upper left')
        plt.title("电商知识图谱 - 节点按类型着色",
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('23-example3-ecommerce-kg.png', dpi=150, bbox_inches='tight')
        print("\n可视化已保存为: 23-example3-ecommerce-kg.png")
    except Exception as e:
        print(f"\n可视化失败: {e}")

    print("\n5. 电商KG建模最佳实践")
    print("-" * 70)
    print("✅ 实践1: 为高频查询路径建立索引")
    print("✅ 实践2: 避免超级节点(如热门商品)")
    print("✅ 实践3: 使用关系属性存储交易信息")
    print("✅ 实践4: 定期清理无关系的孤立节点")
    print("✅ 实践5: 为推荐算法预留评分字段")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("模块 23: 图数据建模原则 - 实践示例")
    print("=" * 70)

    # 运行示例
    example_1_property_graph_modeling()
    example_2_rdf_graph_modeling()
    example_3_ecommerce_kg_modeling()

    print("\n" + "=" * 70)
    print("所有示例运行完成!")
    print("=" * 70)
    print("\n核心要点:")
    print("  1. Property Graph: 灵活、易用、适合应用开发")
    print("  2. RDF Graph: 标准化、可互操作、适合数据交换")
    print("  3. 建模要从查询出发,不要过度设计")
    print("  4. 好的建模能让查询简单高效")
    print("  5. 定期评估和优化数据模型")
    print("=" * 70)


if __name__ == "__main__":
    main()
