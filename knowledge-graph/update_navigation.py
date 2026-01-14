#!/usr/bin/env python3
"""
批量更新知识图谱课程所有模块的导航链接
参照24-rdf-basics.html的标准导航格式
"""

import os
import re

# 定义所有模块及其上下文
modules = [
    {"num": 21, "file": "21-graph-theory.html", "prev": None, "next": "22-knowledge-representation.html"},
    {"num": 22, "file": "22-knowledge-representation.html", "prev": "21-graph-theory.html", "next": "23-graph-modeling.html"},
    {"num": 23, "file": "23-graph-modeling.html", "prev": "22-knowledge-representation.html", "next": "24-rdf-basics.html"},
    {"num": 24, "file": "24-rdf-basics.html", "prev": "23-graph-modeling.html", "next": "25-ontology-modeling.html"},
    {"num": 25, "file": "25-ontology-modeling.html", "prev": "24-rdf-basics.html", "next": "26-graph-databases.html"},
    {"num": 26, "file": "26-graph-databases.html", "prev": "25-ontology-modeling.html", "next": "27-sparql.html"},
    {"num": 27, "file": "27-sparql.html", "prev": "26-graph-databases.html", "next": "28-knowledge-extraction.html"},
    {"num": 28, "file": "28-knowledge-extraction.html", "prev": "27-sparql.html", "next": "29-graph-algorithms.html"},
    {"num": 29, "file": "29-graph-algorithms.html", "prev": "28-knowledge-extraction.html", "next": "30-kg-visualization.html"},
    {"num": 30, "file": "30-kg-visualization.html", "prev": "29-graph-algorithms.html", "next": "31-graphrag.html"},
    {"num": 31, "file": "31-graphrag.html", "prev": "30-kg-visualization.html", "next": "32-kg-applications.html"},
    {"num": 32, "file": "32-kg-applications.html", "prev": "31-graphrag.html", "next": "33-reasoning.html"},
    {"num": 33, "file": "33-reasoning.html", "prev": "32-kg-applications.html", "next": None},
]

# 标准导航模板
def generate_navigation(module_num, prev_file, next_file):
    """生成标准导航HTML"""

    # 上一模块部分
    if prev_file:
        prev_html = f'''                <a href="{prev_file}" class="text-gray-400 hover:text-white transition-colors flex items-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
                    </svg>
                    上一模块
                </a>'''
    else:
        prev_html = '''                <span class="text-gray-600">第一模块</span>'''

    # 下一模块部分
    if next_file:
        next_html = f'''                <a href="{next_file}" class="text-gray-400 hover:text-white transition-colors flex items-center gap-2">
                    下一模块
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                    </svg>
                </a>'''
    else:
        next_html = '''                <span class="text-gray-600">最后模块</span>'''

    navigation = f'''        <!-- Navigation -->
        <div class="nav-card">
            <div class="flex items-center justify-between flex-wrap gap-4">
                <a href="index.html" class="text-green-400 hover:text-green-300 transition-colors flex items-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
                    </svg>
                    返回课程目录
                </a>
                <div class="flex items-center gap-4">
{prev_html}
{next_html}
                </div>
                <div class="text-gray-400 text-sm">模块 {module_num} / 33</div>
            </div>
        </div>'''

    return navigation

def update_module_navigation(file_path, module_num, prev_file, next_file):
    """更新单个模块的导航"""

    print(f"正在处理: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 生成新的导航HTML
        new_nav = generate_navigation(module_num, prev_file, next_file)

        # 使用正则表达式替换导航部分
        # 匹配从 <!-- Navigation --> 到 </div> 之间的内容
        pattern = r'        <!-- Navigation -->.*?</div>\n\n        <!--'
        replacement = new_nav + '\n\n        <!--'

        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

        # 如果正则没有匹配到，尝试另一种模式
        if new_content == content:
            print(f"  [WARN] 未找到标准导航模式，尝试简单替换...")
            # 尝试找到 <div class="container"> 后的导航
            pattern2 = r'(<div class="container">\\s*<!-- Navigation -->\\s*<div class="nav-card">).*?(</div>\\s*</div>\\s*\\n)'
            match = re.search(pattern2, content, flags=re.DOTALL)

            if match:
                # 找到了，替换整个导航div
                new_content = content[:match.start()] + new_nav + '\\n' + content[match.end():]
                print(f"  [OK] 使用第二种模式替换成功")
            else:
                print(f"  [ERROR] 无法找到导航部分，跳过")
                return False

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"  [OK] 更新成功")
        return True

    except FileNotFoundError:
        print(f"  [ERROR] 文件不存在: {file_path}")
        return False
    except Exception as e:
        print(f"  [ERROR] 错误: {e}")
        return False

def main():
    base_dir = r"d:\workspaces\claudeworkspace\clanchain_study\knowledge-graph"

    print("=" * 60)
    print("开始批量更新所有模块的导航链接")
    print("=" * 60)
    print()

    success_count = 0
    fail_count = 0

    for module in modules:
        file_path = os.path.join(base_dir, module["file"])

        if os.path.exists(file_path):
            if update_module_navigation(file_path, module["num"], module["prev"], module["next"]):
                success_count += 1
            else:
                fail_count += 1
        else:
            print(f"[WARN] 文件不存在，跳过: {module['file']}")
            fail_count += 1

        print()

    print("=" * 60)
    print(f"更新完成!")
    print(f"成功: {success_count} 个模块")
    print(f"失败: {fail_count} 个模块")
    print("=" * 60)

if __name__ == "__main__":
    main()
