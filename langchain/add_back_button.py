import os
import re
import sys

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 需要更新的文件列表
files_to_update = [
    "02-prompt-template-demo.html",
    "03-chains-sequentials-demo.html",
    "04-memory-conversation-demo.html",
    "05-agents-basic-demo.html",
    "06-document-loading-demo.html",
    "07-vector-storage-rag-demo.html",
    "08-output-parsers-advanced-demo.html",
    "09-callbacks-streaming-demo.html",
    "10-complete-rag-app-demo.html"
]

# 新的HTML按钮代码
new_button_html = '''            <a href="index.html" class="nav-btn back-to-list">
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h7"/>
                </svg>
                课程列表
            </a>'''

# CSS样式代码
new_css = '''        /* 针对 a 标签的导航按钮样式 */
        .nav-btn.back-to-list {
            text-decoration: none;
            background: rgba(59, 130, 246, 0.2);
            border-color: rgba(59, 130, 246, 0.5);
            color: #60a5fa;
        }

        .nav-btn.back-to-list:hover {
            background: rgba(59, 130, 246, 0.4);
            box-shadow: 0 5px 20px rgba(59, 130, 246, 0.4);
        }

'''

base_dir = "D:/workspaces/claudeworkspace/clanchain_study/langchain/"

for filename in files_to_update:
    filepath = os.path.join(base_dir, filename)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否已经添加了返回按钮
        if 'back-to-list' in content:
            print(f"✓ {filename} 已经包含返回按钮，跳过")
            continue

        # 1. 在HTML中添加返回按钮
        # 查找 "上一节" 按钮后，在 "下一节" 按钮前插入
        pattern1 = r'(                上一节\n            </button>)\n(            <button class="nav-btn next")'
        replacement1 = r'\1\n' + new_button_html + '\n\2'

        if re.search(pattern1, content):
            content = re.sub(pattern1, replacement1, content)
            print(f"✓ {filename} HTML部分已更新")
        else:
            print(f"✗ {filename} HTML部分未找到匹配模式")
            continue

        # 2. 在CSS中添加样式
        # 查找 .nav-btn:hover:not(:disabled) 后面添加新样式
        pattern2 = r'(        \.nav-btn:hover:not\(:disabled\) \{[^}]+\}\n\n)'
        if re.search(pattern2, content):
            content = re.sub(pattern2, r'\1' + new_css, content)
            print(f"✓ {filename} CSS部分已更新")
        else:
            print(f"✗ {filename} CSS部分未找到匹配模式")
            continue

        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ {filename} 更新完成\n")

    except Exception as e:
        print(f"✗ {filename} 处理失败: {str(e)}\n")

print("所有文件处理完成！")
