import os
import re
import sys

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 需要修复的文件列表
files_to_fix = [
    "00-framework-introduction.html",
    "01-basic-chain-demo.html",
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

base_dir = "D:/workspaces/claudeworkspace/clanchain_study/langchain/"

for filename in files_to_fix:
    filepath = os.path.join(base_dir, filename)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修复截断的button标签
        original_content = content
        content = re.sub(r'</a>\n id="nextBtn">', '</a>\n            <button class="nav-btn next" id="nextBtn">', content)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ {filename} 修复完成")
        else:
            print(f"- {filename} 无需修复")

    except Exception as e:
        print(f"✗ {filename} 处理失败: {str(e)}")

print("\n所有文件修复完成！")
