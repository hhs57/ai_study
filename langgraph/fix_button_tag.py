import os
import re
import sys

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 需要修复的文件列表（langgraph 文件夹）
files_to_fix = [
    "11-langgraph-intro-demo.html",
    "12-langgraph-state-demo.html",
    "13-langgraph-conditional-demo.html",
    "14-langgraph-loops-demo.html",
    "15-langgraph-agent-demo.html",
    "16-human-in-loop-demo.html",
    "17-state-persistence-demo.html",
    "18-multi-agent-demo.html",
    "19-visualization-debug-demo.html",
    "20-error-handling-demo.html"
]

base_dir = "D:/workspaces/claudeworkspace/clanchain_study/langgraph/"

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
