import os
import re

# 更新所有Python文件中的年份
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 更新版权年份
                new_content = re.sub(r'© 2025', '© 2025', content)
                new_content = re.sub(r'2025 年', '2025 年', new_content)
                new_content = re.sub(r'2025-', '2025-', new_content)
                
                if new_content != content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"更新了: {filepath}")
            except:
                pass

print("年份更新完成！")
