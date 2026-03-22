import os
import re

kernels_dir = r"tensor_graphs_cpp/kernels"

for root, dirs, files in os.walk(kernels_dir):
    for file in files:
        if file.endswith(('.hpp', '.cu')):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            
            new_content = content
            
            # Remove the std::vector<Backend>(...) wrapper since we won't need it.
            new_content = re.sub(r'std::vector<Backend>\((\{.*?\})\)', r'\1', new_content)
            
            # 1. REGISTER_REF_KERNEL(OpType, {backends}, match, run)
            # Find REGISTER_REF_KERNEL(..., { ... }, match, run)
            new_content = re.sub(
                r'REGISTER_REF_KERNEL\(\s*(OpType::[\w_]+)\s*,\s*(\{[^}]+\})\s*,\s*([\w_]+)\s*,\s*([\w_]+)\s*\)',
                r'REGISTER_REF_KERNEL(\1, \3, \4, \2)',
                new_content
            )

            # 2. REGISTER_KERNEL and REGISTER_KERNEL_INPLACE
            # REGISTER_KERNEL("...", N, {backends}, match, run, ref, ...)
            new_content = re.sub(
                r'(REGISTER_KERNEL(?:_INPLACE)?)\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*(\{[^}]+\})\s*,\s*([\w_]+)\s*,\s*([\w_]+)\s*,\s*([\w_]+)\s*,',
                r'\1(\2, \3, \5, \6, \7, \4,',
                new_content
            )
            
            # 3. REGISTER_KERNEL_INPLACE_VIEW
            # REGISTER_KERNEL_INPLACE_VIEW("...", N, {backends}, match, run, ref, inview, ...)
            new_content = re.sub(
                r'(REGISTER_KERNEL_INPLACE_VIEW)\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*(\{[^}]+\})\s*,\s*([\w_]+)\s*,\s*([\w_]+)\s*,\s*([\w_]+)\s*,\s*([\w_]+)\s*,',
                r'\1(\2, \3, \5, \6, \7, \8, \4,',
                new_content
            )

            if new_content != content:
                print(f"Updating {path}")
                with open(path, 'w') as f:
                    f.write(new_content)
