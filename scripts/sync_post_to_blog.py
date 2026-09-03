import os
import re
import shutil

post_src = '/home/jkk/sdf_reverse_perform/docs/post.md'
post_dst = '/home/jkk/jedrzej-kolbert.github.io/_posts/2026-07-21-training-a-false-belief-harder-doesnt-make-it-harder-to-undo.md'
figures_dir = '/home/jkk/sdf_reverse_perform/docs/figures'
assets_dir = '/home/jkk/jedrzej-kolbert.github.io/assets/img'

with open(post_src, 'r', encoding='utf-8') as f:
    content = f.read()

# Copy figure files to assets/img
img_matches = re.findall(r'!\[([^\]]*)\]\((https://raw\.githubusercontent\.com/[^)]+/docs/figures/([^)]+))\)', content)
copied_count = 0
for alt, url, filename in img_matches:
    src_path = os.path.join(figures_dir, filename)
    dst_path = os.path.join(assets_dir, filename)
    if os.path.exists(src_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copied_count += 1

def img_replacer(match):
    alt_text = match.group(1)
    filename = match.group(3)
    return f'![{alt_text}](' + '{{ "/assets/img/' + filename + '" | relative_url }})'

content = re.sub(
    r'!\[([^\]]*)\]\((https://raw\.githubusercontent\.com/[^)]+/docs/figures/([^)]+))\)',
    img_replacer,
    content
)

# Fix tables for Kramdown: ensure a blank line exists before a markdown table header line starting with '|'
lines = content.splitlines()
fixed_lines = []
for i, line in enumerate(lines):
    if line.strip().startswith('|') and i > 0 and lines[i-1].strip() != '' and not lines[i-1].strip().startswith('|'):
        fixed_lines.append('')
    fixed_lines.append(line)

front_matter = """---
layout: post
title: "Training a False Belief Harder Doesn't Make It Harder to Undo"
date: 2026-07-21
---

"""

final_content = front_matter + '\n'.join(fixed_lines) + '\n'

with open(post_dst, 'w', encoding='utf-8') as f:
    f.write(final_content)

print(f"Successfully updated {post_dst} and copied {copied_count} figures to {assets_dir}")
