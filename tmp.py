import json

with open("tmp.json", "r", encoding="utf-8") as f:
    data = json.load(f)
text = data["replacement"]
parts = text.split("====")
old = parts[0][4:]
new = parts[1][:-4]
with open("build.py", "r", encoding="utf-8") as f:
    text = f.read()
new_text = text.replace(old, new)
if text != new_text:
    with open("build.py", "w", encoding="utf-8") as f:
        f.write(new_text)