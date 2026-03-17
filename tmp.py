import json

with open("dirty_region_caches/gemma-3-270m-cpp.jsonl", "r", encoding="utf-8") as f:
    text = f.readline()
    data = json.loads(text)
    print(data)