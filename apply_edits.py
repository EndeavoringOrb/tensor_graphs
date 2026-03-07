import json
import os
import sys

def apply_edits_from_json(json_file_path):
    """Reads a JSON file and applies the file edits contained within."""
    print(f"--- Processing: {json_file_path} ---")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Could not read or parse {json_file_path}: {e}")
        return

    # Handle both a single edit object or a list of edit objects
    edits = data if isinstance(data, list) else [data]

    for i, edit in enumerate(edits):
        # Support both raw argument dicts or nested "arguments" keys from tool call logs
        args = edit.get("arguments", edit)
        
        filepath = args.get("filepath")
        search_pattern = args.get("search_pattern")
        replacement = args.get("replacement")

        if not all([filepath, search_pattern, replacement]):
            print(f"Skipping item {i}: Missing required fields (filepath, search_pattern, or replacement).")
            continue

        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            if search_pattern not in content:
                print(f"Warning in {filepath}: Search pattern not found. Skipping.")
                continue

            # Apply the replacement
            new_content = content.replace(search_pattern, replacement)

            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                f.write(new_content)
            
            print(f"Successfully applied edit to: {filepath}")

        except Exception as e:
            print(f"Failed to edit {filepath}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python apply_edits.py <file1.json> <file2.json> ...")
        sys.exit(1)

    for json_path in sys.argv[1:]:
        apply_edits_from_json(json_path)