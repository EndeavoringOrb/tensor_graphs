import os
import argparse


def parse_patches(patch_file):
    with open(patch_file, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = content.split("====")
    if len(chunks) % 2 != 0:
        raise ValueError("Patch file must contain pairs of old/new separated by ====")

    patches = []
    for i in range(0, len(chunks), 2):
        old = chunks[i].strip("\n")
        new = chunks[i + 1].strip("\n")
        patches.append((old, new))

    return patches


def process_file(path, patches, dry_run=False):
    try:
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()
    except Exception:
        return False

    modified = original
    any_change = False

    for idx, (old, new) in enumerate(patches):
        count = modified.count(old)

        if count == 1:
            print(f"[APPLY] {path} (patch #{idx})")
            modified = modified.replace(old, new)
            any_change = True
        elif count > 1:
            print(f"[SKIP >1] {path} (patch #{idx}, matches={count})")
        else:
            # count == 0 → ignore silently or log if you want
            pass

    if any_change and not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            f.write(modified)

    return any_change


def walk_and_apply(root, patches, extensions=None, dry_run=False):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if extensions:
                if not any(name.endswith(ext) for ext in extensions):
                    continue

            full_path = os.path.join(dirpath, name)
            process_file(full_path, patches, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("patch_file")
    parser.add_argument("--ext", nargs="*", default=[".cpp", ".hpp", ".cu"])
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    patches = parse_patches(args.patch_file)
    walk_and_apply(args.directory, patches, args.ext, args.dry_run)