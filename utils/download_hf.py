import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


DEFAULT_MODEL_DIR = "./models"


def createModelsSymlink(model_dir: Path, repo_id: str) -> None:
    target_path = model_dir / repo_id
    models_repo_path = Path("models") / repo_id
    if target_path.resolve() == models_repo_path.resolve():
        return
    validateModelsSymlink(model_dir, repo_id)

    if models_repo_path.is_symlink():
        if models_repo_path.resolve() == target_path.resolve():
            return
        models_repo_path.unlink()
    models_repo_path.parent.mkdir(parents=True, exist_ok=True)
    models_repo_path.symlink_to(target_path.resolve(), target_is_directory=True)


def validateModelsSymlink(model_dir: Path, repo_id: str) -> None:
    target_path = model_dir / repo_id
    models_repo_path = Path("models") / repo_id
    if (
        target_path.resolve() != models_repo_path.resolve()
        and models_repo_path.exists()
        and not models_repo_path.is_symlink()
    ):
        raise RuntimeError(
            f"Cannot link '{models_repo_path}' to '{target_path}': the destination "
            "already exists and is not a symlink. Move it or remove it, then run "
            "this script again."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model.")
    parser.add_argument("repo_id", help="Hugging Face repository ID to download")
    args = parser.parse_args()

    load_dotenv()
    model_dir = Path(os.getenv("MODEL_DIR") or DEFAULT_MODEL_DIR).expanduser()
    local_dir = model_dir / args.repo_id
    validateModelsSymlink(model_dir, args.repo_id)
    print(f"Downloading {args.repo_id} to {local_dir}")
    snapshot_download(repo_id=args.repo_id, local_dir=local_dir)
    createModelsSymlink(model_dir, args.repo_id)


if __name__ == "__main__":
    main()
