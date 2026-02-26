import os
import sys
import argparse
import shutil

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, GatedRepoError


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_registry():
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from app.config import MODEL_REGISTRY  # pylint: disable=import-error

    return MODEL_REGISTRY


def _download_one(entry: dict, models_dir: str) -> str:
    os.makedirs(models_dir, exist_ok=True)
    target_path = os.path.join(models_dir, entry["filename"])
    if os.path.exists(target_path):
        return target_path

    repo_id = entry["hf_repo"]
    hf_file = entry["hf_file"]

    try:
        cached_path = hf_hub_download(repo_id=repo_id, filename=hf_file)
    except EntryNotFoundError as e:
        raise RuntimeError(
            f"HF file not found: repo={repo_id} file={hf_file} (fix MODEL_REGISTRY)"
        ) from e
    except RepositoryNotFoundError as e:
        raise RuntimeError(
            f"HF repo not found: repo={repo_id} (fix MODEL_REGISTRY)"
        ) from e
    except GatedRepoError as e:
        raise RuntimeError(
            f"HF repo is gated (needs token / access): repo={repo_id}"
        ) from e

    shutil.copyfile(cached_path, target_path)
    return target_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "./models"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--missing-only", action="store_true", default=True)
    args = parser.parse_args()

    registry = _load_registry()
    if not registry:
        print("No models in registry")
        return 1

    if args.model_id:
        targets = [m for m in registry if m.get("id") == args.model_id]
        if not targets:
            print(f"Unknown model_id: {args.model_id}")
            return 2
    elif args.all:
        targets = list(registry)
    else:
        targets = [m for m in registry if m.get("default")] or [registry[0]]

    if args.missing_only:
        targets = [m for m in targets if not os.path.exists(os.path.join(args.models_dir, m["filename"]))]

    if not targets:
        print("Nothing to download")
        return 0

    for i, m in enumerate(targets, start=1):
        name = m.get("name") or m.get("id")
        print(f"[{i}/{len(targets)}] {name} -> {m['filename']}")
        path = _download_one(m, args.models_dir)
        size = os.path.getsize(path)
        print(f"Saved: {path} ({size} bytes)")

    print("Done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
