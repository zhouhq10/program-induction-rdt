"""
publish_hf.py — Build the demo and upload it to a Hugging Face Static Space.

    pip install huggingface_hub
    hf auth login                    # once, with a token that has write access
    python demo/publish_hf.py        # → <your username>/program-induction-rdt
    python demo/publish_hf.py user/space-name

The Space is created if it does not exist.  Its files are replaced by the
contents of demo/site/, so anything uploaded there by hand is removed.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO))
import build  # noqa: E402

SPACE_NAME = "program-induction-rdt"


def main() -> None:
    api = HfApi()
    repo_id = sys.argv[1] if len(sys.argv) > 1 else f"{api.whoami()['name']}/{SPACE_NAME}"

    build.main()
    api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(
        folder_path=build.SITE, repo_id=repo_id, repo_type="space",
        commit_message="Update demo", delete_patterns="*",
    )
    print(f"Published https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
