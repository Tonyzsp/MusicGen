from __future__ import annotations

import argparse
from pathlib import Path
import sys


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.profile_prompt.profile_pipeline import build_or_load_profile_pipeline


def resolve_user_embedding_paths(embedding_variant: str) -> tuple[str, str]:
    embeddings_root = REPO_ROOT / "outputs" / "embeddings" / "music4all"
    emb_path = embeddings_root / f"user_embeddings__{embedding_variant}.npy"
    ids_path = embeddings_root / f"user_ids__{embedding_variant}.npy"
    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Missing embedding files for variant `{embedding_variant}`: {emb_path} and {ids_path}."
        )
    return str(emb_path), str(ids_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build or reuse profile artifacts with a parameter-stable variant key. "
            "Outputs are written to outputs/profiles/<user>__<variant>*."
        )
    )
    parser.add_argument("--user-id", required=True, help="Target user ID.")
    parser.add_argument(
        "--embedding-variant",
        required=True,
        help="Embedding variant key, e.g. rk10_dl0.08_mt0.2_mk5.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top-k retrieval size.")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Optional cosine threshold for retrieval filtering.",
    )
    parser.add_argument(
        "--exclude-recent",
        action="store_true",
        help="Exclude songs already listened by the user from retrieval candidates.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5.4-mini",
        help="OpenAI model for profile paragraph and generation prompt.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild artifacts even if matching variant files already exist.",
    )
    args = parser.parse_args()

    user_emb_path, user_ids_path = resolve_user_embedding_paths(args.embedding_variant)
    result = build_or_load_profile_pipeline(
        user_id=str(args.user_id),
        embedding_variant=str(args.embedding_variant),
        user_emb_path=user_emb_path,
        user_ids_path=user_ids_path,
        top_k=max(1, int(args.top_k)),
        min_similarity=args.min_similarity,
        exclude_recent=bool(args.exclude_recent),
        openai_model=str(args.openai_model),
        rebuild=bool(args.rebuild),
    )
    cache_status = result.get("cache_status", {})
    raw_hit = bool(cache_status.get("raw_profile_cache_hit"))
    summary_hit = bool(cache_status.get("summary_cache_hit"))
    prompt_hit = bool(cache_status.get("prompt_cache_hit"))
    full_hit = bool(cache_status.get("full_cache_hit"))

    print("Profile pipeline completed.")
    if args.rebuild:
        print("Rebuild mode enabled: cache was bypassed.")
    elif full_hit:
        print("Cache hit: reused raw profile, summary, and prompt artifacts.")
    else:
        print(
            "Partial rebuild: "
            f"raw_profile_cache_hit={raw_hit}, "
            f"summary_cache_hit={summary_hit}, "
            f"prompt_cache_hit={prompt_hit}"
        )
    print(f"Profile variant: {result['profile_variant']}")
    print(f"Raw profile: {result['paths']['raw_profile']}")
    print(f"Summary: {result['paths']['summary']}")
    print(f"Prompt: {result['paths']['prompt']}")


if __name__ == "__main__":
    main()
