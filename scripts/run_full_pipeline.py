from __future__ import annotations

import argparse
from pathlib import Path
import sys


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.pipeline_service import build_or_load_profile
from app.services.pipeline_service import build_user_embedding_variant
from app.services.pipeline_service import run_generation_for_user


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Gen4Rec pipeline in one command: "
            "embedding -> profile -> generation -> rerank -> eval."
        )
    )
    parser.add_argument("--user-id", required=True, help="Target user ID.")

    # Embedding stage
    parser.add_argument("--recent-k", type=int, default=10)
    parser.add_argument("--decay-lambda", type=float, default=0.08)
    parser.add_argument("--medoid-threshold", type=float, default=0.2)
    parser.add_argument("--min-keep", type=int, default=5)

    # Profile stage
    parser.add_argument("--top-k", type=int, default=20, help="Profile retrieval top-k.")
    parser.add_argument("--min-similarity", type=float, default=None, help="Optional profile retrieval threshold.")
    parser.add_argument("--exclude-recent", action="store_true", help="Exclude already listened songs in profile retrieval.")
    parser.add_argument("--openai-model", default="gpt-5.4-mini", help="OpenAI model for profile + prompt.")

    # Generation stage
    parser.add_argument("--generation-model", default="chirp-v4-5", help="Generation backend model.")
    parser.add_argument("--num-calls", type=int, default=5, help="Number of generation API calls.")
    parser.add_argument("--max-concurrency", type=int, default=2, help="Parallel generation call cap.")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--lyrics-file", default=None, help="Optional text file for lyrics/timestamp cues.")
    parser.add_argument("--tempo-hint-bpm", type=int, default=None)
    parser.add_argument("--duration-hint-seconds", type=int, default=None)

    # Rerank/Eval stage
    parser.add_argument("--rerank-top-k", type=int, default=2)
    parser.add_argument("--rerank-diversity-threshold", type=float, default=None)
    parser.add_argument("--rerank-encoder", choices=["auto", "finetuned", "zeroshot"], default="auto")
    parser.add_argument("--eval-recent-k", type=int, default=20)
    parser.add_argument("--eval-reference-top-k", type=int, default=3)
    parser.add_argument("--eval-encoder", choices=["auto", "finetuned", "zeroshot"], default="finetuned")
    parser.add_argument("--eval-no-plot", action="store_true", help="Disable eval plot saving.")
    parser.add_argument("--eval-imitation-threshold", type=float, default=0.9)

    # Rebuild switches
    parser.add_argument("--rebuild-generation", action="store_true", help="Force generation recompute.")
    parser.add_argument("--rebuild-rerank", action="store_true", help="Force rerank recompute.")
    parser.add_argument("--rebuild-eval", action="store_true", help="Force eval recompute.")
    args = parser.parse_args()

    lyrics = ""
    if args.lyrics_file:
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")

    print("[Stage A] Building/reusing user embedding variant...")
    emb_result = build_user_embedding_variant(
        recent_k=max(1, int(args.recent_k)),
        decay_lambda=float(args.decay_lambda),
        medoid_threshold=float(args.medoid_threshold),
        min_keep=max(1, int(args.min_keep)),
    )
    embedding_variant = str(emb_result["variant"])
    print(f"  embedding_variant: {embedding_variant}")

    print("[Stage B] Building/reusing profile artifacts...")
    profile_artifacts = build_or_load_profile(
        user_id=str(args.user_id),
        embedding_variant=embedding_variant,
        top_k=max(1, int(args.top_k)),
        min_similarity=args.min_similarity,
        exclude_recent=bool(args.exclude_recent),
        openai_model=str(args.openai_model),
    )
    if not profile_artifacts.prompt:
        raise RuntimeError("Profile prompt is missing after profile stage.")
    print(f"  prompt_path: {profile_artifacts.prompt_path}")

    print("[Stage C/D] Running generation, rerank, and eval...")
    gen_result = run_generation_for_user(
        user_id=str(args.user_id),
        embedding_variant=embedding_variant,
        prompt_output=profile_artifacts.prompt,
        generation_model=str(args.generation_model),
        num_calls=max(1, int(args.num_calls)),
        max_concurrency=max(1, int(args.max_concurrency)),
        negative_prompt=args.negative_prompt,
        lyrics=lyrics,
        tempo_hint_bpm=args.tempo_hint_bpm,
        duration_hint_seconds=args.duration_hint_seconds,
        rerank_top_k=max(1, int(args.rerank_top_k)),
        rerank_diversity_threshold=args.rerank_diversity_threshold,
        rerank_encoder=str(args.rerank_encoder),
        eval_recent_k=max(1, int(args.eval_recent_k)),
        eval_reference_top_k=max(1, int(args.eval_reference_top_k)),
        eval_encoder=str(args.eval_encoder),
        eval_save_plot=not bool(args.eval_no_plot),
        eval_imitation_threshold=float(args.eval_imitation_threshold),
        generation_rebuild=bool(args.rebuild_generation),
        rerank_rebuild=bool(args.rebuild_rerank),
        eval_rebuild=bool(args.rebuild_eval),
    )

    run_id = str(gen_result["run_id"])
    manifest_path = str(gen_result["manifest_path"])
    rerank_path = str(gen_result["rerank_output_path"])
    eval_summary_path = str(gen_result["eval_result"]["artifacts"]["eval_summary_json"])

    print("Pipeline completed.")
    print(f"  user_id: {args.user_id}")
    print(f"  embedding_variant: {embedding_variant}")
    print(f"  run_id: {run_id}")
    print(f"  manifest_path: {manifest_path}")
    print(f"  rerank_path: {rerank_path}")
    print(f"  eval_summary_path: {eval_summary_path}")


if __name__ == "__main__":
    main()
