"""
Export materials for a lightweight human preference test on text-to-music retrieval.

For each prompt: top-k from base CLAP and top-k from fine-tuned CLAP, shuffle four clips,
copy MP3s with blind filenames, and write a researcher-only manifest.

Usage (from repo root):
  python scripts/export_human_retrieval_eval.py --out-dir outputs/human_retrieval_eval

Requires: Music4All audio files and zeroshot / finetuned song embedding matrices (same as Streamlit retrieval).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.text_compare_service import load_embedding_matrices, retrieve_both


EVAL_PROMPTS: list[tuple[str, str]] = [
    (
        "Rainy Night",
        "slow emotional music for walking alone on a rainy night, gentle piano, quiet, lonely, and reflective",
    ),
    (
        "Morning Coffee",
        "relaxing acoustic guitar music for drinking coffee in the morning, warm, calm, and comfortable",
    ),
    (
        "Workout Energy",
        "fast high-energy music for working out, powerful, motivating, and intense",
    ),
    (
        "Beach Sunset",
        "laid-back summer music for watching the beach sunset, relaxed, sunny, and peaceful",
    ),
    (
        "City Drive",
        "cool pop or R&B music for driving through the city at night, stylish, smooth, and confident",
    ),
    (
        "Happy Party",
        "fun dance music for a small party with friends, cheerful, catchy, and easy to move to",
    ),
    (
        "Sad Memory",
        "slow sad music for thinking about an old memory, soft piano, emotional, nostalgic, and lonely",
    ),
    (
        "Study Focus",
        "calm background music for studying, steady, relaxing, and not distracting",
    ),
    (
        "Adventure Scene",
        "cinematic music that feels like starting an adventure, with dramatic strings and drums, exciting, hopeful, and movie-like",
    ),
    (
        "Romantic Dinner",
        "smooth romantic dinner music with soft saxophone, warm, intimate, relaxing, and elegant",
    ),
]


def _build_four_candidates(
    base_hits: list[dict[str, Any]],
    finetune_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank, item in enumerate(base_hits, start=1):
        out.append({**item, "source": "base", "retrieval_rank_within_model": rank})
    for rank, item in enumerate(finetune_hits, start=1):
        out.append({**item, "source": "finetuned", "retrieval_rank_within_model": rank})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export blinded audio clips for base vs fine-tuned CLAP retrieval evaluation."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "human_retrieval_eval",
        help="Output directory (creates audio/ and writes manifest).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for shuffling (per prompt: seed + prompt index).",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Top-k per model (default: 2).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies only; do not write files.",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir.resolve()
    audio_dir = out_dir / "audio"
    if not args.dry_run:
        audio_dir.mkdir(parents=True, exist_ok=True)

    matrices = load_embedding_matrices()
    manifest: dict[str, Any] = {
        "description": (
            "Researcher-only manifest: includes base/finetuned labels. "
            "Participants should only receive files under audio/; do not share this JSON with them."
        ),
        "seed": args.seed,
        "top_k_per_model": args.top_k,
        "embedding_sources": {
            "zeroshot_matrix": matrices.zeroshot_path,
            "finetuned_matrix": matrices.finetuned_path,
        },
        "prompts": [],
    }
    missing_audio: list[str] = []

    for pi, (title, query_text) in enumerate(EVAL_PROMPTS, start=1):
        pid = f"p{pi:02d}"
        base_hits, ft_hits = retrieve_both(query_text, matrices, top_k=args.top_k)
        candidates = _build_four_candidates(base_hits, ft_hits)
        rng = random.Random(args.seed + pi)
        order = list(range(len(candidates)))
        rng.shuffle(order)
        shuffled = [candidates[i] for i in order]

        prompt_block: dict[str, Any] = {
            "prompt_id": pid,
            "title": title,
            "query_text": query_text,
            "clips": [],
        }

        for display_idx, c in enumerate(shuffled, start=1):
            src = Path(c["audio_path"])
            fname = f"{pid}_clip{display_idx:02d}.mp3"
            dst = audio_dir / fname
            if not c.get("audio_exists", False) or not src.is_file():
                missing_audio.append(str(src))
            elif not args.dry_run:
                shutil.copy2(src, dst)

            prompt_block["clips"].append(
                {
                    "display_index": display_idx,
                    "filename": fname,
                    "source": c["source"],
                    "retrieval_rank_within_model": c["retrieval_rank_within_model"],
                    "song_id": c["song_id"],
                    "cosine_similarity": c["cosine_similarity"],
                    "source_audio_path": str(src),
                    "copied_ok": bool(c.get("audio_exists")) and src.is_file(),
                }
            )

        manifest["prompts"].append(prompt_block)

    participant_path = out_dir / "participant_instructions.txt"
    lines: list[str] = [
        "Human preference evaluation: text-to-music retrieval",
        "",
        "For each item below, read the text description, then listen to the four audio clips",
        "(clip01–clip04 in the audio folder for that prompt id). Rank the clips from best to worst",
        "match to the description (paper form or survey). Clip order is randomized and is not",
        "informative about which model produced which retrieval.",
        "",
    ]
    for pb in manifest["prompts"]:
        lines.append(f"--- {pb['prompt_id']} {pb['title']} ---")
        lines.append(pb["query_text"])
        lines.append("Clips: " + ", ".join(c["filename"] for c in pb["clips"]))
        lines.append("")
    lines.append("Example response row: prompt_id | best_clip | second | third | worst")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        participant_path.write_text("\n".join(lines), encoding="utf-8")

    if missing_audio:
        print("Warning: missing or unreadable audio (see manifest copied_ok=false):")
        for p in sorted(set(missing_audio)):
            print(" ", p)

    print(f"Done. Output directory: {out_dir}")
    if args.dry_run:
        print("(dry-run: no files written)")
    else:
        print(f"  Researcher manifest: {out_dir / 'manifest.json'}")
        print(f"  Participant sheet: {participant_path}")
        print(f"  Audio directory: {audio_dir}")


if __name__ == "__main__":
    main()
