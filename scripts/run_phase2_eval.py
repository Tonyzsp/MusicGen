"""
Phase 2 lab pipeline: WAV clips -> finetuned CLAP user vector -> Music4All retrieval
-> profile + Suno prompt -> Suno generation (artifacts under eval_phase_2/<participant>/result/).

Prerequisites (same as main Gen4Rec stack):
- music4all dataset CSVs, `outputs/embeddings/music4all/music4all_embeddings.npy` (+ ids),
  optional FAISS index, finetuned CLAP weights, OpenAI + Suno API keys in `.env`.

Example:
  python scripts/run_phase2_eval.py --participant Tony --top-k 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.data import EvalDataConfig
from src.eval.clap_audio import embed_audio_paths
from src.generate.rerank import run_rerank_from_manifest
from src.generate.run_generate import run_generation_pipeline
from src.profile_prompt.profile_pipeline import build_or_load_profile_pipeline


def _mean_l2_normalize(vectors: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(vectors, axis=0).astype(np.float32)
    mean = stacked.mean(axis=0)
    t = torch.from_numpy(mean)
    t = F.normalize(t.unsqueeze(0), dim=-1).squeeze(0)
    out = t.numpy().astype(np.float32)
    if float(np.linalg.norm(out)) < 1e-6:
        raise ValueError("Degenerate user embedding after pooling (zero norm).")
    return out


def _collect_wavs(clips_dir: Path) -> list[Path]:
    wavs = sorted(clips_dir.glob("*.wav"), key=lambda p: p.name.lower())
    if not wavs:
        raise FileNotFoundError(f"No .wav files under {clips_dir}")
    return wavs


def _copy_if_exists(src_path: str | None, dst: Path) -> bool:
    if not src_path:
        return False
    src = Path(src_path)
    if not src.exists():
        return False
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: average finetuned CLAP embeddings of 30s WAVs, retrieve from Music4All, "
            "build profile + prompt, run Suno; write under src/eval/eval_phase_2/<participant>/result/."
        )
    )
    parser.add_argument(
        "--participant",
        required=True,
        help="Folder name under src/eval/eval_phase_2/ (e.g. Tony). Expects clips_30s/*.wav there.",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=None,
        help="Override WAV directory (default: src/eval/eval_phase_2/<participant>/clips_30s).",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Override output root (default: src/eval/eval_phase_2/<participant>/result).",
    )
    parser.add_argument(
        "--encoder",
        choices=["finetuned", "zeroshot", "auto"],
        default="finetuned",
        help="CLAP encoder for WAV pooling (default: finetuned).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Music4All retrieval top-k.")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Optional cosine similarity floor for retrieval.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-5.4-mini",
        help="OpenAI model for profile + Suno prompt.",
    )
    parser.add_argument("--generation-model", default="chirp-v4-5", help="Suno hosted model id.")
    parser.add_argument(
        "--num-calls",
        type=int,
        default=1,
        help="Parallel Suno API calls (each call may return two variants).",
    )
    parser.add_argument("--max-concurrency", type=int, default=1, help="Max concurrent Suno calls.")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--rerank-top-k", type=int, default=2, help="Final number of tracks after reranking.")
    parser.add_argument(
        "--rerank-encoder",
        choices=["finetuned", "zeroshot", "auto"],
        default="finetuned",
        help="CLAP encoder used for rerank cosine similarity.",
    )
    parser.add_argument(
        "--rerank-diversity-threshold",
        type=float,
        default=None,
        help="Optional diversity filter threshold applied during rerank.",
    )
    parser.add_argument(
        "--rebuild-profile",
        action="store_true",
        help="Ignore cached outputs/profiles artifacts for this user+variant and rebuild.",
    )
    args = parser.parse_args()

    participant = str(args.participant).strip()
    if not participant:
        raise SystemExit("--participant must be non-empty.")

    base = REPO_ROOT / "src" / "eval" / "eval_phase_2" / participant
    clips_dir = (args.clips_dir or (base / "clips_30s")).resolve()
    result_dir = (args.result_dir or (base / "result")).resolve()
    emb_dir = result_dir / "_phase2_emb"
    result_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    wav_paths = _collect_wavs(clips_dir)
    print(f"[1/4] Embedding {len(wav_paths)} WAV(s) with encoder={args.encoder!r} …")
    embeddings, enc_cfg = embed_audio_paths([str(p) for p in wav_paths], encoder=args.encoder)
    ordered = [embeddings[str(p)] for p in wav_paths]
    user_vec = _mean_l2_normalize(ordered)

    synthetic_user_id = f"phase2_{participant}"
    embedding_variant = "phase2_wav_mean"
    user_emb_path = emb_dir / f"user_embeddings__{embedding_variant}.npy"
    user_ids_path = emb_dir / f"user_ids__{embedding_variant}.npy"
    np.save(user_emb_path, user_vec.reshape(1, -1))
    np.save(user_ids_path, np.array([synthetic_user_id], dtype=object))

    meta = {
        "participant": participant,
        "synthetic_user_id": synthetic_user_id,
        "embedding_variant": embedding_variant,
        "encoder": enc_cfg.get("encoder_name"),
        "clips": [str(p) for p in wav_paths],
        "user_emb_path": str(user_emb_path),
        "user_ids_path": str(user_ids_path),
    }
    (result_dir / "wav_embedding_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[2/4] Retrieval + profile + prompt …")
    profile_out = build_or_load_profile_pipeline(
        user_id=synthetic_user_id,
        embedding_variant=embedding_variant,
        user_emb_path=str(user_emb_path),
        user_ids_path=str(user_ids_path),
        top_k=max(1, int(args.top_k)),
        min_similarity=args.min_similarity,
        exclude_recent=False,
        openai_model=str(args.openai_model),
        rebuild=bool(args.rebuild_profile),
    )
    paths = profile_out["paths"]
    shutil.copy2(paths["raw_profile"], result_dir / "profile_retrieval_raw.json")
    shutil.copy2(paths["summary"], result_dir / "profile_summary.json")
    shutil.copy2(paths["prompt"], result_dir / "music_prompt.json")

    prompt_payload = profile_out["prompt"]
    print("[3/4] Suno generation …")
    run_id, manifest, manifest_path = run_generation_pipeline(
        prompt_output=prompt_payload,
        provider="suno",
        generation_model=str(args.generation_model),
        user_id=synthetic_user_id,
        num_calls=max(1, int(args.num_calls)),
        max_concurrency=max(1, int(args.max_concurrency)),
        negative_prompt=args.negative_prompt,
        outputs_root=result_dir,
    )

    print("[4/5] Rerank generated tracks …")
    EvalDataConfig.USER_EMB_PATH = str(user_emb_path)
    EvalDataConfig.USER_IDS_PATH = str(user_ids_path)
    rerank_result, rerank_output_path = run_rerank_from_manifest(
        manifest_path=manifest_path,
        top_k=max(1, int(args.rerank_top_k)),
        diversity_threshold=args.rerank_diversity_threshold,
        encoder=str(args.rerank_encoder),
    )

    ranked = rerank_result.get("final_selected_tracks") or rerank_result.get("candidates") or []
    song_aliases: list[dict[str, object]] = []
    run_root = Path(manifest_path).parent
    for idx, item in enumerate(ranked[:2], start=1):
        alias = f"song{idx}"
        copied_mp3 = _copy_if_exists(item.get("path"), run_root / f"{alias}.mp3")
        copied_json = _copy_if_exists(item.get("metadata_path"), run_root / f"{alias}.json")
        copied_lyrics = _copy_if_exists(item.get("lyric_path"), run_root / f"{alias}_lyrics.txt")
        song_aliases.append(
            {
                "alias": alias,
                "source_path": item.get("path"),
                "clap_cosine_score": item.get("clap_cosine_score"),
                "copied_mp3": copied_mp3,
                "copied_json": copied_json,
                "copied_lyrics": copied_lyrics,
            }
        )

    done = {
        "run_id": run_id,
        "manifest_path": manifest_path,
        "generation_artifacts_root": str(Path(manifest_path).parent),
        "rerank_output_path": rerank_output_path,
        "rerank_top_k": max(1, int(args.rerank_top_k)),
        "song_aliases": song_aliases,
        "profile_cache_paths": paths,
    }
    (result_dir / "phase2_generation_meta.json").write_text(
        json.dumps(done, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[5/5] Done.")
    print(f"  result_dir: {result_dir}")
    print(f"  run_id: {run_id}")
    print(f"  manifest: {manifest_path}")
    print(f"  rerank: {rerank_output_path}")
    if song_aliases:
        print("  aliases: " + ", ".join(f"{x['alias']}={x['clap_cosine_score']:.4f}" for x in song_aliases if isinstance(x.get("clap_cosine_score"), float)))


if __name__ == "__main__":
    main()
