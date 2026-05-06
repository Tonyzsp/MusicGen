"""
WAV clips → pooled CLAP user vector → profile + Suno (same core as scripts/run_phase2_eval.py).
Used by Streamlit when the user uploads .wav files only (no CSV).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.eval.clap_audio import embed_audio_paths
from src.eval.data import EvalDataConfig
from src.generate.rerank import run_rerank_from_manifest
from src.generate.run_generate import run_generation_pipeline
from src.profile_prompt.profile_pipeline import build_or_load_profile_pipeline


def safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return stem.strip("._") or "song"


def is_wav_bytes(raw: bytes) -> bool:
    """True if buffer looks like a RIFF/WAVE container (standard .wav)."""
    if len(raw) < 12:
        return False
    return raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"


def assert_only_wav_files(files: list[tuple[str, bytes]]) -> None:
    """Reject non-.wav names or payloads that are not RIFF/WAVE."""
    if not files:
        raise ValueError("No WAV files provided.")
    for name, raw in files:
        if Path(name).suffix.lower() != ".wav":
            raise ValueError(f"Only .wav files are allowed (got {name!r}).")
        if not is_wav_bytes(raw):
            raise ValueError(f"File is not a valid WAV (expected RIFF/WAVE): {name!r}.")


def clip_paths_in_upload_order(clips_dir: Path, files: list[tuple[str, bytes]]) -> list[Path]:
    """Resolve written clip paths in the same order as `files` (after `write_wav_files_to_clips_dir`)."""
    clips_dir = Path(clips_dir).resolve()
    seen: set[str] = set()
    out: list[Path] = []
    for name, _ in files:
        base = Path(name).name
        key = base.lower()
        if key in seen:
            raise ValueError(f"Duplicate WAV filename in upload: {base!r}. Use unique names.")
        seen.add(key)
        out.append((clips_dir / base).resolve())
    return out


def _mean_l2_normalize(vectors: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(vectors, axis=0).astype(np.float32)
    mean = stacked.mean(axis=0)
    t = torch.from_numpy(mean)
    t = F.normalize(t.unsqueeze(0), dim=-1).squeeze(0)
    out = t.numpy().astype(np.float32)
    if float(np.linalg.norm(out)) < 1e-6:
        raise ValueError("Degenerate user embedding after pooling (zero norm).")
    return out


def write_wav_files_to_clips_dir(
    clips_dir: Path,
    files: list[tuple[str, bytes]],
    *,
    clear: bool = True,
) -> None:
    assert_only_wav_files(files)
    clips_dir = Path(clips_dir)
    if clear and clips_dir.exists():
        shutil.rmtree(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    for name, raw in files:
        dest = clips_dir / Path(name).name
        dest.write_bytes(raw)


def run_custom_playlist_pipeline(
    *,
    wav_paths_ordered: list[Path],
    participant_slug: str,
    work_root: Path,
    encoder: str = "finetuned",
    top_k: int = 10,
    min_similarity: float | None = None,
    openai_model: str = "gpt-5.4-mini",
    generation_model: str = "chirp-v4-5",
    num_calls: int = 1,
    max_concurrency: int = 1,
    negative_prompt: str | None = None,
    lyrics: str = "",
    tempo_hint_bpm: int | None = None,
    duration_hint_seconds: int | None = None,
    rerank_top_k: int = 2,
    rerank_encoder: str = "finetuned",
    rerank_diversity_threshold: float | None = None,
    rebuild_profile: bool = False,
) -> dict[str, Any]:
    participant_slug = safe_file_stem(participant_slug) or "playlist"
    result_dir = (work_root / participant_slug).resolve()
    emb_dir = result_dir / "_emb"
    result_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    str_paths = [str(p) for p in wav_paths_ordered]
    embeddings, enc_cfg = embed_audio_paths(str_paths, encoder=encoder)
    ordered = [embeddings[str(p)] for p in wav_paths_ordered]
    user_vec = _mean_l2_normalize(ordered)

    synthetic_user_id = f"custompl_{participant_slug}"
    embedding_variant = "custom_wav_mean"
    user_emb_path = emb_dir / f"user_embeddings__{embedding_variant}.npy"
    user_ids_path = emb_dir / f"user_ids__{embedding_variant}.npy"
    np.save(user_emb_path, user_vec.reshape(1, -1))
    np.save(user_ids_path, np.array([synthetic_user_id], dtype=object))

    meta_pre = {
        "participant_slug": participant_slug,
        "synthetic_user_id": synthetic_user_id,
        "embedding_variant": embedding_variant,
        "encoder": enc_cfg.get("encoder_name"),
        "clips": str_paths,
        "user_emb_path": str(user_emb_path),
        "user_ids_path": str(user_ids_path),
    }
    (result_dir / "wav_embedding_meta.json").write_text(
        json.dumps(meta_pre, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    profile_out = build_or_load_profile_pipeline(
        user_id=synthetic_user_id,
        embedding_variant=embedding_variant,
        user_emb_path=str(user_emb_path),
        user_ids_path=str(user_ids_path),
        top_k=max(1, int(top_k)),
        min_similarity=min_similarity,
        exclude_recent=False,
        openai_model=str(openai_model),
        rebuild=bool(rebuild_profile),
    )
    prompt_payload = profile_out["prompt"]

    run_id, _manifest, manifest_path = run_generation_pipeline(
        prompt_output=prompt_payload,
        provider="suno",
        generation_model=str(generation_model),
        user_id=synthetic_user_id,
        num_calls=max(1, int(num_calls)),
        max_concurrency=max(1, int(max_concurrency)),
        negative_prompt=negative_prompt,
        lyrics=lyrics,
        tempo_hint_bpm=tempo_hint_bpm,
        duration_hint_seconds=duration_hint_seconds,
        outputs_root=result_dir,
    )

    EvalDataConfig.USER_EMB_PATH = str(user_emb_path)
    EvalDataConfig.USER_IDS_PATH = str(user_ids_path)
    rerank_result, rerank_output_path = run_rerank_from_manifest(
        manifest_path=manifest_path,
        top_k=max(1, int(rerank_top_k)),
        diversity_threshold=rerank_diversity_threshold,
        encoder=str(rerank_encoder),
    )

    done = {
        "run_id": run_id,
        "manifest_path": manifest_path,
        "run_root": str(Path(manifest_path).parent),
        "rerank_output_path": rerank_output_path,
        "synthetic_user_id": synthetic_user_id,
        "profile_variant": profile_out["profile_variant"],
        "encoder": enc_cfg.get("encoder_name"),
    }
    (result_dir / "custom_playlist_generation_meta.json").write_text(
        json.dumps(done, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        **done,
        "profile_out": profile_out,
        "rerank_result": rerank_result,
    }
