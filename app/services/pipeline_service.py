from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.services.artifact_service import (
    OUTPUTS_ROOT,
    ProfileArtifacts,
    load_generation_run,
    load_profile_artifacts,
)
from src.embed.export_user_profile_json import export_user_profile_payload
from src.embed.build_user_embeddings import (
    Config as UserEmbConfig,
    build_user_embedding_variant_tag,
    build_user_embeddings as build_user_embeddings_matrix,
    ensure_local_file as ensure_useremb_local_file,
    ensure_song_ids as ensure_useremb_song_ids,
    load_listening_history as load_useremb_history,
)
from src.eval.data import EvalDataConfig
from src.eval.run_eval import evaluate_generation_run
from src.generate.rerank import run_rerank_from_manifest
from src.generate.run_generate import run_generation_pipeline
from src.profile_prompt.build_profile_features import build_profile_features, save_summary
from src.profile_prompt.generate_user_profile_and_prompt import generate_music_prompt, save_output


EMBEDDINGS_ROOT = OUTPUTS_ROOT / "embeddings" / "music4all"


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _embedding_variant_from_ids_path(path: Path) -> str | None:
    prefix = "user_ids__"
    suffix = ".npy"
    name = path.name
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    return name[len(prefix) : -len(suffix)]


def list_user_embedding_variants() -> list[str]:
    if not EMBEDDINGS_ROOT.exists():
        return []
    variants: list[str] = []
    for ids_path in EMBEDDINGS_ROOT.glob("user_ids__*.npy"):
        variant = _embedding_variant_from_ids_path(ids_path)
        if not variant:
            continue
        emb_path = EMBEDDINGS_ROOT / f"user_embeddings__{variant}.npy"
        if emb_path.exists():
            variants.append(variant)
    return sorted(set(variants))


def resolve_user_embedding_paths(embedding_variant: str) -> tuple[Path, Path]:
    ids_path = EMBEDDINGS_ROOT / f"user_ids__{embedding_variant}.npy"
    emb_path = EMBEDDINGS_ROOT / f"user_embeddings__{embedding_variant}.npy"
    if not ids_path.exists() or not emb_path.exists():
        raise FileNotFoundError(
            f"Missing embedding files for variant `{embedding_variant}`: "
            f"{emb_path} and {ids_path} must both exist."
        )
    return emb_path, ids_path


def build_user_embedding_variant(
    *,
    recent_k: int = 10,
    decay_lambda: float = 0.08,
    medoid_threshold: float = 0.2,
    min_keep: int = 5,
) -> dict[str, Any]:
    variant = build_user_embedding_variant_tag(
        recent_k=recent_k,
        decay_lambda=decay_lambda,
        medoid_threshold=medoid_threshold,
        min_keep=min_keep,
    )
    os.makedirs(UserEmbConfig.EMBEDDINGS_DIR, exist_ok=True)
    user_emb_path = Path(UserEmbConfig.EMBEDDINGS_DIR) / f"user_embeddings__{variant}.npy"
    user_ids_path = Path(UserEmbConfig.EMBEDDINGS_DIR) / f"user_ids__{variant}.npy"
    stats_path = Path(UserEmbConfig.EMBEDDINGS_DIR) / f"user_embedding_stats__{variant}.csv"

    if user_emb_path.exists() and user_ids_path.exists() and stats_path.exists():
        return {
            "variant": variant,
            "created": False,
            "user_emb_path": str(user_emb_path),
            "user_ids_path": str(user_ids_path),
            "stats_path": str(stats_path),
        }

    song_embs = np.load(ensure_useremb_local_file(UserEmbConfig.SONG_EMB_PATH, "Song embedding matrix")).astype(np.float32)
    song_ids = ensure_useremb_song_ids(
        UserEmbConfig.SONG_IDS_PATH,
        UserEmbConfig.ID_GENRES_PATH,
        expected_n=song_embs.shape[0],
    )
    listening_df = load_useremb_history(
        ensure_useremb_local_file(UserEmbConfig.LISTENING_HISTORY_PATH, "Listening history table")
    )

    user_ids, user_embs, stats_df = build_user_embeddings_matrix(
        listening_df=listening_df,
        song_ids=song_ids,
        song_embs=song_embs,
        recent_k=max(1, int(recent_k)),
        decay_lambda=float(decay_lambda),
        medoid_threshold=float(medoid_threshold),
        min_keep=max(1, int(min_keep)),
    )
    np.save(user_emb_path, user_embs)
    np.save(user_ids_path, user_ids)
    stats_df.to_csv(stats_path, index=False)
    return {
        "variant": variant,
        "created": True,
        "user_emb_path": str(user_emb_path),
        "user_ids_path": str(user_ids_path),
        "stats_path": str(stats_path),
    }


def load_available_users(embedding_variant: str) -> list[str]:
    _, user_ids_path = resolve_user_embedding_paths(embedding_variant)
    if not user_ids_path.exists():
        raise FileNotFoundError(
            f"User ID array not found at {user_ids_path}. "
            "Please build or copy user embeddings before using the demo."
        )
    user_ids = np.load(user_ids_path, allow_pickle=True).astype(str)
    return sorted(user_ids.tolist())


def build_profile_variant_tag(
    *,
    embedding_variant: str,
    top_k: int,
    min_similarity: float | None,
    exclude_recent: bool,
    openai_model: str,
) -> str:
    sim = "none" if min_similarity is None else f"{float(min_similarity):.3f}".rstrip("0").rstrip(".")
    model = (openai_model or "default").strip().replace(" ", "-")
    emb = (embedding_variant or "unknown").strip().replace(" ", "-")
    return f"ev{emb}_tk{int(top_k)}_ms{sim}_ex{int(bool(exclude_recent))}_om{model}"


def build_or_load_profile(
    *,
    user_id: str,
    embedding_variant: str,
    top_k: int = 20,
    min_similarity: float | None = None,
    exclude_recent: bool = True,
    openai_model: str = "gpt-5.4-mini",
) -> ProfileArtifacts:
    user_emb_path, user_ids_path = resolve_user_embedding_paths(embedding_variant)
    profile_variant = build_profile_variant_tag(
        embedding_variant=embedding_variant,
        top_k=top_k,
        min_similarity=min_similarity,
        exclude_recent=exclude_recent,
        openai_model=openai_model,
    )
    artifacts = load_profile_artifacts(user_id, profile_variant=profile_variant)

    raw_payload = artifacts.raw_profile
    if raw_payload is None:
        raw_payload = export_user_profile_payload(
            user_id=user_id,
            top_k=top_k,
            min_similarity=min_similarity,
            user_emb_path=str(user_emb_path),
            user_ids_path=str(user_ids_path),
            exclude_recent=exclude_recent,
        )
        _write_json(raw_payload, artifacts.raw_profile_path)

    summary = artifacts.summary
    if summary is None:
        summary = build_profile_features(raw_payload)
        save_summary(summary, artifacts.summary_path)

    prompt = artifacts.prompt
    if prompt is None:
        prompt = generate_music_prompt(summary, model=openai_model)
        save_output(prompt, artifacts.prompt_path)

    return load_profile_artifacts(user_id, profile_variant=profile_variant)


def run_generation_for_user(
    *,
    user_id: str,
    embedding_variant: str,
    prompt_output: dict[str, Any],
    generation_model: str = "chirp-v4-5",
    num_calls: int = 5,
    max_concurrency: int = 2,
    negative_prompt: str | None = None,
    lyrics: str = "",
    tempo_hint_bpm: int | None = None,
    duration_hint_seconds: int | None = None,
    prompt_version: str = "existing-profile-prompt-v1",
    rerank_top_k: int = 2,
    rerank_diversity_threshold: float | None = None,
    rerank_encoder: str = "auto",
    eval_recent_k: int = 20,
    eval_reference_top_k: int = 3,
    eval_encoder: str = "finetuned",
    eval_save_plot: bool = True,
    eval_imitation_threshold: float = 0.9,
) -> dict[str, Any]:
    user_emb_path, user_ids_path = resolve_user_embedding_paths(embedding_variant)
    EvalDataConfig.USER_EMB_PATH = str(user_emb_path)
    EvalDataConfig.USER_IDS_PATH = str(user_ids_path)

    run_id, manifest, manifest_path = run_generation_pipeline(
        prompt_output=prompt_output,
        provider="suno",
        generation_model=generation_model,
        user_id=user_id,
        num_calls=num_calls,
        max_concurrency=max_concurrency,
        negative_prompt=negative_prompt,
        lyrics=lyrics,
        tempo_hint_bpm=tempo_hint_bpm,
        duration_hint_seconds=duration_hint_seconds,
        prompt_version=prompt_version,
    )
    rerank_result, rerank_output_path = run_rerank_from_manifest(
        manifest_path=manifest_path,
        top_k=rerank_top_k,
        diversity_threshold=rerank_diversity_threshold,
        encoder=rerank_encoder,
    )
    eval_result = evaluate_generation_run(
        manifest_path=manifest_path,
        recent_k=eval_recent_k,
        reference_top_k=eval_reference_top_k,
        encoder=eval_encoder,
        rerank_top_k=rerank_top_k,
        diversity_threshold=rerank_diversity_threshold,
        save_plot=eval_save_plot,
        imitation_threshold=eval_imitation_threshold,
    )
    run_artifacts = load_generation_run(Path(manifest_path).parent)
    return {
        "run_id": run_id,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "rerank_result": rerank_result,
        "rerank_output_path": rerank_output_path,
        "eval_result": eval_result,
        "run_artifacts": run_artifacts,
    }
