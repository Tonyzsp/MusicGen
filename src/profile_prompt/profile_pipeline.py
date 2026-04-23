from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.embed.export_user_profile_json import export_user_profile_payload
from src.generate.artifacts import sanitize_segment
from src.profile_prompt.build_profile_features import build_profile_features, save_summary
from src.profile_prompt.generate_user_profile_and_prompt import generate_music_prompt, save_output


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent.parent
PROFILES_ROOT = REPO_ROOT / "outputs" / "profiles"


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


@dataclass
class ProfilePaths:
    user_id: str
    profile_variant: str
    raw_profile_path: Path
    summary_path: Path
    prompt_path: Path
    validation_path: Path


def get_profile_paths(user_id: str, *, profile_variant: str) -> ProfilePaths:
    safe_user_id = sanitize_segment(user_id)
    suffix = f"__{sanitize_segment(profile_variant)}"
    return ProfilePaths(
        user_id=user_id,
        profile_variant=profile_variant,
        raw_profile_path=PROFILES_ROOT / f"{safe_user_id}{suffix}.json",
        summary_path=PROFILES_ROOT / f"{safe_user_id}{suffix}_topk_summary.json",
        prompt_path=PROFILES_ROOT / f"{safe_user_id}{suffix}_prompt.json",
        validation_path=PROFILES_ROOT / f"{safe_user_id}{suffix}_validation.json",
    )


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_or_load_profile_pipeline(
    *,
    user_id: str,
    embedding_variant: str,
    user_emb_path: str,
    user_ids_path: str,
    top_k: int = 20,
    min_similarity: float | None = None,
    exclude_recent: bool = True,
    openai_model: str = "gpt-5.4-mini",
    rebuild: bool = False,
) -> dict[str, Any]:
    profile_variant = build_profile_variant_tag(
        embedding_variant=embedding_variant,
        top_k=top_k,
        min_similarity=min_similarity,
        exclude_recent=exclude_recent,
        openai_model=openai_model,
    )
    paths = get_profile_paths(user_id, profile_variant=profile_variant)

    raw_cache_hit = False
    summary_cache_hit = False
    prompt_cache_hit = False

    raw_payload = None if rebuild else _load_json_if_exists(paths.raw_profile_path)
    if raw_payload is None:
        raw_payload = export_user_profile_payload(
            user_id=user_id,
            top_k=top_k,
            min_similarity=min_similarity,
            user_emb_path=user_emb_path,
            user_ids_path=user_ids_path,
            exclude_recent=exclude_recent,
        )
        _write_json(raw_payload, paths.raw_profile_path)
    else:
        raw_cache_hit = True

    summary = None if rebuild else _load_json_if_exists(paths.summary_path)
    if summary is None:
        summary = build_profile_features(raw_payload)
        save_summary(summary, paths.summary_path)
    else:
        summary_cache_hit = True

    prompt = None if rebuild else _load_json_if_exists(paths.prompt_path)
    if prompt is None:
        prompt = generate_music_prompt(summary, model=openai_model)
        save_output(prompt, paths.prompt_path)
    else:
        prompt_cache_hit = True

    return {
        "profile_variant": profile_variant,
        "paths": {
            "raw_profile": str(paths.raw_profile_path),
            "summary": str(paths.summary_path),
            "prompt": str(paths.prompt_path),
            "validation": str(paths.validation_path),
        },
        "raw_profile": raw_payload,
        "summary": summary,
        "prompt": prompt,
        "cache_status": {
            "raw_profile_cache_hit": raw_cache_hit,
            "summary_cache_hit": summary_cache_hit,
            "prompt_cache_hit": prompt_cache_hit,
            "full_cache_hit": raw_cache_hit and summary_cache_hit and prompt_cache_hit,
            "rebuild_requested": bool(rebuild),
        },
    }
