from __future__ import annotations

import os
from pathlib import Path
import sys
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib
import streamlit as st

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.artifact_service import (
    EvalArtifacts,
    GenerationRunArtifacts,
    PROFILES_ROOT,
    ProfileArtifacts,
    load_eval_artifacts,
    load_generation_run,
    load_profile_artifacts,
    read_binary_file,
)
from app.services.custom_playlist_service import (
    clip_paths_in_upload_order,
    run_custom_playlist_pipeline,
    write_wav_files_to_clips_dir,
)
from app.services.pipeline_service import (
    build_user_embedding_variant,
    build_or_load_profile,
    build_profile_variant_tag,
    list_user_embedding_variants,
    load_available_users,
    resolve_user_embedding_paths,
    run_generation_for_user,
)
from app.services.viz_service import build_user_generation_figure
from app.services.query_compare import render_query_compare_page
from src.embed.build_user_embeddings import (
    Config as UserEmbConfig,
    ensure_local_file as ensure_useremb_local_file,
    ensure_song_ids as ensure_useremb_song_ids,
    load_listening_history as load_useremb_history,
)
from src.embed.recommend_topk import Config as RecConfig
from src.embed.recommend_topk import load_song_metadata
from src.generate.artifacts import sanitize_segment

# Generate AI Song: single page that stacks Phase 1 → 3 (sidebar shows all related controls).
GENERATE_SECTION_FULL_PIPELINE = "Full pipeline (Phases 1–3)"


st.set_page_config(
    page_title="Gen4Rec Demo",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_available_embedding_variants() -> list[str]:
    return list_user_embedding_variants()


@st.cache_data(show_spinner=False)
def get_available_users(embedding_variant: str) -> list[str]:
    return load_available_users(embedding_variant)


@st.cache_data(show_spinner=False)
def list_profile_variants_for_user(user_id: str) -> list[str]:
    safe_user_id = sanitize_segment(user_id)
    if not safe_user_id:
        return []
    variants: set[str] = set()
    for path in PROFILES_ROOT.glob(f"{safe_user_id}__*.json"):
        stem = path.stem
        prefix = f"{safe_user_id}__"
        if not stem.startswith(prefix):
            continue
        variant = stem[len(prefix) :]
        if variant.endswith("_prompt") or variant.endswith("_validation") or variant.endswith("_topk_summary"):
            continue
        variants.add(variant)
    return sorted(variants)


@st.cache_data(show_spinner=False)
def get_all_known_users() -> list[str]:
    history_df = load_useremb_history(
        ensure_useremb_local_file(UserEmbConfig.LISTENING_HISTORY_PATH, "Listening history table")
    )
    return sorted(history_df["user_id"].astype(str).unique().tolist())


@st.cache_resource(show_spinner=False)
def get_phase1_static_resources() -> dict[str, object]:
    song_embs = np.load(ensure_useremb_local_file(UserEmbConfig.SONG_EMB_PATH, "Song embedding matrix")).astype(np.float32)
    song_ids = ensure_useremb_song_ids(
        UserEmbConfig.SONG_IDS_PATH,
        UserEmbConfig.ID_GENRES_PATH,
        expected_n=song_embs.shape[0],
    )
    history_df = load_useremb_history(ensure_useremb_local_file(UserEmbConfig.LISTENING_HISTORY_PATH, "Listening history table"))
    info_df = load_song_metadata(ensure_useremb_local_file(RecConfig.ID_INFORMATION_PATH, "Song information table"))
    info_df = info_df.rename(columns={"id": "song_id"})
    song2idx = {str(sid): i for i, sid in enumerate(song_ids.astype(str))}
    return {
        "song_embs": song_embs,
        "history_df": history_df,
        "info_df": info_df,
        "song2idx": song2idx,
    }


@st.cache_resource(show_spinner=False)
def get_variant_user_embeddings(embedding_variant: str) -> tuple[np.ndarray, np.ndarray]:
    user_emb_path, user_ids_path = resolve_user_embedding_paths(embedding_variant)
    user_embs = np.load(user_emb_path).astype(np.float32)
    user_ids = np.load(user_ids_path, allow_pickle=True).astype(str)
    return user_embs, user_ids


@st.cache_data(show_spinner=False)
def _load_music4all_aa_song_index(path_str: str, mtime: float) -> dict[str, dict[str, object]]:
    path = Path(path_str)
    df = pd.read_parquet(path)
    if "song_id" not in df.columns:
        return {}
    df = df.astype(object).where(pd.notna(df), None)
    return {str(row["song_id"]): row for row in df.to_dict("records") if row.get("song_id")}


def get_music4all_aa_song_index() -> dict[str, dict[str, object]]:
    path = Path(
        os.environ.get(
            "GEN4REC_MUSIC4ALL_AA_INDEX_PATH",
            str(REPO_ROOT / "data" / "derived" / "music4all_aa_song_index.parquet"),
        )
    )
    if not path.is_file():
        return {}
    return _load_music4all_aa_song_index(str(path), path.stat().st_mtime)


def _format_metric_value(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _aa_text(row: dict[str, object] | None, key: str) -> str:
    if not row:
        return ""
    value = row.get(key)
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_label(value: object) -> str:
    return str(value or "").strip().casefold()


def _aa_genre_caption(row: dict[str, object] | None, key: str, *, limit: int = 3) -> str:
    raw = _aa_text(row, key)
    if not raw:
        return ""
    values = [part.strip() for part in raw.split("|") if part.strip()]
    return ", ".join(values[:limit])


def _lastfm_tag_url(tag: str) -> str:
    return f"https://www.last.fm/tag/{quote_plus(tag)}"


def _html_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _linked_tag_chips(values: str, *, limit: int = 3) -> str:
    tags = [part.strip() for part in values.split("|") if part.strip()][:limit]
    if not tags:
        return ""
    chips = "".join(
        (
            '<a class="aa-profile-chip" '
            f'href="{_html_escape(_lastfm_tag_url(tag))}" '
            f'target="_blank" rel="noopener noreferrer">{_html_escape(tag)}</a>'
        )
        for tag in tags
    )
    return f'<div class="aa-profile-chip-row">{chips}</div>'


def _inject_profile_visual_styles() -> None:
    st.markdown(
        """
<style>
  .aa-profile-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.06);
    height: 100%;
    display: flex;
    flex-direction: column;
  }
  .aa-profile-card-body {
    padding: 0.55rem 0.7rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.12rem;
    flex: 1 1 auto;
  }
  .aa-profile-visual-thumb {
    width: 100%;
    aspect-ratio: 1 / 1;
    overflow: hidden;
    background: #f1f5f9;
    flex-shrink: 0;
  }
  .aa-profile-visual-thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    display: block;
  }
  .aa-profile-kicker {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin: 0 0 0.1rem 0;
  }
  .aa-profile-title-row {
    margin: 0;
    line-height: 1.28;
  }
  .aa-profile-title-text {
    font-size: 1.02rem;
    font-weight: 700;
    color: #0f172a;
  }
  .aa-profile-title-link {
    color: #0f172a !important;
    text-decoration: none !important;
  }
  .aa-profile-title-link:hover {
    color: #2563eb !important;
    text-decoration: underline !important;
  }
  .aa-profile-subtitle {
    font-size: 0.8125rem;
    font-weight: 500;
    color: #475569;
    margin: 0.18rem 0 0 0;
    line-height: 1.35;
  }
  .aa-profile-album-line {
    margin: 0.35rem 0 0.15rem 0;
    padding: 0;
    font-size: 0.8rem;
    line-height: 1.4;
  }
  .aa-profile-album-label {
    font-weight: 700;
    color: #64748b;
    margin-right: 0.28rem;
  }
  .aa-profile-album-link {
    color: #1d4ed8 !important;
    font-weight: 600;
    text-decoration: none !important;
  }
  .aa-profile-album-link:hover {
    color: #1e3a8a !important;
    text-decoration: underline !important;
  }
  .aa-profile-album-plain {
    color: #334155;
    font-weight: 600;
  }
  .aa-profile-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.38rem;
    margin-top: auto;
    padding-top: 0.45rem;
  }
  .aa-profile-chip {
    border: 1px solid #cbd5e1;
    border-radius: 999px;
    background: #f1f5f9;
    color: #0f172a !important;
    display: inline-block;
    font-size: 0.8125rem;
    font-weight: 600;
    line-height: 1.25;
    padding: 0.38rem 0.62rem;
    text-decoration: none !important;
  }
  .aa-profile-chip:hover {
    border-color: #2563eb;
    background: #e0e7ff;
    color: #0f172a !important;
  }
</style>
""",
        unsafe_allow_html=True,
    )


def _render_visual_card(item: dict[str, str]) -> None:
    img_url = _html_escape(item["image_url"])
    kicker = _html_escape(item.get("kicker") or "")
    title_esc = _html_escape(item["title"])
    if item.get("url"):
        title_block = (
            f'<strong class="aa-profile-title-text"><a class="aa-profile-title-link" '
            f'href="{_html_escape(item["url"])}" target="_blank" rel="noopener noreferrer">'
            f"{title_esc}</a></strong>"
        )
    else:
        title_block = f'<strong class="aa-profile-title-text">{title_esc}</strong>'
    subtitle_html = ""
    if item.get("subtitle"):
        subtitle_html = f'<div class="aa-profile-subtitle">{_html_escape(item["subtitle"])}</div>'
    album_html = ""
    if item.get("detail_html"):
        album_html = item["detail_html"]
    elif item.get("detail"):
        album_html = (
            f'<p class="aa-profile-album-line"><span class="aa-profile-album-label">Album</span> '
            f'<span class="aa-profile-album-plain">{_html_escape(item["detail"])}</span></p>'
        )
    chips_html = item.get("detail_md") or ""
    st.markdown(
        (
            f'<div class="aa-profile-card">'
            f'<div class="aa-profile-visual-thumb"><img src="{img_url}" alt="" loading="lazy" /></div>'
            f'<div class="aa-profile-card-body">'
            f'<div class="aa-profile-kicker">{kicker}</div>'
            f'<div class="aa-profile-title-row">{title_block}</div>'
            f"{subtitle_html}{album_html}{chips_html}"
            f"</div></div>"
        ),
        unsafe_allow_html=True,
    )


def _render_visual_gallery(title: str, caption: str, items: list[dict[str, str]], *, columns: int = 5) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    st.caption(caption)
    for start in range(0, len(items), columns):
        chunk = items[start : start + columns]
        cols = st.columns(len(chunk), gap="small")
        for col, item in zip(cols, chunk):
            with col:
                _render_visual_card(item)


def _profile_visual_items(profile_artifacts: ProfileArtifacts, summary: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    aa_index = get_music4all_aa_song_index()
    raw = profile_artifacts.raw_profile or {}
    songs = raw.get("songs") or []
    if not aa_index or not songs:
        return [], []

    representative_artists = {
        _normalize_label(name): i
        for i, name in enumerate(summary.get("representative_artists", []))
    }
    representative_tracks = {
        (_normalize_label(item.get("artist")), _normalize_label(item.get("song"))): i
        for i, item in enumerate(summary.get("representative_tracks", []))
        if isinstance(item, dict)
    }

    artist_candidates: dict[str, tuple[tuple[int, int], dict[str, str]]] = {}
    album_candidates: dict[str, tuple[tuple[int, int], dict[str, str]]] = {}

    for retrieval_rank, song_row in enumerate(songs):
        song_id = str(song_row.get("song_id") or "")
        enrichment = aa_index.get(song_id)
        if not enrichment:
            continue
        info = song_row.get("info") or {}
        track_artist = str(info.get("artist") or _aa_text(enrichment, "album_artist") or _aa_text(enrichment, "artist_name"))
        track_title = str(info.get("song") or _aa_text(enrichment, "music4all_song") or song_id)

        artist_image = _aa_text(enrichment, "artist_image_url")
        artist_name = _aa_text(enrichment, "artist_name") or track_artist
        if artist_image and artist_name:
            artist_key = _aa_text(enrichment, "artist_mbid") or _normalize_label(artist_name)
            priority = representative_artists.get(_normalize_label(artist_name), 999)
            item = {
                "image_url": artist_image,
                "kicker": "Artist",
                "title": artist_name,
                "url": _aa_text(enrichment, "artist_lastfm_url"),
                "subtitle": " · ".join(
                    part
                    for part in (
                        _aa_text(enrichment, "artist_type"),
                        _aa_text(enrichment, "artist_country"),
                    )
                    if part
                ),
                "detail_md": _linked_tag_chips(_aa_text(enrichment, "artist_genres")),
            }
            score = (priority, retrieval_rank)
            current = artist_candidates.get(artist_key)
            if current is None or score < current[0]:
                artist_candidates[artist_key] = (score, item)

        cover_url = _aa_text(enrichment, "album_cover_url")
        album_name = _aa_text(enrichment, "album_name") or _aa_text(enrichment, "music4all_album_name")
        if cover_url and album_name:
            album_key = _aa_text(enrichment, "album_mbid") or f"{_normalize_label(track_artist)}::{_normalize_label(album_name)}"
            priority = representative_tracks.get((_normalize_label(track_artist), _normalize_label(track_title)), 999)
            item = {
                "image_url": cover_url,
                "kicker": "Track",
                "title": track_title,
                "subtitle": track_artist,
                "detail_html": (
                    f'<p class="aa-profile-album-line"><span class="aa-profile-album-label">Album</span> '
                    f'<a class="aa-profile-album-link" href="{_html_escape(_aa_text(enrichment, "album_lastfm_url"))}" '
                    f'target="_blank" rel="noopener noreferrer">{_html_escape(album_name)}</a></p>'
                    if _aa_text(enrichment, "album_lastfm_url")
                    else (
                        f'<p class="aa-profile-album-line"><span class="aa-profile-album-label">Album</span> '
                        f'<span class="aa-profile-album-plain">{_html_escape(album_name)}</span></p>'
                    )
                ),
                "detail": "",
                "detail_md": _linked_tag_chips(_aa_text(enrichment, "album_genres")),
            }
            score = (priority, retrieval_rank)
            current = album_candidates.get(album_key)
            if current is None or score < current[0]:
                album_candidates[album_key] = (score, item)

    artists = [item for _, item in sorted(artist_candidates.values(), key=lambda pair: pair[0])[:5]]
    albums = [item for _, item in sorted(album_candidates.values(), key=lambda pair: pair[0])[:5]]
    return artists, albums


def _render_profile_visual_context(profile_artifacts: ProfileArtifacts, summary: dict) -> None:
    artists, albums = _profile_visual_items(profile_artifacts, summary)
    if not artists and not albums:
        return

    with st.expander("Visual listening context", expanded=True):
        _inject_profile_visual_styles()
        st.caption("Images are matched from Music4All A+A using the retrieved song IDs; track artwork uses album covers.")
        if artists:
            _render_visual_gallery(
                "Retrieved artists",
                "Artist images from the A+A artist metadata.",
                artists,
            )
        if albums:
            _render_visual_gallery(
                "Retrieved tracks / album covers",
                "Track artwork is represented by the matched album cover.",
                albums,
            )


def _render_profile_section(profile_artifacts: ProfileArtifacts) -> None:
    prompt = profile_artifacts.prompt
    summary = (prompt or {}).get("input_summary") or profile_artifacts.summary or {}
    if not prompt:
        st.info("No prompt artifact is available yet. Click `Load profile` to build or load it.")
        return

    st.subheader("Recent listening snapshot")
    audio_profile = summary.get("audio_profile", {})
    metrics = st.columns(4)
    metrics[0].metric("Danceability", audio_profile.get("danceability_mean"))
    metrics[1].metric("Energy", audio_profile.get("energy_mean"))
    metrics[2].metric("Valence", audio_profile.get("valence_mean"))
    metrics[3].metric("Tempo", audio_profile.get("tempo_mean"))
    if summary.get("mood_summary"):
        st.caption("Mood summary: " + ", ".join(summary["mood_summary"]))

    st.divider()
    st.subheader("LLM listener profile")
    st.caption(
        "This section is generated from retrieved songs in Phase 2 (metadata/tags + summary), "
        "then refined by the selected LLM into profile text and style keywords."
    )
    st.write(prompt.get("profile_paragraph", ""))

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Style keywords**")
        st.write(", ".join(prompt.get("style_keywords", [])) or "None")
        st.markdown("**Top genres**")
        st.write(", ".join(summary.get("top_genres", [])) or "None")
        st.markdown("**Top tags**")
        st.write(", ".join(summary.get("top_tags", [])) or "None")
        language_profile = summary.get("language_profile", {})
        if language_profile:
            st.markdown("**Language / vocal mode**")
            dominant_mode = language_profile.get("dominant_mode", "unknown")
            instrumental_ratio = language_profile.get("instrumental_ratio")
            vocal_ratio = language_profile.get("vocal_or_language_ratio")
            st.write(
                f"Dominant mode: `{dominant_mode}` | "
                f"Instrumental ratio: {_format_metric_value(instrumental_ratio)} | "
                f"Vocal/language ratio: {_format_metric_value(vocal_ratio)}"
            )
            language_counts = language_profile.get("language_counts", {})
            if language_counts:
                counts_preview = ", ".join(
                    f"{label}: {count}" for label, count in list(language_counts.items())[:6]
                )
                st.caption("Language counts: " + counts_preview)
    with right:
        st.markdown("**Representative artists**")
        st.write(", ".join(summary.get("representative_artists", [])) or "None")
        st.markdown("**Representative tracks**")
        representative_tracks = summary.get("representative_tracks", [])
        if representative_tracks:
            for item in representative_tracks:
                st.write(f"- {item.get('artist', 'Unknown')} - {item.get('song', 'Unknown')}")
        else:
            st.write("None")

    _render_profile_visual_context(profile_artifacts, summary)

    if profile_artifacts.validation:
        with st.expander("Validation summary"):
            validation = profile_artifacts.validation
            if validation.get("human_readable_summary"):
                st.write(validation["human_readable_summary"])
            st.json(validation)


def _extract_llm_generation_prompt(profile_artifacts: ProfileArtifacts) -> str:
    prompt_payload = profile_artifacts.prompt or {}
    return (
        prompt_payload.get("suno_generation_prompt")
        or prompt_payload.get("generation_prompt")
        or prompt_payload.get("prompt")
        or ""
    )


def _render_retrieval_snapshot(profile_artifacts: ProfileArtifacts) -> None:
    raw = profile_artifacts.raw_profile or {}
    retrieval = raw.get("retrieval") or {}
    songs = raw.get("songs") or []
    if not raw:
        st.info("No retrieval JSON yet. Build/load profile first.")
        return

    stats = st.columns(4)
    stats[0].metric("Top-k requested", retrieval.get("top_k_requested", raw.get("top_k")))
    stats[1].metric("Top-k returned", raw.get("top_k"))
    stats[2].metric("Min similarity", retrieval.get("min_similarity"))
    stats[3].metric("Candidates after filter", retrieval.get("candidate_count_after_filter"))

    if retrieval.get("threshold_relaxed"):
        st.warning("Similarity threshold was relaxed because no songs passed the cutoff.")

    if songs:
        total_songs = len(songs)
        preview_limit = min(10, total_songs)
        st.caption(f"Showing first {preview_limit} songs (out of {total_songs} returned).")
        preview = []
        for row in songs[:preview_limit]:
            info = row.get("info") or {}
            metadata = row.get("metadata") or {}
            genres = row.get("genres")
            tags = row.get("tags")
            preview.append(
                {
                    "rank": row.get("rank"),
                    "song_id": row.get("song_id"),
                    "artist": info.get("artist"),
                    "song": info.get("song"),
                    "score": row.get("similarity_score"),
                    "genres": ", ".join(genres) if isinstance(genres, list) else genres,
                    "tags": ", ".join(tags) if isinstance(tags, list) else tags,
                    "danceability": metadata.get("danceability"),
                    "energy": metadata.get("energy"),
                    "valence": metadata.get("valence"),
                    "tempo": metadata.get("tempo"),
                    "release": metadata.get("release"),
                }
            )
        st.dataframe(preview, width="stretch")
        with st.expander("Show more retrieval rows"):
            show_n = st.slider(
                "Rows to display",
                min_value=1,
                max_value=total_songs,
                value=min(20, total_songs),
                step=1,
                key=f"retrieval_rows::{profile_artifacts.user_id}",
            )
            full_rows = []
            for row in songs[:show_n]:
                info = row.get("info") or {}
                metadata = row.get("metadata") or {}
                genres = row.get("genres")
                tags = row.get("tags")
                full_rows.append(
                    {
                        "rank": row.get("rank"),
                        "song_id": row.get("song_id"),
                        "artist": info.get("artist"),
                        "song": info.get("song"),
                        "score": row.get("similarity_score"),
                        "genres": ", ".join(genres) if isinstance(genres, list) else genres,
                        "tags": ", ".join(tags) if isinstance(tags, list) else tags,
                        "danceability": metadata.get("danceability"),
                        "energy": metadata.get("energy"),
                        "valence": metadata.get("valence"),
                        "tempo": metadata.get("tempo"),
                        "release": metadata.get("release"),
                    }
                )
            st.dataframe(full_rows, width="stretch")
        with st.expander("Raw retrieval JSON"):
            st.json(raw)
    else:
        st.info("No songs in retrieval payload.")


def _render_custom_playlist_page() -> None:
    st.markdown("## Custom WAV → AI music")
    st.caption(
        "Upload one or more **.wav** files only (RIFF/WAVE). Listen to them below, then run the pipeline: "
        "pool CLAP embeddings → retrieval + profile → Suno → rerank. **No CSV.** Generated MP3s appear at the bottom."
    )
    work_root = REPO_ROOT / "outputs" / "custom_playlist_streamlit"

    with st.sidebar:
        st.subheader("Custom WAV")
        slug_raw = st.text_input(
            "Playlist / run label",
            value="my_playlist",
            help="Sanitized for folder names and synthetic user id.",
        )
        safe_slug = sanitize_segment(str(slug_raw).strip()) or "my_playlist"
        st.caption(f"Output folder: `outputs/custom_playlist_streamlit/{safe_slug}/`")
        encoder = st.selectbox("CLAP encoder (WAV pool)", ["finetuned", "zeroshot", "auto"], index=0)
        top_k = st.number_input("Retrieval top-k", min_value=5, max_value=100, value=10, step=1)
        min_sim_on = st.checkbox("Min similarity filter", value=False)
        min_similarity: float | None = None
        if min_sim_on:
            min_similarity = float(
                st.number_input("Min similarity", min_value=-1.0, max_value=1.0, value=0.2, step=0.05)
            )
        openai_model = st.text_input("OpenAI model", value="gpt-5.4-mini")
        rebuild_profile = st.checkbox("Rebuild profile (ignore cache)", value=False)
        st.markdown("### Suno + rerank")
        generation_model = st.text_input("Generation model", value="chirp-v4-5")
        num_calls = st.number_input("API calls", min_value=1, max_value=10, value=1, step=1)
        max_concurrency = st.number_input("Max concurrency", min_value=1, max_value=5, value=1, step=1)
        negative_prompt = st.text_input("Negative prompt", value="")
        lyrics = st.text_area("Lyrics / cues (optional)", value="", height=80)
        tempo_hint_bpm = st.number_input("Tempo hint (BPM)", min_value=0, max_value=300, value=0, step=1)
        duration_hint_seconds = st.number_input("Duration hint (s)", min_value=0, max_value=600, value=0, step=1)
        rerank_top_k = st.number_input("Rerank keep top-k", min_value=1, max_value=10, value=2, step=1)
        rerank_encoder = st.selectbox("Rerank encoder", ["finetuned", "zeroshot", "auto"], index=0)
        div_text = st.text_input("Diversity threshold (optional)", value="")

    uploaded_wavs = st.file_uploader(
        "WAV files (.wav only)",
        type=["wav"],
        accept_multiple_files=True,
        help="Only RIFF/WAVE .wav files. Order is preserved for embedding pooling (use unique filenames).",
    ) or []

    col_a, col_b = st.columns(2)
    with col_a:
        load_clicked = st.button("Save clips & preview", use_container_width=True)
    with col_b:
        run_clicked = st.button("Generate music", type="primary", use_container_width=True)

    if load_clicked:
        if not uploaded_wavs:
            st.warning("Please upload at least one .wav file.")
            return
        wav_payloads: list[tuple[str, bytes]] = []
        for uf in uploaded_wavs:
            if Path(uf.name).suffix.lower() != ".wav":
                st.error(f"Only .wav files are allowed: {uf.name!r}")
                return
            wav_payloads.append((uf.name, uf.getvalue()))
        try:
            clips_dir = work_root / safe_slug / "clips_30s"
            write_wav_files_to_clips_dir(clips_dir, wav_payloads, clear=True)
            wav_paths = clip_paths_in_upload_order(clips_dir, wav_payloads)
            preview = [
                {"#": i, "filename": p.name, "wav_path": str(p)}
                for i, p in enumerate(wav_paths, start=1)
            ]
        except Exception as exc:
            st.error(str(exc))
            return
        st.session_state["custom_pl_slug"] = safe_slug
        st.session_state["custom_pl_paths"] = [str(p) for p in wav_paths]
        st.session_state["custom_pl_preview"] = preview
        st.success(f"Saved {len(preview)} clip(s). You can play them below.")

    if run_clicked:
        paths_raw = st.session_state.get("custom_pl_paths")
        use_slug = st.session_state.get("custom_pl_slug") or safe_slug
        if not paths_raw:
            st.warning('Click "Save clips & preview" first after uploading your .wav files.')
            return
        wav_paths_ordered = [Path(p) for p in paths_raw]
        for p in wav_paths_ordered:
            if not p.is_file():
                st.error(f"Missing WAV on disk: {p}. Load the playlist again.")
                return
        try:
            parsed_div = float(div_text) if str(div_text).strip() else None
        except ValueError:
            st.error("Diversity threshold must be empty or a number.")
            return
        with st.spinner("Running custom playlist pipeline (embed → profile → Suno → rerank)…"):
            try:
                result = run_custom_playlist_pipeline(
                    wav_paths_ordered=wav_paths_ordered,
                    participant_slug=use_slug,
                    work_root=work_root,
                    encoder=str(encoder),
                    top_k=int(top_k),
                    min_similarity=min_similarity,
                    openai_model=str(openai_model),
                    generation_model=str(generation_model),
                    num_calls=int(num_calls),
                    max_concurrency=int(max_concurrency),
                    negative_prompt=negative_prompt.strip() or None,
                    lyrics=str(lyrics),
                    tempo_hint_bpm=int(tempo_hint_bpm) or None,
                    duration_hint_seconds=int(duration_hint_seconds) or None,
                    rerank_top_k=int(rerank_top_k),
                    rerank_encoder=str(rerank_encoder),
                    rerank_diversity_threshold=parsed_div,
                    rebuild_profile=bool(rebuild_profile),
                )
            except Exception as exc:
                st.exception(exc)
                return
        st.session_state["custom_pl_result"] = result
        st.success(f"Done. Run id: `{result['run_id']}`")

    preview = st.session_state.get("custom_pl_preview")
    if preview:
        st.subheader("Your WAV clips")
        st.dataframe(pd.DataFrame(preview), width="stretch", hide_index=True)
        st.markdown("**Listen to uploads**")
        for row in preview:
            p = Path(str(row["wav_path"]))
            label = f"{row.get('#', '')}. {row.get('filename', p.name)}"
            with st.expander(label):
                if p.is_file():
                    st.audio(p.read_bytes(), format="audio/wav")
                else:
                    st.warning(f"File not found: {p}. Click **Save clips & preview** again.")

    result = st.session_state.get("custom_pl_result")
    if result:
        st.divider()
        st.subheader("Generated music")
        st.caption(
            f"Run id: `{result['run_id']}` · Synthetic user: `{result['synthetic_user_id']}` · "
            f"Manifest: `{result['manifest_path']}`"
        )
        uid = str(result["synthetic_user_id"])
        prof_var = str(result["profile_variant"])
        profile_artifacts = load_profile_artifacts(uid, profile_variant=prof_var)
        try:
            run_artifacts = load_generation_run(Path(result["run_root"]))
        except Exception as exc:
            st.error(f"Could not load generation run: {exc}")
            run_artifacts = None
        if run_artifacts is not None:
            _render_generation_section(run_artifacts)
        with st.expander("Pipeline details (retrieval, profile, prompt, embedding plot)", expanded=False):
            st.markdown("#### Retrieval snapshot")
            _render_retrieval_snapshot(profile_artifacts)
            st.divider()
            _render_profile_section(profile_artifacts)
            llm_prompt_text = _extract_llm_generation_prompt(profile_artifacts)
            if llm_prompt_text:
                st.divider()
                st.subheader("LLM generation prompt")
                st.text_area("Prompt text", value=llm_prompt_text, height=120, disabled=True, key="custom_pl_prompt_txt")
            if run_artifacts is not None:
                st.divider()
                _render_visualization_section(uid, run_artifacts)


def _render_procedure_brief() -> None:
    st.markdown("### Pipeline overview")
    st.markdown(
        """
```text
[Phase 1: User Embedding]
listening_history + song CLAP embeddings
-> pick recent-k songs from listening history
-> remove outlier songs (medoid filter)
-> weighted average (more recent + repeated songs get more weight)
-> normalize user embedding

                ↓

[Phase 2: Retrieval + Prompt]
user embedding
-> find nearest songs by cosine (top-k, min similarity)
-> attach metadata + genres/tags from CSV
-> export raw profile JSON
-> ask LLM to refine a better music-generation prompt

                ↓

[Phase 3: Generation + Selection]
prompt
-> generate multiple songs with Suno
-> rerank by CLAP similarity to user embedding
-> diversity filter + evaluation + visualization
```
"""
    )


def _compute_phase1_pca_data(
    *,
    user_id: str,
    embedding_variant: str,
    recent_k: int,
    medoid_threshold: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    phase1_resources = get_phase1_static_resources()
    song_embs = phase1_resources["song_embs"]
    history_df = phase1_resources["history_df"]
    info_df = phase1_resources["info_df"]
    song2idx = phase1_resources["song2idx"]
    user_df = history_df.loc[history_df["user_id"] == user_id].copy()
    user_df = user_df.sort_values("timestamp", ascending=False, na_position="last")
    recent = user_df.head(max(1, int(recent_k))).copy()
    if recent.empty:
        raise ValueError("No listening history rows found for selected user.")
    recent["rank"] = np.arange(len(recent), dtype=np.int32)
    agg = (
        recent.groupby("song_id", as_index=False)
        .agg(min_rank=("rank", "min"), play_count=("song_id", "count"))
        .sort_values("min_rank")
        .reset_index(drop=True)
    )
    agg = agg.merge(info_df[["song_id", "artist", "song", "album_name"]], on="song_id", how="left")
    agg = agg[agg["song_id"].astype(str).isin(song2idx)].reset_index(drop=True)
    if agg.empty:
        raise ValueError("No recent songs matched song embedding IDs.")

    idxs = np.array([song2idx[str(sid)] for sid in agg["song_id"].tolist()], dtype=np.int64)
    embs = song_embs[idxs].astype(np.float32)
    sims = embs @ embs.T
    mean_sims = sims.mean(axis=1)
    medoid_local_idx = int(np.argmax(mean_sims))
    medoid_song_id = str(agg.iloc[medoid_local_idx]["song_id"])
    keep_mask = sims[medoid_local_idx] >= float(medoid_threshold)

    user_embs, user_ids = get_variant_user_embeddings(embedding_variant)
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    if user_id not in user_to_idx:
        raise ValueError(f"user_id not found in selected embedding variant: {user_id}")
    user_vec = user_embs[user_to_idx[user_id]].astype(np.float32)

    labels: list[str] = []
    kinds: list[str] = []
    meta_rows: list[dict[str, object]] = []
    vecs: list[np.ndarray] = []
    for i, row in agg.iterrows():
        sid = str(row["song_id"])
        kind = "medoid" if i == medoid_local_idx else ("kept" if bool(keep_mask[i]) else "filtered")
        cos_medoid = float(sims[medoid_local_idx, i])
        artist = str(row.get("artist") or "Unknown")
        song_name = str(row.get("song") or sid)
        album_name = str(row.get("album_name") or "—")
        vecs.append(embs[i])
        labels.append(f"{artist} - {song_name}")
        kinds.append(kind)
        meta_rows.append(
            {
                "type": kind,
                "song_id": sid,
                "artist": artist,
                "song": song_name,
                "album": album_name,
                "recency_rank": int(row["min_rank"]),
                "repeat_count": int(row["play_count"]),
                "cos_to_medoid": round(cos_medoid, 4),
                "is_medoid": sid == medoid_song_id,
            }
        )
    vecs.append(user_vec)
    labels.append("USER_EMBEDDING")
    kinds.append("user")

    X = np.vstack(vecs).astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]

    plot_rows: list[dict[str, object]] = []
    for i, (kind, label) in enumerate(zip(kinds, labels)):
        plot_rows.append(
            {
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "group": kind,
                "label": label,
                "song_id": meta_rows[i]["song_id"] if i < len(meta_rows) else "USER",
                "artist": meta_rows[i]["artist"] if i < len(meta_rows) else "USER",
                "song": meta_rows[i]["song"] if i < len(meta_rows) else "USER_EMBEDDING",
                "album": meta_rows[i]["album"] if i < len(meta_rows) else "—",
                "recency_rank": meta_rows[i]["recency_rank"] if i < len(meta_rows) else None,
                "repeat_count": meta_rows[i]["repeat_count"] if i < len(meta_rows) else None,
                "cos_to_medoid": meta_rows[i]["cos_to_medoid"] if i < len(meta_rows) else None,
            }
        )
    return pd.DataFrame(plot_rows), meta_rows


def _render_track_card(track, index: int) -> None:
    st.markdown(f"### {index}. {track.title}")
    left, right = st.columns([1, 2])
    with left:
        if track.cover_large_url or track.cover_url:
            st.image(track.cover_large_url or track.cover_url, width="stretch")
        else:
            st.caption("No cover image available.")
    with right:
        badges = []
        if track.is_selected:
            badges.append("selected")
        if track.call_index is not None:
            badges.append(f"call {track.call_index}")
        if track.variant_index is not None:
            badges.append(f"variant {track.variant_index}")
        if badges:
            st.caption(" | ".join(badges))

        if track.rerank_score is not None:
            st.metric("CLAP rerank score", f"{track.rerank_score:.4f}")
        if track.duration_seconds is not None:
            st.caption(f"Duration: {track.duration_seconds:.2f}s")

        audio_bytes = read_binary_file(track.path)
        if audio_bytes is not None:
            st.audio(audio_bytes, format="audio/mpeg")
        elif track.source_url:
            st.audio(track.source_url, format="audio/mpeg")
        else:
            st.warning("Audio file is missing.")

        if track.lyric_text:
            with st.expander("Lyrics / text companion"):
                st.text(track.lyric_text)

        with st.expander("Track details"):
            st.write(f"Local audio path: `{track.path}`")
            if track.metadata_path:
                st.write(f"Metadata path: `{track.metadata_path}`")
            if track.lyric_path:
                st.write(f"Lyrics path: `{track.lyric_path}`")
            if track.prompt:
                st.write("Prompt:")
                st.write(track.prompt)
            if track.style:
                st.write("Style:")
                st.write(track.style)


def _render_generation_section(run_artifacts: GenerationRunArtifacts | None) -> None:
    st.subheader("Generated tracks")
    if run_artifacts is None:
        st.info("No generation run found yet. Use the form below to generate tracks.")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Run ID", run_artifacts.run_id)
    summary_cols[1].metric("Candidates", len(run_artifacts.tracks))
    selected_count = len([track for track in run_artifacts.tracks if track.is_selected])
    with st.expander("Run diagnostics", expanded=False):
        st.caption("Loaded artifact paths and selection status for troubleshooting.")
        st.write("Run root:", str(run_artifacts.run_root))
        st.write("Rerank result path:", str(run_artifacts.rerank_path))
        st.write("Selected track count:", selected_count)
    summary_cols[2].metric("Selected", selected_count)
    summary_cols[3].metric("Provider", run_artifacts.manifest.get("provider", "unknown"))

    if run_artifacts.prompt_input:
        with st.expander("Generation prompt input"):
            profile_paragraph = run_artifacts.prompt_input.get("profile_paragraph")
            style_keywords = run_artifacts.prompt_input.get("style_keywords") or []
            if profile_paragraph:
                st.markdown("**Profile paragraph used for generation**")
                st.write(profile_paragraph)
            if style_keywords:
                st.markdown("**Style keywords used**")
                st.write(", ".join(style_keywords))
            if run_artifacts.prompt_input_path:
                st.caption(f"Prompt input path: `{run_artifacts.prompt_input_path}`")
            st.json(run_artifacts.prompt_input)

    selected_tab, candidates_tab, report_tab = st.tabs(["Selected tracks", "All candidates", "Run report"])

    with selected_tab:
        selected_tracks = [track for track in run_artifacts.tracks if track.is_selected]
        if not selected_tracks:
            st.info("No selected tracks found in rerank output.")
        else:
            for idx, track in enumerate(selected_tracks, start=1):
                _render_track_card(track, idx)

    with candidates_tab:
        if not run_artifacts.tracks:
            st.info("No candidate tracks found.")
        else:
            for idx, track in enumerate(run_artifacts.tracks, start=1):
                _render_track_card(track, idx)

    with report_tab:
        if run_artifacts.report_markdown:
            st.markdown(run_artifacts.report_markdown)
        else:
            st.info("No markdown report found for this run.")
        if run_artifacts.prompt_input:
            with st.expander("Prompt input JSON"):
                st.json(run_artifacts.prompt_input)
        with st.expander("Manifest JSON"):
            st.json(run_artifacts.manifest)
        if run_artifacts.rerank:
            with st.expander("Rerank JSON"):
                st.json(run_artifacts.rerank)


def _render_saved_eval_summary(eval_artifacts: EvalArtifacts) -> None:
    if eval_artifacts.summary:
        run_summary = eval_artifacts.summary.get("run", {})
        panels = eval_artifacts.summary.get("metric_panels", {})
        personalization_panel = panels.get("personalization", {})
        reference_topk_value = personalization_panel.get("selected_reference_topk_mean_cosine_mean")
        if reference_topk_value is None:
            reference_topk_value = personalization_panel.get("selected_reference_topn_mean_cosine_mean")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Eval encoder", run_summary.get("encoder", "unknown"))
        summary_cols[1].metric("Eval candidates", run_summary.get("candidate_count"))
        summary_cols[2].metric("Eval selected", run_summary.get("selected_count"))
        summary_cols[3].metric("Recent-K", eval_artifacts.summary.get("reference_set", {}).get("recent_k"))

        personalization_col, diversity_col, risk_col = st.columns(3)
        with personalization_col:
            st.markdown("**Personalization**")
            st.metric(
                "User alignment",
                _format_metric_value(personalization_panel.get("selected_user_embedding_cosine_mean")),
            )
            st.metric(
                "Centroid alignment",
                _format_metric_value(personalization_panel.get("selected_recent_centroid_cosine_mean")),
            )
            st.metric(
                "Reference top-k",
                _format_metric_value(reference_topk_value),
            )

        with diversity_col:
            st.markdown("**Diversity**")
            st.metric(
                "Selected pairwise cosine",
                _format_metric_value(panels.get("diversity", {}).get("selected_mean_pairwise_cosine")),
            )
            st.metric(
                "Selected nearest-neighbor",
                _format_metric_value(panels.get("diversity", {}).get("selected_mean_nearest_neighbor_cosine")),
            )
            st.caption("Lower values usually mean the kept songs are less redundant.")

        with risk_col:
            st.markdown("**Risk**")
            st.metric(
                "Selected too-close count",
                _format_metric_value(panels.get("risk", {}).get("selected_too_close_to_reference_count")),
            )
            st.metric(
                "Candidate too-close count",
                _format_metric_value(panels.get("risk", {}).get("candidate_too_close_to_reference_count")),
            )
            st.caption("Tracks flagged here may be overly close to one reference song.")

        with st.expander("Eval summary JSON"):
            st.json(eval_artifacts.summary)

    if eval_artifacts.report_markdown:
        with st.expander("Eval report"):
            st.markdown(eval_artifacts.report_markdown)


def _render_visualization_section(user_id: str, run_artifacts: GenerationRunArtifacts | None) -> None:
    st.subheader("Embedding visualization")
    if run_artifacts is None:
        st.info("Generate or load a run first to visualize it in embedding space.")
        return

    eval_artifacts = load_eval_artifacts(user_id, run_artifacts.run_id)
    has_saved_plot = eval_artifacts.plot_path.exists()
    mode_options = ["saved eval plot", "live compute"]
    default_mode = "saved eval plot" if has_saved_plot else "live compute"
    mode_key = f"viz_mode::{user_id}::{run_artifacts.run_id}"
    selected_mode = st.radio(
        "Visualization mode",
        options=mode_options,
        horizontal=True,
        index=mode_options.index(default_mode),
        key=mode_key,
    )

    if selected_mode == "saved eval plot":
        if not has_saved_plot:
            st.info(
                "No saved eval plot was found for this run yet. "
                "Run eval with `--save-plot`, or switch to `live compute`."
            )
            return
        st.image(str(eval_artifacts.plot_path), width="stretch")
        st.caption(f"Saved eval plot: `{eval_artifacts.plot_path}`")
        _render_saved_eval_summary(eval_artifacts)
        return

    viz_state_key = f"show_viz::{user_id}::{run_artifacts.run_id}"
    viz_button_key = f"render_viz_button::{user_id}::{run_artifacts.run_id}"
    if st.button("Render embedding space", key=viz_button_key):
        st.session_state[viz_state_key] = True

    if not st.session_state.get(viz_state_key):
        st.caption("Rendering the plot loads CLAP and embeds recent listens plus generated tracks, so it is kept explicit.")
        return

    with st.spinner("Building embedding-space visualization..."):
        figure, plot_df, encoder_name = build_user_generation_figure(
            user_id=user_id,
            run_root=run_artifacts.run_root,
        )
    st.pyplot(figure, clear_figure=True)
    st.caption(f"Visualization encoder: {encoder_name}")
    st.dataframe(
        plot_df[["encoder", "group", "label", "rerank_score", "path", "x", "y"]],
        width="stretch",
    )


def main() -> None:
    st.title("Gen4Rec Streamlit Demo")

    global_user_id = st.session_state.get("selected_user_id", "")
    with st.sidebar:
        st.header("Controls")
        app_mode = st.radio(
            "App section",
            options=[
                "Embedding Retrieval (Base vs Finetuned)",
                "Generate AI Song",
                "Custom WAV → AI music",
            ],
            index=0,
            horizontal=False,
            help="Use one app with multiple modules.",
        )
        show_query_compare = app_mode == "Embedding Retrieval (Base vs Finetuned)"
        show_generate_page = app_mode == "Generate AI Song"
        show_custom_playlist = app_mode == "Custom WAV → AI music"
        generate_section = "Overview"
        if show_generate_page:
            generate_section = st.radio(
                "Generate section",
                options=[
                    "Overview",
                    GENERATE_SECTION_FULL_PIPELINE,
                    "Phase 1 - User Embedding",
                    "Phase 2 - Retrieval + Prompt",
                    "Phase 3 - Generate + Rerank + Evaluate",
                ],
                index=0,
                horizontal=False,
            )
            try:
                all_known_users = get_all_known_users()
            except Exception:
                all_known_users = []
            if all_known_users:
                selected_idx = all_known_users.index(global_user_id) if global_user_id in all_known_users else 0
                global_user_id = st.selectbox("Select user", all_known_users, index=selected_idx)
                st.session_state["selected_user_id"] = global_user_id
            else:
                st.warning("No users found in listening history.")
        show_phase1_controls = show_generate_page and (
            generate_section == "Phase 1 - User Embedding" or generate_section == GENERATE_SECTION_FULL_PIPELINE
        )
        show_phase2_controls = show_generate_page and (
            generate_section == "Phase 2 - Retrieval + Prompt" or generate_section == GENERATE_SECTION_FULL_PIPELINE
        )
        show_phase3_controls = show_generate_page and (
            generate_section == "Phase 3 - Generate + Rerank + Evaluate"
            or generate_section == GENERATE_SECTION_FULL_PIPELINE
        )

        phase1_user_id = st.session_state.get("phase1_user_id", "")
        phase1_embedding_variant = st.session_state.get("phase1_embedding_variant", "")
        build_recent_k = 10
        build_decay_lambda = 0.08
        build_medoid_threshold = 0.2
        build_min_keep = 5
        build_embedding_clicked = False
        if show_phase1_controls:
            st.markdown("### Phase 1: User embedding")
            phase1_user_id = global_user_id
            st.session_state["phase1_user_id"] = phase1_user_id

            phase1_variants = get_available_embedding_variants()
            phase1_variant_options = ["(none)"] + phase1_variants
            selected_phase1_variant = phase1_embedding_variant if phase1_embedding_variant in phase1_variants else "(none)"
            phase1_variant_idx = phase1_variant_options.index(selected_phase1_variant)
            selected_phase1_variant = st.selectbox(
                "Embedding variant for Phase 1 PCA (optional)",
                phase1_variant_options,
                index=phase1_variant_idx,
            )
            phase1_embedding_variant = "" if selected_phase1_variant == "(none)" else selected_phase1_variant
            st.session_state["phase1_embedding_variant"] = phase1_embedding_variant

            build_recent_k = st.number_input(
                "Recent-k",
                min_value=1,
                max_value=200,
                value=10,
                step=1,
                help="Use the most recent K listening events per user for embedding construction.",
            )
            build_decay_lambda = st.number_input(
                "Decay lambda",
                min_value=0.0,
                max_value=2.0,
                value=0.08,
                step=0.01,
                help="Recency weight: w_time = exp(-lambda * rank), where rank=0 is most recent.",
            )
            build_medoid_threshold = st.number_input(
                "Medoid threshold",
                min_value=-1.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Medoid = song with max average cosine; keep songs with cos(e_i, e_medoid) >= threshold.",
            )
            build_min_keep = st.number_input(
                "Min keep",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
                help="After coherence filtering, keep at least this many nearest-to-medoid songs.",
            )
            build_embedding_clicked = st.button("Build user embedding variant")

    if show_query_compare:
        st.caption("Compare retrieval behavior in zeroshot vs finetuned embedding spaces.")
        render_query_compare_page()
        return

    if show_custom_playlist:
        _render_custom_playlist_page()
        return

    if build_embedding_clicked:
        with st.spinner("Building user embedding variant..."):
            try:
                build_result = build_user_embedding_variant(
                    recent_k=int(build_recent_k),
                    decay_lambda=float(build_decay_lambda),
                    medoid_threshold=float(build_medoid_threshold),
                    min_keep=int(build_min_keep),
                )
            except Exception as exc:
                st.error(str(exc))
                st.stop()
        get_available_embedding_variants.clear()
        get_available_users.clear()
        if build_result["created"]:
            st.success(f"Built embedding variant: {build_result['variant']}")
        else:
            st.info(f"Variant already exists, reused: {build_result['variant']}")
        st.rerun()

    try:
        embedding_variants = get_available_embedding_variants()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    embedding_variant = st.session_state.get("selected_embedding_variant", "")
    user_id = global_user_id
    top_k = int(st.session_state.get("phase2_top_k", 5))
    min_similarity: float | None = st.session_state.get("phase2_min_similarity")
    exclude_recent = bool(st.session_state.get("phase2_exclude_recent", True))
    openai_model = str(st.session_state.get("phase2_openai_model", "gpt-5.4-mini"))
    profile_source_mode = str(st.session_state.get("phase2_profile_source_mode", "auto"))
    manual_profile_variant = str(st.session_state.get("phase2_manual_profile_variant", ""))
    active_profile_user = str(st.session_state.get("active_profile_user", ""))
    active_profile_mode = str(st.session_state.get("active_profile_mode", ""))
    active_profile_variant = st.session_state.get("active_profile_variant")
    active_auto_profile_variant = str(st.session_state.get("active_auto_profile_variant", ""))
    profile_ready = bool(st.session_state.get("profile_ready", False))
    load_profile_clicked = False

    if show_phase2_controls or show_phase3_controls:
        if not embedding_variants:
            st.warning("No user embedding variants were found yet. Build one in sidebar Phase 1 to continue.")
            st.stop()

        if not embedding_variant or embedding_variant not in embedding_variants:
            embedding_variant = embedding_variants[0]
            st.session_state["selected_embedding_variant"] = embedding_variant

        if show_phase2_controls:
            default_variant_index = embedding_variants.index(embedding_variant) if embedding_variant in embedding_variants else 0
            with st.sidebar:
                embedding_variant = st.selectbox(
                    "User embedding variant",
                    embedding_variants,
                    index=default_variant_index,
                    help="Embedding variant used for retrieval/profile and subsequent generation.",
                )
                st.session_state["selected_embedding_variant"] = embedding_variant

        try:
            available_users = get_available_users(embedding_variant)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        if not available_users:
            st.error(f"No users are available in variant `{embedding_variant}`.")
            st.stop()

        if not user_id:
            user_id = available_users[0]
            st.session_state["selected_user_id"] = user_id
        elif user_id not in available_users:
            fallback_user = available_users[0]
            st.warning(
                f"Selected user `{user_id}` is not in embedding variant `{embedding_variant}`. "
                f"Auto-switched to `{fallback_user}`."
            )
            user_id = fallback_user
            st.session_state["selected_user_id"] = user_id

        if show_phase2_controls:
            existing_profile_variants = list_profile_variants_for_user(user_id)
            with st.sidebar:
                st.markdown("### Phase 2: Profile build")
                top_k = st.number_input(
                    "Retrieval top-k",
                    min_value=5,
                    max_value=100,
                    value=int(top_k),
                    step=1,
                    help="Number of nearest songs used to build retrieval JSON, summary, and prompt.",
                )
                enable_min_similarity = st.checkbox(
                    "Enable min similarity filter",
                    value=min_similarity is not None,
                    help="Turn on to keep only retrieval hits with cosine similarity >= threshold.",
                )
                if enable_min_similarity:
                    min_similarity = float(
                        st.number_input(
                            "Min similarity threshold",
                            min_value=-1.0,
                            max_value=1.0,
                            value=float(min_similarity) if min_similarity is not None else 0.2,
                            step=0.05,
                            help="Cosine similarity cutoff for retrieval results.",
                        )
                    )
                else:
                    min_similarity = None
                exclude_recent = st.checkbox(
                    "Exclude recently listened songs",
                    value=bool(exclude_recent),
                    help="Mask songs already in the user's listening history before retrieval ranking.",
                )
                openai_model = st.text_input(
                    "OpenAI model",
                    value=openai_model,
                    help="Model used to generate profile paragraph and Suno prompt text.",
                )
                profile_source_mode = st.radio(
                    "Profile source",
                    options=["Build/reuse from current parameters", "Select existing profile"],
                    index=0 if profile_source_mode != "manual" else 1,
                    help="Use current parameters to build/reuse, or directly pick an existing profile artifact.",
                )
                profile_source_mode = "manual" if profile_source_mode == "Select existing profile" else "auto"
                if profile_source_mode == "manual":
                    profile_options = ["None"] + existing_profile_variants
                    default_profile = (
                        manual_profile_variant if manual_profile_variant in existing_profile_variants else "None"
                    )
                    manual_display = st.selectbox(
                        "Existing profile variant",
                        profile_options,
                        index=profile_options.index(default_profile),
                    )
                    manual_profile_variant = "" if manual_display == "None" else manual_display
                load_profile_clicked = st.button("Build or reuse profile")

            st.session_state["phase2_top_k"] = int(top_k)
            st.session_state["phase2_min_similarity"] = min_similarity
            st.session_state["phase2_exclude_recent"] = bool(exclude_recent)
            st.session_state["phase2_openai_model"] = openai_model
            st.session_state["phase2_profile_source_mode"] = profile_source_mode
            st.session_state["phase2_manual_profile_variant"] = manual_profile_variant

            # Strict behavior: changing source/selection invalidates loaded profile.
            if profile_source_mode == "manual":
                if not manual_profile_variant:
                    profile_ready = False
                    active_profile_variant = None
                    st.session_state["profile_ready"] = False
                    st.session_state["active_profile_variant"] = None
                    st.session_state["active_profile_mode"] = "manual"
                else:
                    # Manual + existing variant should load immediately without clicking.
                    st.session_state["active_profile_variant"] = manual_profile_variant
                    st.session_state["active_profile_mode"] = "manual"
                    st.session_state["active_profile_user"] = user_id
                    st.session_state["profile_ready"] = True
            else:
                current_auto_variant = build_profile_variant_tag(
                    embedding_variant=embedding_variant,
                    top_k=int(top_k),
                    min_similarity=min_similarity,
                    exclude_recent=exclude_recent,
                    openai_model=openai_model,
                )
                if (
                    active_profile_mode != "auto"
                    or active_auto_profile_variant != current_auto_variant
                    or active_profile_user != user_id
                ):
                    profile_ready = False
                    st.session_state["profile_ready"] = False

    profile_variant = ""
    profile_artifacts = ProfileArtifacts(
        user_id="",
        raw_profile_path=Path(),
        summary_path=Path(),
        prompt_path=Path(),
        validation_path=Path(),
        raw_profile=None,
        summary=None,
        prompt=None,
        validation=None,
    )
    run_artifacts = None
    if show_phase2_controls or show_phase3_controls:
        computed_profile_variant = build_profile_variant_tag(
            embedding_variant=embedding_variant,
            top_k=int(top_k),
            min_similarity=min_similarity,
            exclude_recent=exclude_recent,
            openai_model=openai_model,
        )
        profile_variant = manual_profile_variant if profile_source_mode == "manual" else computed_profile_variant
        if profile_ready and active_profile_user == user_id:
            active_variant_to_load = st.session_state.get("active_profile_variant")
            profile_artifacts = load_profile_artifacts(user_id, profile_variant=active_variant_to_load)
            if st.session_state.get("active_profile_mode") == "manual":
                st.caption(f"Using selected profile variant: `{active_variant_to_load or 'None'}`")
            else:
                st.caption(f"Profile variant key: `{active_variant_to_load}`")
        else:
            st.info("No profile loaded yet. Click `Build or reuse profile` in Phase 2.")

    if load_profile_clicked:
        if profile_source_mode == "manual":
            with st.spinner("Loading selected profile artifacts..."):
                if manual_profile_variant:
                    profile_artifacts = load_profile_artifacts(user_id, profile_variant=manual_profile_variant)
                else:
                    profile_artifacts = ProfileArtifacts(
                        user_id=user_id,
                        raw_profile_path=Path(),
                        summary_path=Path(),
                        prompt_path=Path(),
                        validation_path=Path(),
                        raw_profile=None,
                        summary=None,
                        prompt=None,
                        validation=None,
                    )
        else:
            with st.spinner("Building or reusing profile artifacts..."):
                profile_artifacts = build_or_load_profile(
                    user_id=user_id,
                    embedding_variant=embedding_variant,
                    top_k=int(top_k),
                    min_similarity=min_similarity,
                    exclude_recent=exclude_recent,
                    openai_model=openai_model,
                )
        if profile_source_mode == "manual":
            if manual_profile_variant:
                st.session_state["active_profile_variant"] = manual_profile_variant
                st.session_state["active_profile_mode"] = "manual"
                st.session_state["active_profile_user"] = user_id
                st.session_state["profile_ready"] = True
            else:
                st.session_state["active_profile_variant"] = None
                st.session_state["active_profile_mode"] = "manual"
                st.session_state["active_profile_user"] = user_id
                st.session_state["profile_ready"] = False
        else:
            st.session_state["active_profile_variant"] = profile_variant
            st.session_state["active_profile_mode"] = "auto"
            st.session_state["active_profile_user"] = user_id
            st.session_state["active_auto_profile_variant"] = profile_variant
            st.session_state["profile_ready"] = True

        if profile_artifacts.prompt and st.session_state.get("profile_ready"):
            st.success("Profile artifacts ready (existing artifacts are reused when parameters match).")
        elif profile_source_mode == "manual" and not manual_profile_variant:
            st.info("No profile selected (`None`).")
        else:
            st.warning("Profile prompt not ready yet.")

    if show_generate_page and generate_section == "Overview":
        st.markdown("## Generate AI Song")
        st.caption(
            f"Generate personalized songs from listening history. Current user: `{user_id or '—'}`."
        )
        _render_procedure_brief()

    if show_generate_page and (
        generate_section == "Phase 1 - User Embedding" or generate_section == GENERATE_SECTION_FULL_PIPELINE
    ):
        if generate_section == GENERATE_SECTION_FULL_PIPELINE:
            st.markdown("## Full pipeline — Phase 1: User embedding")
            st.caption(
                "Phases 1–3 are stacked on this page in order. The sidebar includes embedding build (Phase 1), "
                "retrieval / profile (Phase 2), and generation (Phase 3) controls."
            )
        else:
            st.markdown("## Phase 1 — User Embedding + Profile Setup")
        p1 = st.columns(3)
        p1[0].metric("Embedding variant", phase1_embedding_variant or "—")
        p1[1].metric("Recent-k", int(build_recent_k))
        p1[2].metric("Medoid threshold", float(build_medoid_threshold))
        st.caption("In Phase 1 controls: choose an existing embedding variant, or build a new one.")
        with st.spinner("Building Phase 1 PCA..."):
            try:
                if not phase1_user_id:
                    raise ValueError("Please select a user in Intro first.")
                if not phase1_embedding_variant:
                    raise ValueError("Select an embedding variant in Phase 1 controls, or build one first.")
                plot_df, phase1_rows = _compute_phase1_pca_data(
                    user_id=phase1_user_id,
                    embedding_variant=phase1_embedding_variant,
                    recent_k=int(build_recent_k),
                    medoid_threshold=float(build_medoid_threshold),
                )
                try:
                    px = importlib.import_module("plotly.express")

                    fig = px.scatter(
                        plot_df,
                        x="x",
                        y="y",
                        color="group",
                        symbol="group",
                        hover_name="label",
                        hover_data={
                            "song_id": True,
                            "artist": True,
                            "song": True,
                            "album": True,
                            "recency_rank": True,
                            "repeat_count": True,
                            "cos_to_medoid": True,
                            "x": False,
                            "y": False,
                        },
                        category_orders={"group": ["medoid", "kept", "filtered", "user"]},
                        color_discrete_map={
                            "medoid": "#ff8f00",
                            "kept": "#2e7d32",
                            "filtered": "#c62828",
                            "user": "#1565c0",
                        },
                        title="Phase 1 PCA: recent songs (kept vs filtered) + user embedding",
                    )
                    fig.update_layout(
                        legend_title_text="Group",
                        legend=dict(
                            x=1.02,
                            y=1.0,
                            xanchor="left",
                            yanchor="top",
                        ),
                        margin=dict(r=160),
                    )
                    fig.update_traces(marker=dict(size=9, opacity=0.9))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Tip: click legend items on the right to hide/show groups.")
                except Exception:
                    fig_mpl, ax = plt.subplots(figsize=(7, 5))
                    for grp, color, marker in [
                        ("medoid", "#ff8f00", "D"),
                        ("kept", "#2e7d32", "o"),
                        ("filtered", "#c62828", "x"),
                        ("user", "#1565c0", "*"),
                    ]:
                        part = plot_df[plot_df["group"] == grp]
                        if not part.empty:
                            ax.scatter(part["x"], part["y"], c=color, marker=marker, s=45, alpha=0.9, label=grp)
                    ax.set_title("Phase 1 PCA: recent songs (kept vs filtered) + user embedding")
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.legend(loc="upper right")
                    ax.grid(alpha=0.25)
                    st.pyplot(fig_mpl, clear_figure=True)
                    st.caption("Install `plotly` for interactive legend toggle.")
                st.caption("`recency_rank`: 0 means most recent event; larger means older in the recent-k window.")
                st.caption("`repeat_count`: how many times the same song appears in the selected recent-k events.")
                st.dataframe(phase1_rows, width="stretch")
            except Exception as exc:
                st.info(f"Phase 1 PCA unavailable: {exc}")

    if show_generate_page and (
        generate_section == "Phase 2 - Retrieval + Prompt" or generate_section == GENERATE_SECTION_FULL_PIPELINE
    ):
        if generate_section == GENERATE_SECTION_FULL_PIPELINE:
            st.divider()
            st.markdown("## Full pipeline — Phase 2: Retrieval + prompt")
        else:
            st.markdown("## Phase 2 — Retrieval + Prompt")
        st.caption(f"Current user: `{user_id}` | Embedding variant: `{embedding_variant}`")
        st.subheader("Retrieval snapshot")
        _render_retrieval_snapshot(profile_artifacts)
        st.divider()
        _render_profile_section(profile_artifacts)
        llm_prompt_text = _extract_llm_generation_prompt(profile_artifacts)
        if llm_prompt_text:
            st.divider()
            st.subheader("LLM generation prompt")
            st.text_area("Prompt text", value=llm_prompt_text, height=120, disabled=True)

    if show_generate_page and (
        generate_section == "Phase 3 - Generate + Rerank + Evaluate"
        or generate_section == GENERATE_SECTION_FULL_PIPELINE
    ):
        if generate_section == GENERATE_SECTION_FULL_PIPELINE:
            st.divider()
            st.markdown("## Full pipeline — Phase 3: Generate + rerank + evaluate")
        else:
            st.markdown("## Phase 3 — Generate + Rerank + Evaluate")
        st.caption(f"Current user: `{user_id}` | Embedding variant: `{embedding_variant}`")
        if not profile_artifacts.prompt:
            st.info("Please run Phase 2 first: click `Build or reuse profile` to prepare the prompt.")
        st.subheader("Generation results")
        with st.sidebar:
            st.markdown("### Phase 3: Generate")
            phase3_profile_input_mode = st.radio(
                "Profile input",
                options=["Use profile from Phase 2", "Build/reuse now before generate"],
                index=0,
                help="Choose whether to use currently loaded Phase 2 profile, or rebuild/reuse profile right before generation.",
            )
            generation_model = st.text_input(
                "Generation model",
                value="chirp-v4-5",
                help="Provider-side model name for Suno generation.",
            )
            num_calls = st.number_input(
                "Number of API calls",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="How many generation calls to run for the same prompt.",
            )
            max_concurrency = st.number_input(
                "Max concurrency",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="Maximum parallel generation calls.",
            )
            negative_prompt = st.text_input(
                "Negative prompt",
                value="",
                help="Optional style/content guidance for what to avoid.",
            )
            lyrics = st.text_area(
                "Lyrics or timestamp cues",
                value="",
                height=120,
                help="Optional lyrics or timing cues sent to generation.",
            )
            tempo_hint_bpm = st.number_input(
                "Tempo hint (BPM)",
                min_value=0,
                max_value=300,
                value=0,
                step=1,
                help="Optional tempo hint for generation request.",
            )
            duration_hint_seconds = st.number_input(
                "Duration hint (seconds)",
                min_value=0,
                max_value=600,
                value=0,
                step=1,
                help="Optional duration hint for generation request.",
            )
            rerank_top_k = st.number_input(
                "Rerank keep top-k",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
                help="How many generated candidates to keep after CLAP reranking.",
            )
            diversity_threshold_text = st.text_input(
                "Diversity threshold (optional)",
                value="",
                help="Optional cosine threshold to avoid near-duplicate selected tracks.",
            )
            rerank_encoder = st.selectbox(
                "Rerank encoder",
                ["auto", "finetuned", "zeroshot"],
                index=0,
                help="CLAP encoder used for reranking generated audio.",
            )
            generate_clicked = st.button("Generate songs")

        if generate_clicked:
            if phase3_profile_input_mode == "Build/reuse now before generate":
                with st.spinner("Building/reusing profile artifacts before generation..."):
                    profile_artifacts = build_or_load_profile(
                        user_id=user_id,
                        embedding_variant=embedding_variant,
                        top_k=int(top_k),
                        min_similarity=min_similarity,
                        exclude_recent=exclude_recent,
                        openai_model=openai_model,
                    )
            elif not profile_artifacts.prompt:
                with st.spinner("No prompt artifact found. Building/reusing profile first..."):
                    profile_artifacts = build_or_load_profile(
                        user_id=user_id,
                        embedding_variant=embedding_variant,
                        top_k=int(top_k),
                        min_similarity=min_similarity,
                        exclude_recent=exclude_recent,
                        openai_model=openai_model,
                    )

            try:
                parsed_diversity = float(diversity_threshold_text) if diversity_threshold_text.strip() else None
            except ValueError:
                st.error("Diversity threshold must be empty or a valid float.")
                st.stop()

            with st.spinner("Running generation, rerank, and eval..."):
                generation_result = run_generation_for_user(
                    user_id=user_id,
                    embedding_variant=embedding_variant,
                    prompt_output=profile_artifacts.prompt,
                    generation_model=generation_model,
                    num_calls=int(num_calls),
                    max_concurrency=int(max_concurrency),
                    negative_prompt=negative_prompt.strip() or None,
                    lyrics=lyrics,
                    tempo_hint_bpm=int(tempo_hint_bpm) or None,
                    duration_hint_seconds=int(duration_hint_seconds) or None,
                    rerank_top_k=int(rerank_top_k),
                    rerank_diversity_threshold=parsed_diversity,
                    rerank_encoder=rerank_encoder,
                )
            run_artifacts = generation_result["run_artifacts"]
            st.session_state["latest_generated_run"] = run_artifacts
            st.success(f"Generation, rerank, and eval finished. Latest run: {run_artifacts.run_id}")

        session_run = st.session_state.get("latest_generated_run")
        if run_artifacts is None and isinstance(session_run, GenerationRunArtifacts):
            run_artifacts = session_run
        _render_generation_section(run_artifacts)
        _render_visualization_section(user_id, run_artifacts)


if __name__ == "__main__":
    main()
