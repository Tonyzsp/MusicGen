"""
Streamlit: text/audio-to-music retrieval — compare base vs fine-tuned CLAP spaces.

Supports two query modes:
- Text description -> retrieve by text-to-audio similarity
- Uploaded audio -> retrieve by audio-to-audio similarity
"""
from __future__ import annotations

import html
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import pandas as pd
import streamlit as st

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.text_compare_service import (
    load_embedding_matrices,
    retrieve_both_from_audio,
    retrieve_both,
)


# Short label → full prompt text (fills the description box when clicked)
QUERY_TEMPLATES: list[tuple[str, str]] = [
    (
        "Soft Piano",
        "slow emotional piano ballad, intimate vocals, sparse arrangement, warm reverb",
    ),
    (
        "Electronic dance",
        "upbeat four-on-the-floor electronic dance, bright synth leads, energetic club mix",
    ),
    (
        "Heavy rock",
        "loud distorted electric guitars, driving drums, aggressive hard rock energy",
    ),
    (
        "Acoustic folk",
        "gentle acoustic guitar and light percussion, folk storytelling vibe, natural room tone",
    ),
    (
        "Lo-fi chill",
        "lo-fi hip hop beats, dusty vinyl texture, mellow relaxed study music",
    ),
    (
        "Orchestral film",
        "cinematic orchestral score, sweeping strings, epic brass, emotional film soundtrack",
    ),
]


def _inject_styles() -> None:
    st.markdown(
        """
<style>
  .tc-preview-title { font-size: 1.35rem; font-weight: 800; line-height: 1.15; margin-bottom: 0.1rem; }
  .tc-muted { color: #94a3b8; font-size: 0.92rem; margin-bottom: 0.5rem; }
  .tc-aa-kicker {
    color: #94a3b8;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
  }
  .tc-aa-title {
    font-size: 1.1rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.2rem;
  }
  .tc-aa-subtitle { color: #94a3b8; font-size: 0.86rem; margin-bottom: 0.45rem; }
  .tc-aa-summary {
    color: inherit;
    font-size: 0.9rem;
    line-height: 1.45;
    margin-top: 0.5rem;
  }
  .tc-chip-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.45rem 0 0.5rem 0; }
  .tc-chip {
    border: 1px solid rgba(148, 163, 184, 0.32);
    border-radius: 999px;
    color: #cbd5e1;
    display: inline-block;
    font-size: 0.72rem;
    line-height: 1;
    padding: 0.34rem 0.55rem;
    background: rgba(148, 163, 184, 0.10);
  }
  .tc-stat-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.35rem; }
  .tc-stat-pill {
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 0.55rem;
    padding: 0.35rem 0.5rem;
    background: rgba(148, 163, 184, 0.07);
  }
  .tc-stat-label { color: #94a3b8; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; }
  .tc-stat-value { font-size: 0.82rem; font-weight: 800; margin-top: 0.08rem; }
  .tc-aa-empty { color: #94a3b8; font-size: 0.86rem; }
  /* Primary buttons → red (selected template, selected track, Retrieve) */
  button[data-testid="baseButton-primary"],
  div[data-testid="stButton"] > button[kind="primary"],
  div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
    background-color: #c62828 !important;
    border: 1px solid #8b1a1a !important;
    color: #ffffff !important;
  }
  button[data-testid="baseButton-primary"]:hover,
  div[data-testid="stButton"] > button[kind="primary"]:hover,
  div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
    background-color: #e53935 !important;
    border-color: #c62828 !important;
    color: #ffffff !important;
  }
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _cached_matrices():
    return load_embedding_matrices()


@st.cache_data(show_spinner=False)
def _cached_aa_song_index() -> dict[str, dict[str, Any]]:
    path = Path(
        os.environ.get(
            "GEN4REC_MUSIC4ALL_AA_INDEX_PATH",
            str(PROJECT_ROOT / "data" / "derived" / "music4all_aa_song_index.parquet"),
        )
    )
    if not path.is_file():
        return {}
    df = pd.read_parquet(path)
    if "song_id" not in df.columns:
        return {}
    df = df.astype(object).where(pd.notna(df), None)
    return {str(row["song_id"]): row for row in df.to_dict("records") if row.get("song_id")}


def _song_artist(hit: dict) -> tuple[str, str]:
    meta = dict(hit.get("metadata") or {})
    meta.pop("_metadata_missing", None)
    song = (meta.get("song") or str(hit.get("song_id", ""))).strip() or str(hit.get("song_id", ""))
    artist = (meta.get("artist") or "—").strip()
    return song, artist


def _meta_field_str(hit: dict, key: str) -> str:
    meta = dict(hit.get("metadata") or {})
    meta.pop("_metadata_missing", None)
    v = meta.get(key)
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return s


def _truncate(s: str, max_len: int) -> str:
    if not s:
        return ""
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _h(s: str) -> str:
    return html.escape(s, quote=True)


def _aa_field(row: dict[str, Any] | None, key: str) -> str:
    if not row:
        return ""
    value = row.get(key)
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _aa_has_any(row: dict[str, Any] | None, keys: tuple[str, ...]) -> bool:
    return any(_aa_field(row, key) for key in keys)


def _aa_pipe_values(row: dict[str, Any] | None, key: str, *, limit: int = 8) -> list[str]:
    raw = _aa_field(row, key)
    if not raw:
        return []
    return [part.strip() for part in raw.split("|") if part.strip()][:limit]


def _compact_text(value: str, max_len: int = 360) -> str:
    value = " ".join(value.split())
    if len(value) <= max_len:
        return value
    return value[: max_len - 1].rstrip() + "…"


def _chip_html(values: list[str]) -> str:
    if not values:
        return ""
    chips = "".join(f'<span class="tc-chip">{_h(value)}</span>' for value in values)
    return f'<div class="tc-chip-row">{chips}</div>'


def _stat_pills(items: list[tuple[str, str]]) -> str:
    visible = [(label, value) for label, value in items if value]
    if not visible:
        return ""
    pills = "".join(
        (
            '<div class="tc-stat-pill">'
            f'<div class="tc-stat-label">{_h(label)}</div>'
            f'<div class="tc-stat-value">{_h(value)}</div>'
            "</div>"
        )
        for label, value in visible
    )
    return f'<div class="tc-stat-row">{pills}</div>'


def _count_label(value: str) -> str:
    if not value:
        return ""
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return value


def _meta_value_present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    return True


def _format_duration_ms(v: Any) -> str:
    if not _meta_value_present(v):
        return "—"
    try:
        ms = float(v)
        sec = max(0, int(round(ms / 1000.0)))
        m, s = divmod(sec, 60)
        if m >= 60:
            h, m = divmod(m, 60)
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:d}:{s:02d}"
    except (TypeError, ValueError):
        return str(v).strip()


def _fmt_feature(key: str, v: Any) -> str:
    if not _meta_value_present(v):
        return "—"
    try:
        if key == "tempo":
            return f"{float(v):.0f} bpm"
        if key in ("danceability", "energy", "valence"):
            return f"{float(v):.3f}"
        if key == "popularity":
            return f"{float(v):.0f}"
        if key == "release":
            return f"{int(float(v))}"
        if key == "duration_ms":
            return _format_duration_ms(v)
        return str(v).strip()
    except (TypeError, ValueError):
        return str(v).strip()


def _render_metadata_block(meta: dict[str, Any], hit: dict, *, missing: str | None) -> None:
    """Catalog + audio-feature layout tuned for full-width preview column."""
    st.divider()
    st.markdown("##### Library metadata")
    if missing:
        st.warning(missing)
        return

    try:
        shell = st.container(border=True)
    except TypeError:
        shell = st.container()
    with shell:
        _render_metadata_block_inner(meta, hit)


def _render_metadata_block_inner(meta: dict[str, Any], hit: dict) -> None:
    left, right = st.columns([1.0, 1.15], gap="medium")
    with left:
        st.markdown("**Song ID**")
        st.code(str(hit.get("song_id", "")), language="text")
        al = meta.get("album_name")
        st.markdown("**Album**")
        st.text(str(al).strip() if _meta_value_present(al) else "—")
    with right:
        g = meta.get("genres")
        t = meta.get("tags")
        st.markdown("**Genres**")
        st.text(str(g).strip() if _meta_value_present(g) else "—")
        st.markdown("**Tags**")
        st.text(str(t).strip() if _meta_value_present(t) else "—")

    feats = ("danceability", "energy", "valence", "tempo", "popularity", "release", "duration_ms")
    if not any(_meta_value_present(meta.get(k)) for k in feats):
        st.caption("No numeric audio features in metadata for this track.")
        return

    st.markdown("**Audio features** (from id_metadata.csv)")
    r1 = st.columns(4)
    for col, key in zip(
        r1,
        ("danceability", "energy", "valence", "tempo"),
    ):
        with col:
            lab = key.replace("_", " ").title()
            st.metric(lab, _fmt_feature(key, meta.get(key)))
    r2 = st.columns(3)
    for col, key in zip(r2, ("popularity", "release", "duration_ms")):
        with col:
            lab = "Length (m:ss)" if key == "duration_ms" else key.title()
            st.metric(lab, _fmt_feature(key, meta.get(key)))


def _render_aa_context(enrichment: dict[str, Any] | None) -> None:
    album_present = _aa_has_any(enrichment, ("album_name", "album_cover_url", "album_summary"))
    artist_present = _aa_has_any(enrichment, ("artist_name", "artist_image_url", "artist_summary"))
    if not album_present and not artist_present:
        return

    st.divider()
    st.markdown("##### Album / Artist context")
    try:
        shell = st.container(border=True)
    except TypeError:
        shell = st.container()

    with shell:
        cover_url = _aa_field(enrichment, "album_cover_url") or _aa_field(enrichment, "artist_image_url")
        left, right = st.columns([0.9, 1.6], gap="medium")
        with left:
            if cover_url:
                st.image(cover_url, use_container_width=True)
            else:
                st.markdown('<p class="tc-aa-empty">No A+A image URL.</p>', unsafe_allow_html=True)
        with right:
            album_name = _aa_field(enrichment, "album_name")
            album_artist = _aa_field(enrichment, "album_artist")
            release_date = _aa_field(enrichment, "album_release_date")
            st.markdown('<div class="tc-aa-kicker">Music4All A+A</div>', unsafe_allow_html=True)
            title = album_name or _aa_field(enrichment, "artist_name") or "Enriched music context"
            subtitle = " · ".join(item for item in (album_artist, release_date) if item)
            st.markdown(f'<div class="tc-aa-title">{_h(title)}</div>', unsafe_allow_html=True)
            if subtitle:
                st.markdown(f'<div class="tc-aa-subtitle">{_h(subtitle)}</div>', unsafe_allow_html=True)

            album_genres = _aa_pipe_values(enrichment, "album_genres", limit=6)
            if album_genres:
                st.markdown(_chip_html(album_genres), unsafe_allow_html=True)

            listeners = _count_label(_aa_field(enrichment, "album_listeners"))
            playcount = _count_label(_aa_field(enrichment, "album_playcount"))
            stats_html = _stat_pills(
                [
                    ("Listeners", listeners),
                    ("Plays", playcount),
                    ("Album MBID", _truncate(_aa_field(enrichment, "album_mbid"), 8)),
                ]
            )
            if stats_html:
                st.markdown(stats_html, unsafe_allow_html=True)

        if artist_present:
            with st.expander("Artist context", expanded=False):
                artist_cols = st.columns([0.68, 1.45], gap="medium")
                with artist_cols[0]:
                    artist_image = _aa_field(enrichment, "artist_image_url")
                    if artist_image:
                        st.image(artist_image, use_container_width=True)
                with artist_cols[1]:
                    artist_name = _aa_field(enrichment, "artist_name")
                    if artist_name:
                        st.markdown(f'<div class="tc-aa-title">{_h(artist_name)}</div>', unsafe_allow_html=True)
                    details = [
                        _aa_field(enrichment, "artist_type"),
                        _aa_field(enrichment, "artist_country"),
                    ]
                    details = [item for item in details if item]
                    if details:
                        st.markdown(
                            f'<div class="tc-aa-subtitle">{_h(" · ".join(details))}</div>',
                            unsafe_allow_html=True,
                        )
                    artist_genres = _aa_pipe_values(enrichment, "artist_genres", limit=8)
                    if artist_genres:
                        st.markdown(_chip_html(artist_genres), unsafe_allow_html=True)


def _render_query_templates() -> None:
    st.markdown("**Quick description templates** (optional)")
    ncols = 3
    for row_start in range(0, len(QUERY_TEMPLATES), ncols):
        chunk = QUERY_TEMPLATES[row_start : row_start + ncols]
        cols = st.columns(len(chunk))
        for col, j, (label, text) in zip(cols, range(len(chunk)), chunk):
            idx = row_start + j
            tpl_sel = st.session_state.get("tc_tpl_selected")
            is_on = tpl_sel == idx
            with col:
                if st.button(
                    label,
                    key=f"tc_tpl_{idx}",
                    type="primary" if is_on else "secondary",
                    use_container_width=True,
                ):
                    if tpl_sel == idx:
                        st.session_state["tc_tpl_selected"] = None
                        st.session_state["tc_query_text"] = ""
                    else:
                        st.session_state["tc_query_text"] = text
                        st.session_state["tc_tpl_selected"] = idx
                    _rerun_app()


def _render_preview(hit: dict) -> None:
    meta = dict(hit.get("metadata") or {})
    missing = meta.pop("_metadata_missing", None)
    song, artist = _song_artist(hit)
    enrichment = _cached_aa_song_index().get(str(hit.get("song_id", "")))

    st.markdown(f'<p class="tc-preview-title">{_h(song)}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="tc-muted">{_h(artist)}</p>', unsafe_allow_html=True)

    st.metric("Text ↔ audio (cosine)", f"{float(hit['cosine_similarity']):.4f}")

    if hit.get("audio_exists"):
        st.audio(hit["audio_path"], format="audio/mp3")
    else:
        st.caption("No local MP3 (expected under music4all/audios/).")

    _render_aa_context(enrichment)
    _render_metadata_block(meta, hit, missing=missing)


def _rerun_app() -> None:
    st.rerun()


def _render_track_choices(hits: list[dict], *, side: str, run_id: int) -> int:
    """Clickable full-width rows; returns selected index."""
    track_key = f"tc_trk_{side}_{run_id}"
    if track_key not in st.session_state:
        st.session_state[track_key] = 0
    try:
        sel = int(st.session_state[track_key])
    except (TypeError, ValueError):
        sel = 0
    if sel < 0 or sel >= len(hits):
        sel = 0
        st.session_state[track_key] = sel

    st.markdown("**Results** — click a row to preview on the right")
    for i, hit in enumerate(hits):
        # Re-read each row: otherwise `type=` uses a stale index and the old row
        # stays "primary" until the next full run (Streamlit evaluates buttons in order).
        sel = int(st.session_state[track_key])

        song, artist = _song_artist(hit)
        cos = float(hit["cosine_similarity"])
        g = _truncate(_meta_field_str(hit, "genres"), 80)
        line = f"{i + 1}. {_truncate(song + ' — ' + artist, 78)}"
        if st.button(
            line,
            key=f"tc_trkbtn_{side}_{run_id}_{i}",
            type="primary" if i == sel else "secondary",
            use_container_width=True,
        ):
            st.session_state[track_key] = i
            _rerun_app()

        st.caption(f"cos {cos:.3f} · {g or '—'}")

    return int(st.session_state[track_key])


def _render_model_section(title: str, hits: list[dict], *, side: str, run_id: int) -> None:
    st.markdown(f"### {title}")
    list_col, preview_col = st.columns([1.15, 1.0], gap="large")

    with list_col:
        if not hits:
            st.caption("No tracks to show.")
            selected = 0
        else:
            selected = _render_track_choices(hits, side=side, run_id=run_id)

    with preview_col:
        try:
            shell = st.container(border=True)
        except TypeError:
            shell = st.container()
        with shell:
            st.markdown("##### Preview")
            if not hits:
                st.caption("No results.")
            else:
                _render_preview(hits[selected])


def render_query_compare_page() -> None:
    _inject_styles()
    st.title("Embedding Retrieval: Base vs Finetuned")
    st.caption(
        "Use a **text description** or **uploaded audio** query; we **retrieve songs** whose **audio embeddings** "
        "are closest in CLAP space (not keyword search in lyrics/metadata)."
    )

    try:
        matrices = _cached_matrices()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Ensure `music4all_embeddings_zeroshot.npy` and `music4all_embeddings.npy` exist "
            "and set `GEN4REC_*` paths if your files are not in the default locations."
        )
        return
    except ValueError as e:
        st.error(str(e))
        return

    n_songs = int(len(matrices.ids))
    max_k = min(200, max(1, n_songs))
    default_k = min(5, max_k)

    if "tc_query_text" not in st.session_state:
        st.session_state["tc_query_text"] = ""

    with st.sidebar:
        st.subheader("Query")
        query_mode = st.radio(
            "Query mode",
            options=["Text description", "Upload audio"],
            index=0,
            help="Compare base vs fine-tuned retrieval using text query or uploaded audio query.",
        )
        top_k = st.number_input(
            "How many songs to retrieve (per model)",
            min_value=1,
            max_value=max_k,
            value=default_k,
            step=1,
            help="Top-K tracks by similarity between your text embedding and each song's audio embedding.",
        )
        with st.expander("Embedding files", expanded=False):
            st.code(matrices.zeroshot_path, language="text")
            st.code(matrices.finetuned_path, language="text")

    uploaded_audio = None
    if query_mode == "Text description":
        _render_query_templates()
        st.text_area(
            "Describe the music you want (text → song retrieval)",
            height=110,
            placeholder="e.g. melancholic piano ballad, upbeat electronic dance, heavy distorted guitars…",
            key="tc_query_text",
        )
    else:
        uploaded_audio = st.file_uploader(
            "Upload query audio (audio → song retrieval)",
            type=["mp3", "wav", "flac", "ogg", "m4a"],
            help="Upload one audio clip as the query. We will compare retrieval in base vs fine-tuned embedding spaces.",
        )
        if uploaded_audio is not None:
            st.audio(uploaded_audio.getvalue(), format=uploaded_audio.type or "audio/mpeg")

    btn_col, _btn_spacer = st.columns([1, 5], gap="small")
    with btn_col:
        run = st.button(
            "Retrieve",
            type="primary",
            use_container_width=True,
            help="Run retrieval (base vs fine-tuned).",
        )

    base_hits: list[dict] = []
    ft_hits: list[dict] = []

    if run:
        if query_mode == "Text description":
            query = (st.session_state.get("tc_query_text") or "").strip()
            if not query:
                st.warning("Please enter a text description of the music you want.")
                return
            had_results = "tc_base_hits" in st.session_state
            spin_msg = (
                "Reloading: encoding your query and retrieving songs…"
                if had_results
                else "First retrieval: loading models and encoding your query (may take a while)…"
            )
            with st.spinner(spin_msg):
                try:
                    base_hits, ft_hits = retrieve_both(query, matrices, int(top_k))
                except FileNotFoundError as e:
                    st.error(str(e))
                    return
                except Exception as e:
                    st.exception(e)
                    return
        else:
            if uploaded_audio is None:
                st.warning("Please upload an audio file first.")
                return
            suffix = Path(uploaded_audio.name or "query.wav").suffix or ".wav"
            tmp_path = ""
            had_results = "tc_base_hits" in st.session_state
            spin_msg = (
                "Reloading: encoding uploaded audio and retrieving songs…"
                if had_results
                else "First retrieval: loading models and encoding uploaded audio (may take a while)…"
            )
            with st.spinner(spin_msg):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                        f.write(uploaded_audio.getvalue())
                        tmp_path = f.name
                    base_hits, ft_hits = retrieve_both_from_audio(tmp_path, matrices, int(top_k))
                except FileNotFoundError as e:
                    st.error(str(e))
                    return
                except Exception as e:
                    st.exception(e)
                    return
                finally:
                    if tmp_path and os.path.isfile(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
        st.session_state["tc_base_hits"] = base_hits
        st.session_state["tc_ft_hits"] = ft_hits
        new_rid = int(st.session_state.get("tc_run_id", 0)) + 1
        st.session_state["tc_run_id"] = new_rid
        st.session_state[f"tc_trk_base_{new_rid}"] = 0
        st.session_state[f"tc_trk_ft_{new_rid}"] = 0

    if "tc_base_hits" not in st.session_state:
        if query_mode == "Upload audio":
            st.info("Upload an audio clip, then click **Retrieve**.")
        else:
            st.info("Pick a template or type your own description, then click **Retrieve**.")
        return

    base_hits: list[dict] = st.session_state["tc_base_hits"]
    ft_hits: list[dict] = st.session_state["tc_ft_hits"]
    run_id = int(st.session_state.get("tc_run_id", 0))

    st.divider()
    _render_model_section("Base (pretrained)", base_hits, side="base", run_id=run_id)
    st.divider()
    _render_model_section("Finetune (fine-tuned)", ft_hits, side="ft", run_id=run_id)


if __name__ == "__main__":
    st.set_page_config(page_title="Gen4Rec · Embedding Retrieval Compare", layout="wide")
    render_query_compare_page()
