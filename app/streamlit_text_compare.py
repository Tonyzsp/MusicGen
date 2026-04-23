"""
Streamlit: text/audio-to-music retrieval — compare base vs fine-tuned CLAP spaces.

Supports two query modes:
- Text description -> retrieve by text-to-audio similarity
- Uploaded audio -> retrieve by audio-to-audio similarity

From repo root:
  python -m streamlit run app/streamlit_query_compare.py
  (legacy entrypoint still works: app/streamlit_text_compare.py)
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.text_compare_service import (
    load_embedding_matrices,
    retrieve_both_from_audio,
    retrieve_both,
)


st.set_page_config(page_title="Gen4Rec · Text/Audio → music retrieval", layout="wide")

# Short label → full prompt text (fills the description box when clicked)
QUERY_TEMPLATES: list[tuple[str, str]] = [
    (
        "Piano",
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
  .tc-preview-title { font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 2px; }
  .tc-muted { color: #64748b; font-size: 0.9rem; }
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

    st.markdown(f'<p class="tc-preview-title">{_h(song)}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="tc-muted">{_h(artist)}</p>', unsafe_allow_html=True)

    st.metric("Text ↔ audio (cosine)", f"{float(hit['cosine_similarity']):.4f}")

    if hit.get("audio_exists"):
        st.audio(hit["audio_path"], format="audio/mp3")
    else:
        st.caption("No local MP3 (expected under music4all/audios/).")

    _render_metadata_block(meta, hit, missing=missing)


def _rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()  # type: ignore[attr-defined]


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


def main() -> None:
    _inject_styles()
    st.title("Text/Audio → music retrieval")
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
    main()
