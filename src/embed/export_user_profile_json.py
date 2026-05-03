"""
Export Top-K songs nearest to a user embedding in CLAP space as JSON.

Output is info + metadata (+ genres/tags) only, for the User Profile & LLM phase.
No natural-language summary here — teammates consume this JSON downstream.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Avoid FAISS/OpenMP aborts in conda macOS environments with duplicate libomp runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import faiss
except ImportError:
    faiss = None

import numpy as np
import pandas as pd


BASE_DIR_PATH = Path(__file__).resolve().parent
REPO_ROOT_PATH = BASE_DIR_PATH.parent.parent


def resolve_path(env_var: str, default_path: Path) -> str:
    env_value = os.environ.get(env_var)
    if env_value:
        return os.path.abspath(os.path.expanduser(env_value))
    return str(default_path.resolve())


class Config:
    DATASET_PATH = resolve_path("GEN4REC_DATASET_PATH", REPO_ROOT_PATH / "music4all")
    EMBEDDINGS_DIR = resolve_path(
        "GEN4REC_EMBED_OUTPUT_DIR",
        REPO_ROOT_PATH / "outputs" / "embeddings" / "music4all",
    )
    LISTENING_HISTORY_PATH = resolve_path(
        "GEN4REC_LISTENING_HISTORY_PATH",
        Path(DATASET_PATH) / "listening_history.csv",
    )
    SONG_EMB_PATH = resolve_path("GEN4REC_SONG_EMB_PATH", Path(EMBEDDINGS_DIR) / "music4all_embeddings.npy")
    SONG_IDS_PATH = resolve_path("GEN4REC_SONG_IDS_PATH", Path(EMBEDDINGS_DIR) / "music4all_ids.npy")
    SONG_FAISS_INDEX_PATH = resolve_path(
        "GEN4REC_SONG_FAISS_INDEX_PATH",
        REPO_ROOT_PATH / "weights" / "clap" / "music4all_faiss.index",
    )
    USER_EMB_PATH = os.environ.get("GEN4REC_USER_EMB_PATH")
    USER_IDS_PATH = os.environ.get("GEN4REC_USER_IDS_PATH")
    ID_INFORMATION_PATH = resolve_path(
        "GEN4REC_ID_INFORMATION_PATH",
        Path(DATASET_PATH) / "id_information.csv",
    )
    ID_METADATA_PATH = resolve_path(
        "GEN4REC_ID_METADATA_PATH",
        Path(DATASET_PATH) / "id_metadata.csv",
    )
    ID_GENRES_PATH = resolve_path("GEN4REC_ID_GENRES_PATH", Path(DATASET_PATH) / "id_genres.csv")
    ID_TAGS_PATH = resolve_path("GEN4REC_ID_TAGS_PATH", Path(DATASET_PATH) / "id_tags.csv")
    ID_LANG_PATH = resolve_path("GEN4REC_ID_LANG_PATH", Path(DATASET_PATH) / "id_lang.csv")


def ensure_local_file(path: str, description: str) -> str:
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"{description} not found at {path}. "
        "Please download or copy this file and place it at that path, "
        "or override the default location with the corresponding GEN4REC_* environment variable."
    )


def load_listening_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if len(df.columns) == 1 and "\t" in df.columns[0]:
        fixed = pd.read_csv(path, sep="\t", header=None)
        fixed.columns = ["user", "song", "timestamp"][: fixed.shape[1]]
        df = fixed
    lower_map = {c.lower(): c for c in df.columns}
    user_col = lower_map.get("user")
    song_col = lower_map.get("song")
    ts_col = lower_map.get("timestamp")
    if user_col is None or song_col is None:
        raise ValueError("listening_history.csv must contain 'user' and 'song' columns.")
    out = df.rename(columns={user_col: "user_id", song_col: "song_id"}).copy()
    if ts_col is not None:
        out = out.rename(columns={ts_col: "timestamp"})
    out["user_id"] = out["user_id"].astype(str)
    out["song_id"] = out["song_id"].astype(str)
    return out[["user_id", "song_id"]]


def _read_tsv_id_df(path: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns:
        df = pd.read_csv(path, sep="\t", header=None, names=["id", value_col])
    df["id"] = df["id"].astype(str)
    return df


def load_id_information(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns:
        df = pd.read_csv(path, sep="\t", header=None, names=["id", "artist", "song", "album_name"])
    df["id"] = df["id"].astype(str)
    return df


def load_id_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=[
                "id",
                "spotify_id",
                "popularity",
                "release",
                "danceability",
                "energy",
                "key",
                "mode",
                "valence",
                "tempo",
                "duration_ms",
            ],
        )
    df["id"] = df["id"].astype(str)
    return df


def jsonable_value(v: Any) -> Any:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if pd.isna(v):
        return None
    if isinstance(v, str):
        return v
    return str(v)


def build_export_payload(
    user_id: str,
    top_k_requested: int,
    top_k_returned: int,
    retrieval_backend: str,
    min_similarity: float | None,
    threshold_relaxed: bool,
    candidate_count_before_filter: int,
    candidate_count_after_filter: int,
    song_ids: np.ndarray,
    scores: np.ndarray,
    ranks: np.ndarray,
    info_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    genres_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    lang_df: pd.DataFrame,
) -> dict[str, Any]:
    base = pd.DataFrame({"song_id": song_ids, "similarity_score": scores, "rank": ranks})
    info_df = info_df.rename(columns={"id": "song_id"})
    meta_df = meta_df.rename(columns={"id": "song_id"})
    genres_df = genres_df.rename(columns={"id": "song_id"})
    tags_df = tags_df.rename(columns={"id": "song_id"})
    lang_df = lang_df.rename(columns={"id": "song_id"})

    merged = base.merge(info_df, on="song_id", how="left")
    merged = merged.merge(meta_df, on="song_id", how="left", suffixes=("", "_meta_dup"))
    merged = merged.merge(genres_df, on="song_id", how="left")
    merged = merged.merge(tags_df, on="song_id", how="left")
    merged = merged.merge(lang_df, on="song_id", how="left")

    # Drop duplicate id columns if any suffix collision
    drop_cols = [c for c in merged.columns if c.endswith("_meta_dup")]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    songs_out: list[dict[str, Any]] = []
    info_key_order = ("artist", "song", "album_name")
    meta_key_order = (
        "spotify_id",
        "popularity",
        "release",
        "danceability",
        "energy",
        "key",
        "mode",
        "valence",
        "tempo",
        "duration_ms",
    )

    for _, row in merged.iterrows():
        info_block = {
            k: jsonable_value(row[k]) if k in row.index and pd.notna(row.get(k)) else None for k in info_key_order
        }
        meta_block = {k: jsonable_value(row[k]) if k in row.index else None for k in meta_key_order}
        entry: dict[str, Any] = {
            "rank": int(row["rank"]),
            "song_id": str(row["song_id"]),
            "similarity_score": float(row["similarity_score"]),
            "info": info_block,
            "metadata": meta_block,
            "genres": jsonable_value(row["genres"]) if "genres" in row else None,
            "tags": jsonable_value(row["tags"]) if "tags" in row else None,
            "language": jsonable_value(row["lang"]) if "lang" in row else None,
        }
        songs_out.append(entry)

    return {
        "schema_version": "1.3",
        "user_id": user_id,
        "top_k": top_k_returned,
        "retrieval": {
            "backend": retrieval_backend,
            "space": "clap_embedding_cosine",
            "note": "similarity_score is dot product on L2-normalized 512-d vectors (equals cosine similarity).",
            "top_k_requested": top_k_requested,
            "min_similarity": min_similarity,
            "threshold_relaxed": threshold_relaxed,
            "candidate_count_before_filter": candidate_count_before_filter,
            "candidate_count_after_filter": candidate_count_after_filter,
        },
        "songs": songs_out,
    }


def _resolve_listened_indices(user_id: str, song_ids_arr: np.ndarray, exclude_recent: bool) -> set[int]:
    if not exclude_recent:
        return set()

    history = load_listening_history(ensure_local_file(Config.LISTENING_HISTORY_PATH, "Listening history table"))
    listened = set(history.loc[history["user_id"] == user_id, "song_id"].astype(str).tolist())
    if not listened:
        return set()

    song_to_idx = {sid: i for i, sid in enumerate(song_ids_arr)}
    return {song_to_idx[sid] for sid in listened if sid in song_to_idx}


def _rank_with_faiss(
    *,
    user_vec: np.ndarray,
    song_count: int,
    top_k: int,
    min_similarity: float | None,
    listened_idxs: set[int],
) -> tuple[np.ndarray, np.ndarray, int, int, bool] | None:
    if faiss is None or not os.path.exists(Config.SONG_FAISS_INDEX_PATH):
        return None

    index = faiss.read_index(Config.SONG_FAISS_INDEX_PATH)
    if index.ntotal != song_count:
        raise ValueError(
            f"FAISS index size mismatch: index has {index.ntotal} rows but song_ids has {song_count}."
        )
    if index.d != int(user_vec.shape[0]):
        raise ValueError(
            f"FAISS index dim mismatch: index dim={index.d}, user vector dim={user_vec.shape[0]}."
        )

    candidate_count_before_filter = song_count - len(listened_idxs)
    search_k = song_count if min_similarity is not None else min(
        song_count,
        max(top_k + len(listened_idxs), top_k),
    )
    distances, indices = index.search(user_vec.astype(np.float32).reshape(1, -1), search_k)

    rows: list[tuple[int, float]] = []
    for idx, score in zip(indices[0].tolist(), distances[0].tolist()):
        if idx < 0 or idx in listened_idxs:
            continue
        if min_similarity is not None and float(score) < float(min_similarity):
            continue
        rows.append((int(idx), float(score)))

    threshold_relaxed = False
    if min_similarity is not None:
        candidate_count_after_filter = len(rows)
        if candidate_count_after_filter == 0:
            threshold_relaxed = True
            rows = [
                (int(idx), float(score))
                for idx, score in zip(indices[0].tolist(), distances[0].tolist())
                if idx >= 0 and idx not in listened_idxs
            ]
            candidate_count_after_filter = candidate_count_before_filter
    else:
        candidate_count_after_filter = candidate_count_before_filter

    selected = rows[: max(1, top_k)]
    if not selected:
        raise ValueError("No FAISS retrieval candidates were available after filtering.")

    idx = np.array([row[0] for row in selected], dtype=np.int64)
    scores = np.array([row[1] for row in selected], dtype=np.float64)
    return idx, scores, candidate_count_before_filter, candidate_count_after_filter, threshold_relaxed


def _rank_with_numpy(
    *,
    user_vec: np.ndarray,
    song_ids_arr: np.ndarray,
    top_k: int,
    min_similarity: float | None,
    listened_idxs: set[int],
) -> tuple[np.ndarray, np.ndarray, int, int, bool]:
    song_embs = np.load(ensure_local_file(Config.SONG_EMB_PATH, "Song embedding matrix")).astype(np.float32)
    scores_for_rank = song_embs @ user_vec

    if listened_idxs:
        scores_for_rank = scores_for_rank.copy()
        scores_for_rank[np.array(sorted(listened_idxs), dtype=np.int64)] = -1e9

    threshold_relaxed = False
    candidate_count_before_filter = int(np.sum(scores_for_rank > -1e8))
    if min_similarity is not None:
        valid_mask = (scores_for_rank > -1e8) & (scores_for_rank >= float(min_similarity))
        candidate_count_after_filter = int(np.sum(valid_mask))
        if candidate_count_after_filter > 0:
            scores_for_rank = np.where(valid_mask, scores_for_rank, -1e9)
        else:
            threshold_relaxed = True
            candidate_count_after_filter = candidate_count_before_filter
    else:
        candidate_count_after_filter = candidate_count_before_filter

    k_eff = min(top_k, max(1, candidate_count_after_filter))
    idx = np.argpartition(-scores_for_rank, k_eff - 1)[:k_eff]
    idx = idx[np.argsort(-scores_for_rank[idx])]
    scores = scores_for_rank[idx].astype(np.float64)
    return idx, scores, candidate_count_before_filter, candidate_count_after_filter, threshold_relaxed


def export_user_profile_payload(
    *,
    user_id: str,
    top_k: int = 20,
    min_similarity: float | None = None,
    user_emb_path: str | None = None,
    user_ids_path: str | None = None,
    exclude_recent: bool = False,
) -> dict[str, Any]:
    song_ids_arr = np.load(ensure_local_file(Config.SONG_IDS_PATH, "Song ID array"), allow_pickle=True).astype(str)
    resolved_user_emb_path = user_emb_path or Config.USER_EMB_PATH
    resolved_user_ids_path = user_ids_path or Config.USER_IDS_PATH
    if not resolved_user_emb_path or not resolved_user_ids_path:
        raise FileNotFoundError(
            "User embedding files are not configured. Pass both --user-emb-path and --user-ids-path "
            "or set GEN4REC_USER_EMB_PATH / GEN4REC_USER_IDS_PATH."
        )
    user_embs = np.load(ensure_local_file(resolved_user_emb_path, "User embedding matrix")).astype(np.float32)
    user_ids = np.load(ensure_local_file(resolved_user_ids_path, "User ID array"), allow_pickle=True).astype(str)

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    if user_id not in user_to_idx:
        raise ValueError(f"user_id not found: {user_id}")
    user_vec = user_embs[user_to_idx[user_id]]

    k = max(1, top_k)
    listened_idxs = _resolve_listened_indices(user_id, song_ids_arr, exclude_recent)
    rank_result = _rank_with_faiss(
        user_vec=user_vec,
        song_count=len(song_ids_arr),
        top_k=k,
        min_similarity=min_similarity,
        listened_idxs=listened_idxs,
    )
    retrieval_backend = "faiss"
    if rank_result is None:
        retrieval_backend = "numpy"
        rank_result = _rank_with_numpy(
            user_vec=user_vec,
            song_ids_arr=song_ids_arr,
            top_k=k,
            min_similarity=min_similarity,
            listened_idxs=listened_idxs,
        )

    idx, sel_scores, candidate_count_before_filter, candidate_count_after_filter, threshold_relaxed = rank_result
    k_eff = len(idx)
    sel_song_ids = song_ids_arr[idx]
    ranks = np.arange(1, len(idx) + 1)

    info_df = load_id_information(ensure_local_file(Config.ID_INFORMATION_PATH, "Song information table"))
    meta_df = load_id_metadata(ensure_local_file(Config.ID_METADATA_PATH, "Song metadata table"))
    genres_df = _read_tsv_id_df(ensure_local_file(Config.ID_GENRES_PATH, "Song genre table"), "genres")
    tags_df = _read_tsv_id_df(ensure_local_file(Config.ID_TAGS_PATH, "Song tag table"), "tags")
    lang_df = _read_tsv_id_df(ensure_local_file(Config.ID_LANG_PATH, "Song language table"), "lang")

    return build_export_payload(
        user_id=user_id,
        top_k_requested=k,
        top_k_returned=int(k_eff),
        retrieval_backend=retrieval_backend,
        min_similarity=min_similarity,
        threshold_relaxed=threshold_relaxed,
        candidate_count_before_filter=candidate_count_before_filter,
        candidate_count_after_filter=candidate_count_after_filter,
        song_ids=sel_song_ids,
        scores=sel_scores,
        ranks=ranks,
        info_df=info_df,
        meta_df=meta_df,
        genres_df=genres_df,
        tags_df=tags_df,
        lang_df=lang_df,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Top-K nearest songs as JSON (info + metadata + genres/tags) for LLM downstream."
    )
    parser.add_argument("--user-id", required=True)
    parser.add_argument(
        "--top-k",
        "--top-m",
        type=int,
        default=20,
        dest="top_k",
        help="Number of nearest songs to retrieve (Top-K). Alias --top-m is deprecated but still works.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Optional cosine threshold; keep only songs with similarity >= this value.",
    )
    parser.add_argument("--exclude-recent", action="store_true", help="Exclude songs already in user listening history.")
    parser.add_argument("--user-emb-path", type=str, required=True, help="Path to user_embeddings__<variant>.npy")
    parser.add_argument("--user-ids-path", type=str, required=True, help="Path to user_ids__<variant>.npy")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Write JSON to this path; default prints to stdout only.",
    )
    args = parser.parse_args()

    payload = export_user_profile_payload(
        user_id=args.user_id,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        user_emb_path=args.user_emb_path,
        user_ids_path=args.user_ids_path,
        exclude_recent=args.exclude_recent,
    )

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
