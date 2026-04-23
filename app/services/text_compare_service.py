"""
Text-to-music (cross-modal) retrieval: given a text query, rank songs by CLAP similarity.

Each song is represented by an *audio* embedding; the query uses the CLAP *text* branch.
Compares zero-shot (base) vs fine-tuned song embedding spaces; text encoding matches
`src/embed/finetune_clap.py` (RobertaTokenizer + text_branch).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer

from src.embed.embed_music4all import load_finetuned_model_and_attention
from src.embed.embed_music4all_zeroshot import load_zeroshot_clap
from src.embed.recommend_topk import Config as RecConfig
from src.embed.recommend_topk import load_audio_metadata, load_song_metadata


BASE_DIR_PATH = Path(__file__).resolve().parent
REPO_ROOT_PATH = BASE_DIR_PATH.parent.parent


def resolve_zeroshot_emb_path() -> str:
    env = os.environ.get("GEN4REC_SONG_EMB_ZEROSHOT_PATH")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return str(Path(RecConfig.EMBEDDINGS_DIR) / "music4all_embeddings_zeroshot.npy")


def resolve_finetuned_emb_path() -> str:
    return RecConfig.SONG_EMB_PATH


def resolve_song_ids_path() -> str:
    return RecConfig.SONG_IDS_PATH


def resolve_audio_path(song_id: str) -> str:
    return str(Path(RecConfig.DATASET_PATH) / "audios" / f"{song_id}.mp3")


@lru_cache(maxsize=1)
def _tokenizer() -> RobertaTokenizer:
    return RobertaTokenizer.from_pretrained("roberta-base")


def encode_text_for_model(model: torch.nn.Module, device: str, query: str) -> np.ndarray:
    tok = _tokenizer()
    batch = tok([query], padding=True, truncation=True, max_length=77, return_tensors="pt")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    model.eval()
    with torch.no_grad():
        text_out = model.text_branch(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_out[1]
        if hasattr(model, "text_projection"):
            text_features = model.text_projection(text_features)
        text_features = F.normalize(text_features, dim=-1)
    return text_features.squeeze(0).cpu().numpy().astype(np.float32)


def cosine_topk(text_vec: np.ndarray, song_matrix: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Assume L2-normalized rows; returns (indices, cosine_scores)."""
    scores = song_matrix @ text_vec
    k = min(top_k, len(scores))
    if k <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    part = np.argpartition(-scores, kth=k - 1)[:k]
    local = part[np.argsort(-scores[part])]
    return local, scores[local]


def _load_id_genres_tags() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load id_genres.csv and id_tags.tsv-style TSVs from the dataset root (optional)."""
    root = Path(RecConfig.DATASET_PATH)
    gpath = root / "id_genres.csv"
    tpath = root / "id_tags.csv"

    def _read(path: Path, value_name: str) -> pd.DataFrame | None:
        if not path.is_file():
            return None
        df = pd.read_csv(path, sep="\t")
        if "id" not in df.columns:
            df = pd.read_csv(path, sep="\t", header=None, names=["id", value_name])
        df["id"] = df["id"].astype(str)
        if value_name not in df.columns:
            return None
        return df[["id", value_name]].copy()

    return _read(gpath, "genres"), _read(tpath, "tags")


@lru_cache(maxsize=1)
def _song_info_merged() -> pd.DataFrame:
    info = load_song_metadata(RecConfig.ID_INFORMATION_PATH)
    meta = load_audio_metadata(RecConfig.ID_METADATA_PATH)
    merged = info.merge(meta, on="id", how="left", suffixes=("", "_meta"))
    genres_df, tags_df = _load_id_genres_tags()
    if genres_df is not None:
        merged = merged.merge(genres_df, on="id", how="left")
    else:
        merged["genres"] = None
    if tags_df is not None:
        merged = merged.merge(tags_df, on="id", how="left")
    else:
        merged["tags"] = None
    return merged


def row_for_song(song_id: str) -> dict[str, Any] | None:
    df = _song_info_merged()
    hit = df[df["id"].astype(str) == str(song_id)]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def format_metadata_row(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"_metadata_missing": "No id_information / id_metadata row for this song_id."}
    keys = [
        "artist",
        "song",
        "album_name",
        "genres",
        "tags",
        "popularity",
        "release",
        "danceability",
        "energy",
        "valence",
        "tempo",
        "duration_ms",
    ]
    return {k: row.get(k) for k in keys}


class EmbeddingMatrices:
    __slots__ = ("ids", "base", "finetuned", "zeroshot_path", "finetuned_path")

    def __init__(
        self,
        ids: np.ndarray,
        base: np.ndarray,
        finetuned: np.ndarray,
        *,
        zeroshot_path: str,
        finetuned_path: str,
    ) -> None:
        self.ids = ids
        self.base = base
        self.finetuned = finetuned
        self.zeroshot_path = zeroshot_path
        self.finetuned_path = finetuned_path


@lru_cache(maxsize=1)
def load_embedding_matrices() -> EmbeddingMatrices:
    zpath = resolve_zeroshot_emb_path()
    fpath = resolve_finetuned_emb_path()
    idpath = resolve_song_ids_path()
    for p, label in [(idpath, "song ids"), (zpath, "zero-shot embeddings"), (fpath, "fine-tuned embeddings")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{label} not found: {p}")
    ids = np.load(idpath, allow_pickle=True).astype(str)
    base = np.load(zpath, mmap_mode="r")
    finetuned = np.load(fpath, mmap_mode="r")
    if base.shape[0] != finetuned.shape[0] or len(ids) != base.shape[0]:
        raise ValueError(
            f"Embedding row count mismatch: ids={len(ids)}, base={base.shape[0]}, finetuned={finetuned.shape[0]}"
        )
    if base.shape[1] != finetuned.shape[1]:
        raise ValueError(f"Embedding dim mismatch: base={base.shape[1]}, finetuned={finetuned.shape[1]}")
    return EmbeddingMatrices(ids, base, finetuned, zeroshot_path=zpath, finetuned_path=fpath)


@lru_cache(maxsize=2)
def _zeroshot_model_device() -> tuple[torch.nn.Module, str]:
    model = load_zeroshot_clap(_device_string())
    return model, str(next(model.parameters()).device)


@lru_cache(maxsize=2)
def _finetuned_model_device() -> tuple[torch.nn.Module, str]:
    model, _, _ = load_finetuned_model_and_attention(_device_string())
    return model, str(next(model.parameters()).device)


def _device_string() -> str:
    return os.environ.get("GEN4REC_CLAP_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def retrieve_both(
    query: str,
    matrices: EmbeddingMatrices,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    z_model, z_dev = _zeroshot_model_device()
    f_model, f_dev = _finetuned_model_device()

    z_text = encode_text_for_model(z_model, z_dev, query)
    f_text = encode_text_for_model(f_model, f_dev, query)

    z_idx, z_scores = cosine_topk(z_text, matrices.base, top_k)
    f_idx, f_scores = cosine_topk(f_text, matrices.finetuned, top_k)

    def pack(indices: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, s in zip(indices.tolist(), scores.tolist()):
            sid = str(matrices.ids[i])
            ap = resolve_audio_path(sid)
            out.append(
                {
                    "song_id": sid,
                    "cosine_similarity": float(s),
                    "audio_path": ap,
                    "audio_exists": os.path.isfile(ap),
                    "metadata": format_metadata_row(row_for_song(sid)),
                }
            )
        return out

    return pack(z_idx, z_scores), pack(f_idx, f_scores)
