from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_AA_ROOT = REPO_ROOT / "music4allA+A"
DEFAULT_MUSIC4ALL_INFO = REPO_ROOT / "music4all" / "id_information.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "derived" / "music4all_aa_song_index.parquet"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _join_pipe(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return "|".join(_text(v) for v in values if _text(v))


def _tag_names_and_urls(tags: Any) -> tuple[str, str]:
    raw_tags = (tags or {}).get("tag") if isinstance(tags, dict) else tags
    if isinstance(raw_tags, dict):
        raw_tags = [raw_tags]
    if not isinstance(raw_tags, list):
        return "", ""

    names: list[str] = []
    urls: list[str] = []
    for item in raw_tags:
        if isinstance(item, str):
            name = _text(item)
            url = ""
        elif isinstance(item, dict):
            name = _text(item.get("name"))
            url = _text(item.get("url"))
        else:
            continue
        if not name:
            continue
        names.append(name)
        urls.append(url)
    return "|".join(names), "|".join(urls)


def _relation_url(relations: Any, relation_type: str) -> str:
    if isinstance(relations, dict):
        relations = [relations]
    if not isinstance(relations, list):
        return ""
    for item in relations:
        if not isinstance(item, dict):
            continue
        if _text(item.get("type")).casefold() == relation_type.casefold():
            return _text(item.get("target"))
    return ""


def _pick_image_url(urls: Any) -> str:
    if isinstance(urls, str):
        return urls.strip()
    if not isinstance(urls, list):
        return ""
    candidates: list[str] = []
    for item in urls:
        if isinstance(item, str):
            url = item.strip()
        elif isinstance(item, dict):
            url = _text(item.get("#text"))
        else:
            url = ""
        if url:
            candidates.append(url)
    if not candidates:
        return ""
    # Prefer the larger Last.fm sizes when available.
    for needle in ("300x300", "174s", "64s"):
        for url in candidates:
            if needle in url:
                return url
    return candidates[0]


def _music4all_info(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "id" not in df.columns:
        df = pd.read_csv(path, sep="\t", header=None, names=["id", "artist", "song", "album_name"], dtype=str)
    for col in ("id", "artist", "song", "album_name"):
        if col not in df.columns:
            df[col] = ""
    return df[["id", "artist", "song", "album_name"]].fillna("")


def _same_name(left: str, right: str) -> bool:
    return left.strip().casefold() == right.strip().casefold() and bool(left.strip())


def _album_score(record: dict[str, str], info: dict[str, dict[str, str]]) -> int:
    row = info.get(record["song_id"], {})
    score = 0
    if _same_name(record.get("album_name", ""), row.get("album_name", "")):
        score += 2
    if _same_name(record.get("album_artist", ""), row.get("artist", "")):
        score += 1
    return score


def _artist_score(record: dict[str, str], info: dict[str, dict[str, str]]) -> int:
    row = info.get(record["song_id"], {})
    return 1 if _same_name(record.get("artist_name", ""), row.get("artist", "")) else 0


def _upsert_best(
    target: dict[str, dict[str, str]],
    record: dict[str, str],
    info: dict[str, dict[str, str]],
    score_fn,
) -> bool:
    song_id = record["song_id"]
    current = target.get(song_id)
    if current is None:
        target[song_id] = record
        return False
    if score_fn(record, info) > score_fn(current, info):
        target[song_id] = record
        return True
    return True


def _build_album_records(album_dir: Path, known_ids: set[str], info: dict[str, dict[str, str]]) -> tuple[dict[str, dict[str, str]], int]:
    by_song: dict[str, dict[str, str]] = {}
    conflicts = 0
    for path in sorted(album_dir.glob("*.json")):
        data = _read_json(path)
        album = ((data.get("album_info") or {}).get("album") or {})
        song_ids = [str(sid) for sid in (data.get("music4all_onion_id") or []) if str(sid) in known_ids]
        if not song_ids:
            continue
        album_tags, album_tag_urls = _tag_names_and_urls(album.get("tags"))
        record_base = {
            "album_mbid": _text(data.get("mbid") or album.get("mbid") or path.stem),
            "album_name": _text(album.get("name")),
            "album_artist": _text(album.get("artist")),
            "album_cover_url": _pick_image_url(data.get("album_image_url") or album.get("image")),
            "album_release_date": _text(data.get("release_date")),
            "album_listeners": _text(album.get("listeners")),
            "album_playcount": _text(album.get("playcount")),
            "album_genres": _join_pipe(album.get("genres")),
            "album_tags": album_tags,
            "album_tag_urls": album_tag_urls,
            "album_summary": _text((album.get("wiki") or {}).get("summary")),
            "album_lastfm_url": _text(album.get("url")),
        }
        for song_id in song_ids:
            conflicts += int(
                _upsert_best(by_song, {"song_id": song_id, **record_base}, info, _album_score)
            )
    return by_song, conflicts


def _build_artist_records(artist_dir: Path, known_ids: set[str], info: dict[str, dict[str, str]]) -> tuple[dict[str, dict[str, str]], int]:
    by_song: dict[str, dict[str, str]] = {}
    conflicts = 0
    for path in sorted(artist_dir.glob("*.json")):
        data = _read_json(path)
        artist = ((data.get("artist_info") or {}).get("artist") or {})
        song_ids = [str(sid) for sid in (data.get("music4all_onion_id") or []) if str(sid) in known_ids]
        if not song_ids:
            continue
        area = artist.get("area") or {}
        life_span = artist.get("life-span") or {}
        record_base = {
            "artist_mbid": _text(data.get("mbid") or artist.get("id") or path.stem),
            "artist_name": _text(artist.get("name")),
            "artist_image_url": _pick_image_url(data.get("artist_image_url")),
            "artist_country": _text(artist.get("country") or area.get("name")),
            "artist_type": _text(artist.get("type")),
            "artist_gender": _text(artist.get("gender")),
            "artist_life_span_begin": _text(life_span.get("begin")),
            "artist_life_span_end": _text(life_span.get("end")),
            "artist_genres": _join_pipe(artist.get("genres")),
            "artist_summary": _text((artist.get("wiki") or {}).get("summary")),
            "artist_lastfm_url": _relation_url(artist.get("url-relation-list"), "last.fm"),
        }
        for song_id in song_ids:
            conflicts += int(
                _upsert_best(by_song, {"song_id": song_id, **record_base}, info, _artist_score)
            )
    return by_song, conflicts


def build_index(aa_root: Path, music4all_info_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    info_df = _music4all_info(music4all_info_path)
    known_ids = set(info_df["id"].astype(str))
    info = info_df.set_index("id").to_dict("index")

    album_records, album_conflicts = _build_album_records(aa_root / "album_json", known_ids, info)
    artist_records, artist_conflicts = _build_artist_records(aa_root / "artists_json", known_ids, info)

    rows: list[dict[str, str]] = []
    for song_id in sorted(set(album_records) | set(artist_records)):
        base = info.get(song_id, {})
        row = {
            "song_id": song_id,
            "music4all_artist": _text(base.get("artist")),
            "music4all_song": _text(base.get("song")),
            "music4all_album_name": _text(base.get("album_name")),
        }
        row.update(album_records.get(song_id, {}))
        row.update(artist_records.get(song_id, {}))
        rows.append(row)

    df = pd.DataFrame(rows).fillna("")
    stats = {
        "music4all_ids": len(known_ids),
        "indexed_song_ids": len(df),
        "album_song_ids": len(album_records),
        "artist_song_ids": len(artist_records),
        "album_conflicts": album_conflicts,
        "artist_conflicts": artist_conflicts,
    }
    return df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a song_id-level Music4All A+A enrichment index.")
    parser.add_argument("--aa-root", type=Path, default=DEFAULT_AA_ROOT)
    parser.add_argument("--music4all-info", type=Path, default=DEFAULT_MUSIC4ALL_INFO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df, stats = build_index(args.aa_root, args.music4all_info)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"Wrote {len(df):,} rows to {args.out}")
    for key, value in stats.items():
        print(f"{key}: {value:,}")
    if stats["music4all_ids"]:
        print(f"coverage: {stats['indexed_song_ids'] / stats['music4all_ids']:.2%}")


if __name__ == "__main__":
    main()
