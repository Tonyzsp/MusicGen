import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "src" / "eval" / "eval_phase_1"
MANIFEST_FIELDS = [
    "song_id",
    "artist",
    "title",
    "query",
    "youtube_title",
    "youtube_url",
    "youtube_id",
    "youtube_duration",
    "raw_path",
    "clip_path",
    "clip_start",
    "status",
    "error",
]


def safe_file_stem(value):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return stem.strip("._") or "song"


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def run_cmd_capture_json(cmd):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def resolve_video(query):
    cmd = [
        "yt-dlp",
        "--dump-single-json",
        "--no-playlist",
        f"ytsearch1:{query}",
    ]
    payload = run_cmd_capture_json(cmd)
    entries = payload.get("entries") or []
    return entries[0] if entries else payload


def video_url_from_info(info):
    if info.get("webpage_url"):
        return info["webpage_url"]
    if info.get("id"):
        return f"https://www.youtube.com/watch?v={info['id']}"
    raise ValueError("yt-dlp did not return a YouTube URL or video id.")


def download_audio(song_id, artist, title, raw_dir, *, query_suffix, force=False):
    file_stem = safe_file_stem(song_id)
    query = " ".join(part for part in [artist, title, query_suffix] if part).strip()
    info = resolve_video(query)
    video_url = video_url_from_info(info)
    output_template = str(raw_dir / f"{file_stem}.%(ext)s")
    raw_path = raw_dir / f"{file_stem}.mp3"

    if raw_path.exists() and not force:
        print(f"Using existing raw audio: {raw_path}")
        return raw_path, info, query

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "-o",
        output_template,
        video_url,
    ]

    print(f"Downloading: {artist} - {title}")
    print(f"Matched YouTube title: {info.get('title', '')}")
    run_cmd(cmd)

    if not raw_path.exists():
        matches = sorted(raw_dir.glob(f"{file_stem}.*"))
        if not matches:
            raise FileNotFoundError(f"yt-dlp finished but no audio file was found for {song_id}")
        raw_path = matches[0]

    return raw_path, info, query

def get_duration(audio_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    return float(result.stdout.strip())

def make_middle_30s_clip(audio_path, output_path, *, force=False):
    if output_path.exists() and not force:
        print(f"Using existing 30s clip: {output_path}")
        duration = get_duration(audio_path)
        return max(0, duration / 2 - 15)

    duration = get_duration(audio_path)

    if duration < 30:
        raise ValueError(f"{audio_path.name} is shorter than 30 seconds")

    start = max(0, duration / 2 - 15)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-ss", str(start),
        "-t", "30",
        "-ac", "1",
        "-ar", "44100",
        str(output_path)
    ]

    print(f"Creating 30s middle clip: {output_path.name}")
    run_cmd(cmd)
    return start


def load_songs(input_path):
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        songs = []
        for idx, row in enumerate(reader, start=1):
            song_id = (row.get("song_id") or row.get("id") or f"song_{idx:03d}").strip()
            artist = (row.get("artist") or row.get("artists") or "").strip()
            title = (row.get("title") or row.get("song") or row.get("name") or "").strip()
            if not title:
                raise ValueError(
                    "Input CSV must contain a title column "
                    "(also supports title aliases: song/name)."
                )
            songs.append(
                {
                    "song_id": str(song_id),
                    "artist": artist,
                    "title": title,
                }
            )
    return songs


def write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download participant history songs from YouTube and create middle 30s clips."
    )
    parser.add_argument("--input", default="songs.csv", help="CSV with song_id/id, artist, and title/song columns.")
    parser.add_argument("--participant-id", default=None, help="Participant id used for default output directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to src/eval/eval_phase_1/<participant-id> "
            "when --participant-id is set, else src/eval/eval_phase_1 (under repo root)."
        ),
    )
    parser.add_argument("--query-suffix", default="official audio", help="Extra words appended to YouTube search.")
    parser.add_argument(
        "--manifest-name",
        default="download_manifest.csv",
        help="Output manifest filename written inside the output directory.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload and recreate clips even if files exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.participant_id:
        output_dir = DEFAULT_OUTPUT_ROOT / safe_file_stem(args.participant_id)
    else:
        output_dir = DEFAULT_OUTPUT_ROOT

    raw_dir = output_dir / "raw"
    clip_dir = output_dir / "clips_30s"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    manifest_rows = []
    for row in load_songs(input_path):
        song_id = row["song_id"]
        artist = row["artist"]
        title = row["title"]
        file_stem = safe_file_stem(song_id)
        manifest_row = {
            "song_id": song_id,
            "artist": artist,
            "title": title,
            "query": "",
            "youtube_title": "",
            "youtube_url": "",
            "youtube_id": "",
            "youtube_duration": "",
            "raw_path": "",
            "clip_path": "",
            "clip_start": "",
            "status": "failed",
            "error": "",
        }

        try:
            audio_path, info, query = download_audio(
                song_id,
                artist,
                title,
                raw_dir,
                query_suffix=args.query_suffix,
                force=args.force,
            )
            clip_path = clip_dir / f"{file_stem}_30s.wav"
            clip_start = make_middle_30s_clip(audio_path, clip_path, force=args.force)

            manifest_row.update(
                {
                    "query": query,
                    "youtube_title": info.get("title", ""),
                    "youtube_url": video_url_from_info(info),
                    "youtube_id": info.get("id", ""),
                    "youtube_duration": info.get("duration", ""),
                    "raw_path": str(audio_path),
                    "clip_path": str(clip_path),
                    "clip_start": f"{clip_start:.3f}",
                    "status": "ok",
                }
            )
        except Exception as e:
            print(f"Failed: {artist} - {title}")
            print(e)
            manifest_row["error"] = str(e)

        manifest_rows.append(manifest_row)
        write_manifest(manifest_path, manifest_rows)

    print(f"Wrote manifest: {manifest_path}")

if __name__ == "__main__":
    main()