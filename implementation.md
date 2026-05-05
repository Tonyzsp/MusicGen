# Gen4Rec Implementation

This document explains the current implementation in `src/` and how artifacts are produced across stages.

---

## 1) Pipeline Overview

1. CLAP finetuning (optional): `src/embed/finetune_clap.py`
2. User embeddings: `src/embed/build_user_embeddings.py`
3. Profile and prompt: `src/profile_prompt/run_profile_pipeline.py`
4. Generation: `src/generate/run_generate.py`
5. Rerank: `src/generate/rerank.py`
6. Eval: `src/eval/run_eval.py`

Default models used in the pipeline:

- Prompt model: `gpt-5.4-mini`
- Generation model: `chirp-v4-5`

---

## 2) Finetuning (CLAP)

Implementation: `src/embed/finetune_clap.py`

### Data

- Metadata sources: `id_genres.csv`, `id_tags.csv`
- Audio source: `music4all/audios/<song_id>.mp3`
- Text per song: `genres: ..., tags: ...`

### Audio preprocessing

- Sample rate: `48k`
- Chunks: `NUM_CHUNKS=3`, `CHUNK_DURATION=10s`
- Short tracks: zero-pad then replicate chunks
- Long tracks: evenly spaced chunk starts

### Model and loss

- Base encoder: LAION CLAP (`HTSAT-base`)
- Chunk pooling: `ContextAttention` (learned weighted pooling over chunks)
- Loss: `SemanticSoftClipLossA2TTextOnly`
  - Audio->Text soft target:
    - `y = (1 - lambda) * I + lambda * softmax(tau * text_sim)`
    - defaults: `lambda = 0.5`, `tau = 10.0`
  - Text->Audio: standard one-hot target

### Training defaults

- Split: `0.8/0.1/0.1`
- Batch size: `32`
- LR: `1e-5`
- Epochs: `16`
- Optimizer: `AdamW(weight_decay=0.01)`
- Best checkpoint: `weights/clap/clap_finetuned_best.pt`

---

## 3) User Embeddings

Implementation: `src/embed/build_user_embeddings.py`

Per user:

1. Take recent `recent_k` events
2. Aggregate by song -> `min_rank`, `play_count`
3. Apply medoid coherence filter
4. Weighted average
5. L2 normalize

Weighting:

- `w_time = exp(-decay_lambda * age)`
- `w_freq = 1 + log(1 + play_count)`
- `w = normalize(w_time * w_freq)`

Medoid coherence filter:

- Build pairwise cosine matrix over candidate songs
- Select medoid = song with highest mean similarity to others
- Keep songs with `sim(song, medoid) >= medoid_threshold`
- If kept songs `< min_keep`, fallback to top-`min_keep` nearest to medoid

---

## 4) Profile and Prompt

Implementations:

- Retrieval export: `src/embed/export_user_profile_json.py`
- Pipeline wrapper: `src/profile_prompt/profile_pipeline.py`
- CLI entry: `src/profile_prompt/run_profile_pipeline.py`

Flow:

1. Top-K retrieval by cosine similarity in embedding space
2. Build raw profile JSON by joining retrieval results with:
   - `id_information.csv` (artist/song/album)
   - `id_metadata.csv` (audio features)
   - `id_genres.csv` (genres)
   - `id_tags.csv` (tags)
3. Build summary (`build_profile_features.py`)
4. Generate profile paragraph + generation prompt with GPT (`gpt-5.4-mini` by default)

Main outputs:

- raw profile JSON
- top-k summary JSON
- prompt JSON
- optional validation JSON

---

## 5) Generation, Rerank, Eval

### Generation (`src/generate/run_generate.py`)

- Input: prompt JSON
- Uses generation model `chirp-v4-5` (default)
- Supports repeated calls (`num_calls`) and bounded parallelism (`max_concurrency`)
- Default artifact layout: `outputs/recSongs/<user>/<run>/` with:
  - `prompt_input.json`
  - `generation_spec.json`
  - `run_manifest.json`
  - `report.md`
  - `audio/`
- Optional `outputs_root` (custom base instead of `outputs/recSongs`); layout stays
  `outputs_root/<user>/<run>/`. Phase 2 script sets `outputs_root` to
  `src/eval/eval_phase_2/<participant>/result` and passes `user_id=phase2_<participant>`,
  so Suno artifacts are under `result/phase2_<participant>/<run_id>/`.

### Rerank (`src/generate/rerank.py`)

- Input: `run_manifest.json` + candidate audio paths
- Embeds generated clips with CLAP (`auto`/`finetuned`/`zeroshot`)
- Scores cosine similarity vs user embedding
- Optional diversity filtering
- Output: `outputs/recSongs/<user>/<run>/rerank_results.json`

### Eval (`src/eval/run_eval.py`)

- Input: manifest + rerank results
- Reuses existing `rerank_results.json` if present
- Computes:
  - personalization
  - diversity
  - imitation risk
- Outputs under `outputs/eval/<user>/<run>/`:
  - `eval_summary.json`
  - `eval_report.md`
  - `reference_alignment.csv`
  - `embedding_space.png` (optional)

Evaluations are split into **phase 1 (retrieval)** vs **phase 2 (custom song list / recommendation)**; see also [`readme.md`](readme.md) (Human evaluation phases).

### Phase 1 User Study (base vs fine-tuned retrieval)

Human-facing check on **text-to-music retrieval**: for fixed text prompts, compare
candidates from **zeroshot (base)** vs **fine-tuned** CLAP embedding matrices,
shuffle clips for blind listening, and write a researcher-only manifest.

- Script: `scripts/run_phase1_eval.py`
- Default output directory: `outputs/phase1_eval/` (`audio/`, `manifest.json`,
  `participant_instructions.txt`, …)
- Override with `--out-dir` if needed.

### Phase 2 User Study (custom song list): WAV clips for recommendation study

This phase collects each participant’s **custom CSV song list** (not Music4All
history), downloads matched audio, converts to local **30-second WAV** clips
under `src/eval/eval_phase_2/`, and feeds the personalized **recommendation /
generation** user-study pipeline.

Participant input template:

- Template: `src/eval/eval_phase_2/manifest_template.csv`
- Per-participant input: `src/eval/eval_phase_2/<participant_id>/manifest.csv`
- Required columns:
  - `song_id`: unique id used for local filenames, e.g. `jerry_001`
  - `artist`: optional but recommended for better YouTube search
  - `title`: song title

Example:

```csv
song_id,artist,title
jerry_001,Mineral,Unfinished
jerry_002,Penfold,I'll Take You Everywhere
```

Download YouTube audio and build middle 30-second clips:

```bash
conda run -n gen4rec python scripts/user_history_download.py \
  --participant-id jerry \
  --input src/eval/eval_phase_2/jerry/manifest.csv
```

The script searches YouTube with `artist + title + official audio`, downloads
audio with `yt-dlp`, converts it with FFmpeg, and clips the middle 30 seconds of
each track.

Outputs:

```text
src/eval/eval_phase_2/<participant_id>/
  manifest.csv              # local participant input; do not commit
  download_manifest.csv     # matched YouTube metadata and processing status
  raw/                        # downloaded mp3 files
  clips_30s/                  # 30-second wav clips
```

- If artist is unknown, leave artist blank in the CSV. The script will search by title.

- Please be aware that template should be placed under human annotator's folder.

- Check `download_manifest.csv` after each run. Confirm that `youtube_title` and
`youtube_url` match the intended songs before using the clips for CLAP embedding
or human evaluation.

- Do not commit participant inputs, downloaded audio, WAV clips, `download_manifest.csv`,
or the entire `result/` directory under each participant. These paths are ignored
by `.gitignore`.


### Phase 2 end-to-end: WAV → Music4All → profile → Suno

`scripts/run_phase2_eval.py` reads `clips_30s/*.wav`, averages **finetuned** CLAP
embeddings into one synthetic user vector (`phase2_<participant>`), runs the
same retrieval + LLM profile/prompt path as the main app (via
`build_or_load_profile_pipeline`), then calls `run_generation_pipeline` with
`outputs_root = src/eval/eval_phase_2/<participant>/result` (same nested layout as
`outputs/recSongs/<user>/<run>/`). Suno artifacts land at:

```text
src/eval/eval_phase_2/<participant>/result/
  wav_embedding_meta.json
  profile_retrieval_raw.json   # copies of pipeline outputs for convenience
  profile_summary.json
  music_prompt.json
  phase2_generation_meta.json
  _phase2_emb/                 # temporary user .npy used for retrieval
  phase2_<participant>/
    <run_id>/                  # Suno run: prompt_input, generation_spec, audio/, run_manifest.json, report.md
```

Example:

```bash
conda run -n gen4rec python scripts/run_phase2_eval.py --participant Tony --top-k 10
```

`run_phase2_eval.py` also supports `--clips-dir`, `--result-dir`, `--encoder`,
`--openai-model`, `--generation-model`, `--num-calls`, `--max-concurrency`,
`--negative-prompt`, and `--rebuild-profile`.

## 6) Naming and Reuse

Gen4Rec uses deterministic parameter-based paths:

- Same params -> same paths -> reuse
- Changed params -> new key/path
- `--force` / `--rebuild` -> recompute under same key

Common keys:

- Embedding key: `rk{recent_k}_dl{decay}_mt{medoid}_mk{min_keep}`
- Profile key: `ev{embedding_variant}_tk{top_k}_ms{sim}_ex{0|1}_om{model}`
