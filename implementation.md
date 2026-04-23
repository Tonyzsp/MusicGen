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
- Writes run artifacts under `outputs/recSongs/<user>/<run>/`:
  - `prompt_input.json`
  - `generation_spec.json`
  - `run_manifest.json`
  - `report.md`
  - `audio/`

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

---

## 6) Naming and Reuse

Gen4Rec uses deterministic parameter-based paths:

- Same params -> same paths -> reuse
- Changed params -> new key/path
- `--force` / `--rebuild` -> recompute under same key

Common keys:

- Embedding key: `rk{recent_k}_dl{decay}_mt{medoid}_mk{min_keep}`
- Profile key: `ev{embedding_variant}_tk{top_k}_ms{sim}_ex{0|1}_om{model}`
