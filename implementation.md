# Gen4Rec Implementation Notes (Code-Aligned)

This document summarizes the current implementation in `src/` and `app/services/`, with emphasis on finetuning, naming, and reuse behavior.

---

## 1. System Architecture

Pipeline stages:

1. CLAP finetuning (optional)
2. User embedding construction
3. Profile and prompt generation
4. Generation and rerank
5. Evaluation

Core entry points:

- `src/embed/finetune_clap.py`
- `src/embed/build_user_embeddings.py`
- `src/profile_prompt/run_profile_pipeline.py`
- `src/generate/run_generate.py`
- `src/generate/rerank.py`
- `src/eval/run_eval.py`

---

## 2. Finetuning Implementation (`src/embed/finetune_clap.py`)

### 2.1 Data and Input Construction

- Uses `music4all/id_genres.csv` and `music4all/id_tags.csv`.
- Keeps only rows with matching local audio file at `music4all/audios/<id>.mp3`.
- Builds one text prompt per song:
  - `genres: <...>, tags: <...>`

### 2.2 Audio Processing

- Target sample rate: `48k` (`SAMPLE_RATE = 48000`).
- Chunk setup:
  - `NUM_CHUNKS = 3`
  - `CHUNK_DURATION = 10s`
  - `CHUNK_SAMPLES = 480000`
- If a track is shorter than one chunk, it is zero-padded and repeated for 3 chunks.
- If longer, 3 evenly spaced chunk start points are used.

### 2.3 Model and Pooling

- Base model: LAION CLAP (`HTSAT-base`) loaded from local checkpoint.
- Chunk features are pooled with a learned attention head:
  - `ContextAttention`
  - input `[B, C, D]`, output `[B, D]`
- This replaces simple mean pooling and lets training learn which chunk matters more.

### 2.4 Loss Design

Implemented loss: `SemanticSoftClipLossA2TTextOnly`.

- Audio->Text uses soft targets built from text-text similarity:
  - `y = (1-lambda)*I + lambda*softmax(tau * text_sim)`
  - default `lambda = 0.5`, `tau = 10.0`
- Text->Audio uses standard one-hot CLIP/CLAP targets.
- This asymmetric setup reduces over-penalization when tag/genre semantics overlap.

### 2.5 Training Setup

- Split: train/val/test = `0.8 / 0.1 / 0.1` (seeded split).
- Batch size: `32`
- Learning rate: `1e-5`
- Epochs: `16`
- Optimizer: `AdamW(weight_decay=0.01)`
- Saves per-epoch checkpoint and best checkpoint:
  - `weights/clap/clap_finetuned_epoch_<N>.pt`
  - `weights/clap/clap_finetuned_best.pt`

---

## 3. User Embedding Implementation

Implementation: `src/embed/build_user_embeddings.py`

For each user:

1. Select latest `recent_k` events.
2. Aggregate duplicate songs into `min_rank` and `play_count`.
3. Apply medoid coherence filter.
4. Weighted average over kept songs.
5. L2-normalize final vector.

Weights:

- `w_time = exp(-decay_lambda * age)`
- `w_freq = 1 + log(1 + play_count)`
- `w = normalize(w_time * w_freq)`

---

## 4. Profile / Prompt Implementation

- Retrieval export: `src/embed/export_user_profile_json.py`
- Profile pipeline: `src/profile_prompt/profile_pipeline.py`
- CLI runner: `src/profile_prompt/run_profile_pipeline.py`

Outputs include:

- raw profile JSON
- top-k summary JSON
- prompt JSON
- optional validation JSON

---

## 5. Generation, Rerank, Eval

### Generation

- `src/generate/run_generate.py`
- Builds generation spec from prompt JSON
- Writes run artifacts under `outputs/recSongs/<user>/<run>/`

### Rerank

- `src/generate/rerank.py`
- CLAP cosine scoring vs user embedding
- Optional diversity filtering
- Output: `rerank_results.json`

### Eval

- `src/eval/run_eval.py`
- Reuses rerank output if already present
- Computes personalization/diversity/risk metrics
- Outputs eval report artifacts under `outputs/eval/<user>/<run>/`

---

## 6. Naming and Reuse

Gen4Rec uses parameter-encoded deterministic paths:

- Same params -> same key/path -> reuse
- Changed params -> new key/path
- `--force` / `--rebuild` -> recompute under same key

Typical keys:

- Embedding key: `rk{recent_k}_dl{decay}_mt{medoid}_mk{min_keep}`
- Profile key: `ev{embedding_variant}_tk{top_k}_ms{sim}_ex{0|1}_om{model}`

