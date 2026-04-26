# Gen4Rec

## Project Introduction

Gen4Rec is an end-to-end generative music recommendation system. Instead of only retrieving existing tracks, it learns user preference embeddings from listening history, creates profile-driven music prompts, generates new candidate songs, reranks candidates in CLAP embedding space, and evaluates the final outputs with personalization, diversity, and imitation-risk metrics.

## Pipeline Overview

The project runs in six main stages:

1. **CLAP Finetuning** *(optional)*: improves audio-text alignment quality.
2. **User Embedding Construction**: builds user representations from listening history.
3. **Profile and Prompt Generation**: creates personalized profile summaries and generation prompts.
4. **Music Generation**: generates candidate tracks from prompts.
5. **Rerank**: reorders candidates using similarity signals in embedding space.
6. **Evaluation**: measures personalization, diversity, and imitation risk.

Implementation details are documented in [`implementation.md`](implementation.md).

---

## Teammates

1. Tony Zhao (`sz3822`) - [GitHub](https://github.com/Tonyzsp)
2. Conny Fan (`jf4644`) - [GitHub](https://github.com/ConnyFan123)
3. Jerry Huang (`jh8186`) - [GitHub](https://github.com/J-hjr)

---

## Project Structure (Current)

```text
Gen4Rec/
├── app/
│   ├── services/
│   │   ├── artifact_service.py
│   │   ├── pipeline_service.py
│   │   └── viz_service.py
│   ├── streamlit_app.py
│   ├── streamlit_query_compare.py
│   └── streamlit_text_compare.py
├── scripts/
│   └── run_full_pipeline.py
├── src/
│   ├── data/
│   ├── embed/
│   ├── profile_prompt/
│   ├── generate/
│   └── eval/
├── music4all/                         # local dataset folder (NOT committed)
│   ├── listening_history.csv          # must download manually (large)
│   ├── id_information.csv             # must download manually (large)
│   ├── id_genres.csv                  # must download manually (large)
│   ├── id_tags.csv                    # must download manually (large)
│   ├── id_metadata.csv                # must download manually (large)
│   └── audios/                        # must download manually (very large)
├── weights/
│   └── clap/                          # local checkpoints (NOT committed)
│       ├── music_audioset_epoch_15_esc_90.14.pt   # download manually
│       └── clap_finetuned_best.pt                 # download manually
├── outputs/                           # generated artifacts (NOT committed)
├── notebooks/
├── environment.yaml
├── environment-windows.yaml
└── .gitignore
```

Note: dataset CSV/audio files and model checkpoints are intentionally not stored in git due size. Each teammate needs to download them locally.

---

## CLI (Phase-by-Phase)

### 1) Set Up Environment

Create the environment (first-time setup):

```bash
conda env create -f environment.yaml
```

If the environment already exists, update it:

```bash
conda env update -f environment.yaml --prune
```

Then activate:

```bash
conda activate gen4rec
```

Required API keys (put them in `.env`):

```env
OPENAI_API_KEY=...
ACE_SUNO_API_KEY=...
```

Demo values used below:

- `user_id`: `user_007XIjOr`
- embedding variant: `rk10_dl0.08_mt0.2_mk5`
- profile key: `evrk10_dl0.08_mt0.2_mk5_tk20_msnone_ex1_omgpt-5.4-mini`
- example run id: `20260423T120000Z-user_007XIjOr-suno` (replace with your actual Stage C output if different)

### 2) Stage A - User Embeddings

```bash
python src/embed/build_user_embeddings.py \
  --recent-k 10 \
  --decay-lambda 0.08 \
  --medoid-threshold 0.2 \
  --min-keep 5
```

Force rebuild:

```bash
python src/embed/build_user_embeddings.py --recent-k 10 --decay-lambda 0.08 --medoid-threshold 0.2 --min-keep 5 --force
```

### 3) Stage B - Retrieval Export (Raw Profile JSON)

```bash
python src/embed/export_user_profile_json.py \
  --user-id user_007XIjOr \
  --top-k 20 \
  --exclude-recent \
  --user-emb-path outputs/embeddings/music4all/user_embeddings__rk10_dl0.08_mt0.2_mk5.npy \
  --user-ids-path outputs/embeddings/music4all/user_ids__rk10_dl0.08_mt0.2_mk5.npy \
  -o outputs/profiles/user_007XIjOr.json
```

### 4) Stage B - Unified Profile Pipeline

```bash
python src/profile_prompt/run_profile_pipeline.py \
  --user-id user_007XIjOr \
  --embedding-variant rk10_dl0.08_mt0.2_mk5 \
  --top-k 20 \
  --exclude-recent \
  --openai-model gpt-5.4-mini
```

Rebuild:

```bash
python src/profile_prompt/run_profile_pipeline.py --user-id user_007XIjOr --embedding-variant rk10_dl0.08_mt0.2_mk5 --top-k 20 --exclude-recent --openai-model gpt-5.4-mini --rebuild
```

### 5) Stage C - Generation

```bash
python src/generate/run_generate.py \
  --prompt-json outputs/profiles/user_007XIjOr__evrk10_dl0.08_mt0.2_mk5_tk20_msnone_ex1_omgpt-5.4-mini_prompt.json \
  --generation-model chirp-v4-5 \
  --num-calls 1 \
  --max-concurrency 1
```

### 6) Stage D1 - Rerank

```bash
python src/generate/rerank.py \
  --manifest outputs/recSongs/user_007XIjOr/20260423T120000Z-user_007XIjOr-suno/run_manifest.json \
  --top-k 2 \
  --encoder auto
```

### 7) Stage D2 - Eval

```bash
python src/eval/run_eval.py \
  --manifest outputs/recSongs/user_007XIjOr/20260423T120000Z-user_007XIjOr-suno/run_manifest.json \
  --recent-k 20 \
  --reference-top-k 3 \
  --encoder finetuned \
  --save-plot
```

---

## Frontend Demo

```bash
conda env create -f environment.yaml
conda activate gen4rec
streamlit run app/streamlit_app.py
```

## Streamlit Query Comparison

```bash
streamlit run app/streamlit_query_compare.py
```

`streamlit_query_compare.py` is intended to compare query/retrieval behavior between finetuned embeddings and the baseline (zeroshot) setup.
