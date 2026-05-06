# Gen4Rec

## Teammates

1. Tony Zhao (`sz3822`) - [GitHub](https://github.com/Tonyzsp)
2. Conny Fan (`jf4644`) - [GitHub](https://github.com/ConnyFan123)
3. Jerry Huang (`jh8186`) - [GitHub](https://github.com/J-hjr)

---

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

### Human evaluation phases

- **Phase 1 — Base vs fine-tuned retrieval**: text-to-music retrieval with **zeroshot (base)** vs **fine-tuned** CLAP embeddings; export blind clips and a researcher manifest via `scripts/run_phase1_eval.py` (default output: `outputs/phase1_eval/`).
- **Phase 2 — Custom song list (recommendation study)**:
  1. `scripts/user_history_download.py`: CSV → YouTube download → **30 s WAV** under `src/eval/eval_phase_2/<participant>/clips_30s/`.
  2. `scripts/run_phase2_eval.py`: pool WAVs with **finetuned CLAP** → Music4All retrieval → profile + Suno prompt → Suno generation → **rerank by CLAP cosine similarity**. It writes under `src/eval/eval_phase_2/<participant>/result/` with the same layout as `outputs/recSongs/`, i.e. `result/<synthetic_user_id>/<run_id>/` (synthetic id is `phase2_<participant>`), saves `rerank_results.json`, and aliases top outputs as `song1` (higher cosine) and `song2`. The whole `result/` tree is **gitignored** (local only).

## Project Structure (Current)

```text
Gen4Rec/
├── app/
│   ├── services/
│   │   ├── artifact_service.py
│   │   ├── pipeline_service.py
│   │   ├── query_compare.py
│   │   └── viz_service.py
│   ├── streamlit_app.py
├── scripts/
│   ├── build_music4all_aa_index.py
│   ├── run_full_pipeline.py
│   ├── run_phase1_eval.py             # phase 1: base vs finetuned retrieval export → outputs/phase1_eval
│   ├── run_phase2_eval.py             # phase 2: WAV clips → profile/prompt → Suno + rerank → eval_phase_2/<id>/result/
│   └── user_history_download.py       # phase 2: custom song list → WAV under src/eval/eval_phase_2/<id>/
├── src/
│   ├── data/
│   ├── embed/
│   ├── profile_prompt/
│   ├── generate/
│   └── eval/
│       ├── eval_phase_1/              # phase 1 retrieval study: bundled researcher manifest + participant sheet
│       │   ├── manifest.json
│       │   └── participant_instructions.txt
│       ├── eval_phase_2/              # phase 2: CSV template, per-participant WAVs, local result/ (gitignored)
│       │   ├── manifest_template.csv
│       │   ├── <participant>/         # e.g. Tony: manifest.csv, raw/, clips_30s/, result/ (local)
│       ├── run_eval.py
│       ├── clap_audio.py
│       ├── data.py
│       ├── metrics.py
│       ├── reporting.py
│       └── viz.py
├── music4all/                         # local dataset folder (NOT committed)
│   ├── listening_history.csv          # must download manually (large)
│   ├── id_information.csv             # must download manually (large)
│   ├── id_genres.csv                  # must download manually (large)
│   ├── id_tags.csv                    # must download manually (large)
│   ├── id_metadata.csv                # must download manually (large)
│   └── audios/                        # must download manually (very large)
├── music4allA+A/                      # optional A+A metadata for visual enrichment (NOT committed)
│   ├── album_json/
│   ├── artists_json/
│   ├── album_modality_splits.json
│   └── artist_modality_splits.json
├── data/
│   └── derived/                       # optional local derived indexes (NOT committed)
│       └── music4all_aa_song_index.parquet
├── weights/
│   └── clap/                          # local checkpoints (NOT committed)
│       ├── music_audioset_epoch_15_esc_90.14.pt   # download manually
│       └── clap_finetuned_best.pt                 # download manually
├── outputs/                           # generated artifacts (NOT committed)
│   └── phase1_eval/                   # default --out-dir for run_phase1_eval.py (audio/, manifest.json, …)
├── notebooks/
├── environment.yaml
├── environment-windows.yaml
└── .gitignore
```

Note: dataset CSV/audio files and model checkpoints are intentionally not stored in git due size. Each teammate needs to download them locally. Phase 2 per-participant folders (`raw/`, `clips_30s/`, `manifest.csv`, `download_manifest.csv`, `result/`) are also gitignored under `src/eval/eval_phase_2/`; commit only code and `manifest_template.csv`.

### Optional: Music4All A+A Visual Enrichment

The retrieval page can optionally enrich preview cards with Music4All A+A artist/album metadata. This adds album covers, artist images, release dates, listeners/play counts, and artist/album genre chips when the retrieved `song_id` is covered by Music4All A+A.

This does not change the retrieval model or ranking. It only improves the Streamlit frontend display.

Expected local paths (for Music4All A+A enrichment):

```text
Gen4Rec/
├── music4all/
│   └── id_information.csv
├── music4allA+A/
│   ├── album_json/
│   └── artists_json/
└── data/
    └── derived/
        └── music4all_aa_song_index.parquet
```

Build the local Parquet index:

```bash
conda activate gen4rec
python scripts/build_music4all_aa_index.py
```

Equivalent explicit command:

```bash
python scripts/build_music4all_aa_index.py \
  --aa-root music4allA+A \
  --music4all-info music4all/id_information.csv \
  --out data/derived/music4all_aa_song_index.parquet
```

The Streamlit app loads this file automatically if it exists. To use a different index path:

```bash
export GEN4REC_MUSIC4ALL_AA_INDEX_PATH=/path/to/music4all_aa_song_index.parquet
streamlit run app/streamlit_app.py
```

Note: Music4All A+A text summaries are masked for research use (`<Person>`, `<Genre>`), so the frontend currently uses only structured fields and images by default.

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

### Use Your Own Playlist (Suno)

Want to generate personalized Suno music from your own playlist? Use this Phase 2 flow:

1. Create your participant folder and CSV:

```text
src/eval/eval_phase_2/<your_name>/manifest.csv
```

Use a CSV like this:

```csv
song_id,artist,title
the_way,XXXTENTACION,The Way
jocelyn_flores,XXXTENTACION,Jocelyn Flores
whoa,XXXTENTACION,whoa
7_years,Lukas Graham,7 Years
a_sky_full_of_stars,Coldplay,A Sky Full of Stars
a_lot,21 Savage,a lot
```

2. Download songs + build 30s WAV clips:

```bash
python scripts/user_history_download.py \
  --participant-id <your_name> \
  --input src/eval/eval_phase_2/<your_name>/manifest.csv
```

3. Generate profile/prompt and call Suno:

```bash
python scripts/run_phase2_eval.py \
  --participant <your_name> \
  --top-k 10
```

Outputs are written under:

```text
src/eval/eval_phase_2/<your_name>/result/
```

You can directly listen to:
- `.../<run_id>/song1.mp3` (higher CLAP cosine similarity)
- `.../<run_id>/song2.mp3`

---

## Frontend Demo

```bash
conda env create -f environment.yaml
conda activate gen4rec
streamlit run app/streamlit_app.py
```

---

## Data Sources

This project uses three related music datasets:

- **Music4All**: the main dataset this project builds on. We use its listening history, track metadata, genres/tags, audio features, and local audio files to build user embeddings, retrieve similar tracks, create listener profiles, and evaluate generated recommendations.
- **Music4All-Onion**: an extended track-level multimodal dataset related to Music4All. It is not the main dataset used by the core pipeline, but it provides the bridge used by Music4All A+A to map artist/album metadata back to Music4All-style song IDs.
- **Music4All A+A**: an artist- and album-level extension. We use it as an optional enrichment layer for the Streamlit frontend, adding album covers, artist images, artist/album genres, Last.fm tags/links, release dates, listener/play counts, and other structured context.

In short, the recommendation and generation pipeline is built on **Music4All**. **Music4All A+A** is used only as an additional artist/album information source to make retrieval and profile views more interpretable and visually informative.

---

## Citation

Santana, I. A. P., Pinhelli, F., Donini, J., Catharin, L., Mangolin, R. B., da Costa, Y. M. G., Feltrim, V. D., & Domingues, M. A. (2020). *Music4All: A New Music Database and its Applications*. In *Proceedings of the 27th International Conference on Systems, Signals and Image Processing (IWSSIP 2020)* (pp. 1-6). Niterói, Brazil.

Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., & Dubnov, S. (2023). *Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation*. In *ICASSP 2023 - IEEE International Conference on Acoustics, Speech and Signal Processing*.

This project uses the LAION-CLAP checkpoint [`music_audioset_epoch_15_esc_90.14.pt`](https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt) from the `lukewys/laion_clap` Hugging Face repository.
