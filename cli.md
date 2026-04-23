# Gen4Rec CLI (Phase-by-Phase)

This document uses only stage-by-stage terminal commands.

---

## 1. Environment Setup

```bash
conda activate gen4rec
```

Common API keys:

```bash
export OPENAI_API_KEY="..."
export ACE_SUNO_API_KEY="..."
```

---

## 2. Default Locations

Relative to repo root:

- Dataset: `music4all/`
- Embeddings: `outputs/embeddings/music4all/`
- Profiles: `outputs/profiles/`
- Generation runs: `outputs/recSongs/<USER_ID>/<RUN_OR_KEY>/`
- Eval outputs: `outputs/eval/<USER_ID>/<RUN_OR_KEY>/`

---

## 3. Stage-by-Stage Commands

Demo values used below:

- `user_id`: `user_007XIjOr`
- embedding variant: `rk10_dl0.08_mt0.2_mk5`
- profile key: `evrk10_dl0.08_mt0.2_mk5_tk20_msnone_ex1_omgpt-5.4-mini`
- example run id: `20260423T120000Z-user_007XIjOr-suno` (replace with your actual Stage C output if different)

## 3.1 Stage A: User Embeddings

```bash
python src/embed/build_user_embeddings.py \
  --recent-k 10 \
  --decay-lambda 0.08 \
  --medoid-threshold 0.2 \
  --min-keep 5
```

Force rebuild:

```bash
python src/embed/build_user_embeddings.py ... --force
```

Naming:

- Variant: `rk{recent_k}_dl{decay}_mt{medoid}_mk{min_keep}`
- Outputs:
  - `user_embeddings__<variant>.npy`
  - `user_ids__<variant>.npy`
  - `user_embedding_stats__<variant>.csv`

## 3.2 Stage B: Retrieval Export (Raw Profile JSON)

```bash
python src/embed/export_user_profile_json.py \
  --user-id user_007XIjOr \
  --top-k 20 \
  --exclude-recent \
  --user-emb-path outputs/embeddings/music4all/user_embeddings__rk10_dl0.08_mt0.2_mk5.npy \
  --user-ids-path outputs/embeddings/music4all/user_ids__rk10_dl0.08_mt0.2_mk5.npy \
  -o outputs/profiles/user_007XIjOr.json
```

Optional flags:

- `--min-similarity <float>`
- `--top-m` (legacy alias of `--top-k`)

## 3.3 Stage B: Unified Profile Pipeline (Recommended)

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
python src/profile_prompt/run_profile_pipeline.py ... --rebuild
```

Profile naming:

- Profile variant: `ev{embedding_variant}_tk{top_k}_ms{sim}_ex{0|1}_om{model}`
- Outputs:
  - `outputs/profiles/<USER_ID>__<PROFILE_KEY>.json`
  - `outputs/profiles/<USER_ID>__<PROFILE_KEY>_topk_summary.json`
  - `outputs/profiles/<USER_ID>__<PROFILE_KEY>_prompt.json`
  - `outputs/profiles/<USER_ID>__<PROFILE_KEY>_validation.json`

## 3.4 Stage C: Generation

```bash
python src/generate/run_generate.py \
  --prompt-json outputs/profiles/user_007XIjOr__evrk10_dl0.08_mt0.2_mk5_tk20_msnone_ex1_omgpt-5.4-mini_prompt.json \
  --generation-model chirp-v4-5 \
  --num-calls 1 \
  --max-concurrency 1
```

Optional:

- `--negative-prompt "..."`
- `--lyrics-file lyrics.txt`
- `--tempo-hint-bpm 120`
- `--duration-hint-seconds 30`

Outputs:

- `outputs/recSongs/<USER_ID>/<RUN_OR_KEY>/run_manifest.json`
- `outputs/recSongs/<USER_ID>/<RUN_OR_KEY>/report.md`
- `outputs/recSongs/<USER_ID>/<RUN_OR_KEY>/audio/...`

For the next two stages, use the run id printed by Stage C. Example run root:

- `outputs/recSongs/user_007XIjOr/20260423T120000Z-user_007XIjOr-suno/`

## 3.5 Stage D1: Rerank

```bash
python src/generate/rerank.py \
  --manifest outputs/recSongs/user_007XIjOr/20260423T120000Z-user_007XIjOr-suno/run_manifest.json \
  --top-k 2 \
  --encoder auto
```

Optional:

- `--diversity-threshold 0.95`
- `--output <custom_path>`

Default output:

- `outputs/recSongs/<USER_ID>/<RUN_OR_KEY>/rerank_results.json`

## 3.6 Stage D2: Eval

```bash
python src/eval/run_eval.py \
  --manifest outputs/recSongs/user_007XIjOr/20260423T120000Z-user_007XIjOr-suno/run_manifest.json \
  --recent-k 20 \
  --reference-top-k 3 \
  --encoder finetuned \
  --save-plot
```

Default outputs:

- `outputs/eval/<USER_ID>/<RUN_OR_KEY>/eval_summary.json`
- `outputs/eval/<USER_ID>/<RUN_OR_KEY>/eval_report.md`
- `outputs/eval/<USER_ID>/<RUN_OR_KEY>/reference_alignment.csv`
- `outputs/eval/<USER_ID>/<RUN_OR_KEY>/embedding_space.png` (`--save-plot`)

---

## 5. Reuse and Rebuild Rules

Shared principle:

- Same parameters -> reuse existing artifacts
- Changed parameters -> new key / new path
- Explicit `--force` / `--rebuild` -> recompute under same key

This is exactly why artifact names are parameter-encoded: naming is used as a deterministic reuse/rebuild contract across phases.

---

## 6. Recommended Execution Order

Run the phases in this order:

1. Stage A (`build_user_embeddings.py`)
2. Stage B (`run_profile_pipeline.py`, or retrieval export first)
3. Stage C (`run_generate.py`)
4. Stage D1 (`rerank.py`)
5. Stage D2 (`run_eval.py`)
