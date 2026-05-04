# User Study 2 – Personalized Music Generation Evaluation

## Overview

This user study evaluates the effectiveness of our personalized music generation pipeline using a real user (Irene). The evaluation focuses on both automatic CLAP-based reranking results and human judgment.

Compared with the previous Eason case, Irene's study includes a fuller generation process: the same personalized prompt was used to generate music three times, and the final candidates were reranked using the fine-tuned CLAP model.

---

## Evaluation Metrics

We evaluate generated music along three dimensions:

- **Preference Match (1–5)**: How well the song aligns with the user's taste  
- **Sound Quality (1–5)**: Audio coherence and production quality  
- **Creativity (1–5)**: Novelty and distinctiveness  

We also conduct a **pairwise comparison (A/B test)** between generated candidates to compare:

- The CLAP-based reranking result
- The user's actual listening preference

---

## Methodology

### Step 1: User Embedding Construction

- Input: Irene’s 10 most recent songs (`.wav`)
- Each song is encoded using the fine-tuned CLAP model
- The 10 song embeddings are averaged to form a **single Irene user embedding**

---

### Step 2: Retrieval

- Irene’s user embedding is compared against **109,269 songs** in the Music4All embedding database
- Similarity metric: cosine similarity
- The Top-5 most similar songs are retrieved

---

### Step 3: Profile & Prompt Generation

Metadata from the Top-5 retrieved songs is used to generate:

- A natural language **Irene user profile**
- A personalized **Suno music generation prompt**

#### User Feedback on Profile

- Accuracy: **8/10**

User explanation:

> The profile captures my usual listening style well because the generated music is rhythmic and upbeat, which matches what I normally like. However, the melody feels slightly messy, as if many instrumental and melodic elements are stacked together, without one clear and unified main melody.

---

### Step 4: Music Generation

Using the same personalized prompt, we generated music three times.

- Number of generation attempts: 3
- Suno usually returns 2 songs per generation attempt
- Total generated candidates: 6

These 6 generated tracks were then reranked using the fine-tuned CLAP model.

---

## CLAP Reranking Results

### Irene Generation Rerank Top 2

| Rank | Track | CLAP Cosine Score |
|------|-------|-------------------|
| Top 1 | Salt on the Window variant 2 | 0.6090 |
| Top 2 | Salt on the Window variant 1 | 0.6022 |

---

## Generated Tracks

### Track A: Salt on the Window variant 1

- CLAP cosine score: **0.6022**
- Local file:  
  `/Users/conny_fan/Desktop/Gen4Rec/outputs/recSongs/irene/20260503T054322Z-irene-suno/audio/call_01/salt_on_the_window_variant_01.mp3`
- Source URL:  
  https://cdn1.suno.ai/6707b88e-e82b-473f-b9c2-37237683ed5e.mp3

### Track B: Salt on the Window variant 2

- CLAP cosine score: **0.6090**
- Local file:  
  `/Users/conny_fan/Desktop/Gen4Rec/outputs/recSongs/irene/20260503T054322Z-irene-suno/audio/call_01/salt_on_the_window_variant_02.mp3`
- Source URL:  
  https://cdn1.suno.ai/5c6d07ec-eca7-428a-80f0-c020c2e2a4b6.mp3

---

## Reranking vs Human Preference

- **Pipeline-selected track via CLAP reranking:** Track B  
- **User-selected favorite:** Track A  

The CLAP reranking result is **not fully consistent** with Irene’s human preference in this case.

Although Track B receives a slightly higher CLAP cosine score, Irene preferred Track A. This suggests that embedding similarity can capture general preference alignment, but it may not fully reflect subjective factors such as melody clarity, emotional preference, or perceived musical coherence.

---

## Human Evaluation Results

| Track | Preference Match | Sound Quality | Creativity |
|------|------------------|--------------|-----------|
| Track A: Salt on the Window variant 1 | 4 | 5 | 2 |
| Track B: Salt on the Window variant 2 | 3 | 3 | 2 |

---

## Key Findings

### Strengths

- The personalized pipeline successfully generated music that broadly matched Irene’s listening preferences
- Irene rated the generated profile as **8/10**, suggesting that the profile captured her general taste reasonably well
- The generated tracks were rhythmic and upbeat, which aligned with Irene’s usual listening style
- The fine-tuned CLAP model produced close reranking scores for the top two tracks, indicating that both candidates were relatively similar to Irene’s user embedding

---

### Limitations

- The CLAP reranking result did not match Irene’s final preference
- The higher-scoring track was not necessarily the one the user liked more
- Irene noted that the generated music felt somewhat cluttered, with too many instrumental or melodic elements layered together
- Both tracks received relatively low creativity scores, suggesting that the outputs may have lacked novelty or distinctiveness
- The current reranking method may not fully capture human preferences related to melody structure, clarity, and overall musical coherence

---

## Notes

- Irene’s user embedding was constructed from 10 `.wav` songs
- The retrieval database contains **109,269 Music4All song embeddings**
- The final recommendation was selected from 6 generated candidates using fine-tuned CLAP reranking
- Audio files are hosted externally due to size constraints
- This case shows that CLAP-based reranking is useful, but human evaluation remains necessary for assessing subjective music quality
