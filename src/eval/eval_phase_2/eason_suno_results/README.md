# User Study 1 – Personalized Music Generation Evaluation

## Overview

This is the very first user study we've conducted in Milestone 2, using the previous pipeline. But the result is still useful. This user study evaluates the effectiveness of our personalized music generation pipeline using a real user (Eason). The evaluation focuses on both automatic reranking results and human judgment.

---

## Evaluation Metrics

We evaluate generated music along three dimensions:

- **Preference Match (1–5)**: How well the song aligns with the user's taste  
- **Sound Quality (1–5)**: Audio coherence and production quality  
- **Creativity (1–5)**: Novelty and distinctiveness  

We also conduct a **pairwise comparison (A/B test)** between:
- Personalized generation (our pipeline)
- Baseline (random / generic prompt)

---

## Methodology

### Step 1: User Embedding Construction

- Input: Eason’s 10 most recent songs (`.wav`)
- Each song is encoded into a CLAP embedding
- The 10 embeddings are averaged to form a **single user embedding**

---

### Step 2: Retrieval

- The user embedding is compared against **109,269 songs** in `music4all_embeddings.npy`
- Similarity metric: cosine similarity
- Top-5 most similar songs are retrieved

---

### Step 3: Profile & Prompt Generation

- Metadata from Top-5 songs is used to generate:
  - A natural language **user profile**
  - A **music generation prompt**

#### User Feedback on Profile

- Accuracy: **8/10**
- Strength:
  - Accurately summarizes known preferences
- Limitation:
  - Does not reveal deeper or latent traits of the user
  - May include artists not actually listened to (due to dataset-based retrieval)

> Potential improvement: avoid explicitly mentioning artist names in generated profiles

---

### Step 4: Music Generation

Three generation attempts were conducted:

| Attempt | Result |
|--------|--------|
| Run 1 | Failed (no audio generated) |
| Run 2 | Failed (no audio generated) |
| Run 3 | Successful |

We proceed with the successful run:

- Run ID: `20260402T203009Z-eason-suno`
- Number of generated candidates: 2

---

## Generated Tracks

- Track A:  
  https://cdn1.suno.ai/a218d7f5-afeb-47b0-8610-5a93dcf038ae.mp3  

- Track B:  
  https://cdn1.suno.ai/02bf76ad-89be-4c77-be1b-0bbf807fbe52.mp3  

---

## Reranking vs Human Preference

- **Pipeline-selected track (via CLAP reranking):** Track B (cosine score 0.22 > 0.19)
- **User-selected favorite:** Track B  

✔ The reranking result is consistent with human preference

User explanation:
> Track B is more distinctive, although both tracks align well with preferences.

---

## Human Evaluation Results

| Track | Preference Match | Sound Quality | Creativity |
|------|------------------|--------------|-----------|
| A    | 5                | 4            | 3         |
| B    | 5                | 4            | 4         |

---

## Key Findings

### Strengths

- The pipeline successfully generates music aligned with user preferences  
- CLAP-based reranking aligns well with human judgment  
- Personalized prompts lead to high preference match scores  

---

### Limitations

- Profile generation mainly summarizes existing preferences rather than uncovering new insights  
- Retrieved metadata may introduce artists not actually listened to by the user  
- Some generated outputs lack vocal elements  
- Generation is not stable (multiple failed runs observed)

---

## Notes

- Only the successful generation run is included in evaluation  
- Audio files are hosted externally due to size constraints 
