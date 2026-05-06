# User Study 4 – Personalized Music Generation Evaluation

## Overview

This user study evaluates the effectiveness of our personalized music generation pipeline using a real user (Nancy). The evaluation focuses on whether the system can capture Nancy’s listening preferences from recent listening history, generate a meaningful user profile and Suno prompt, and produce personalized music that aligns with human judgment.

Nancy’s case is especially useful because the generated profile partially captured the emotional and vocal-centered side of her taste, but also revealed an important limitation: the profile leaned too heavily toward dark emo rap and experimental hip hop, while the user’s actual preference was softer, warmer, more romantic, and more atmospheric. In this case, CLAP-based reranking was consistent with Nancy’s human preference, since the top-ranked song was also the user-selected favorite.

---

## Evaluation Metrics

We evaluate generated music along three dimensions:

- **Preference Match (1–5)**: How well the song aligns with the user's taste  
- **Sound Quality (1–5)**: Audio coherence and production quality  
- **Creativity (1–5)**: Novelty and distinctiveness  

We also collect qualitative feedback on:

- Accuracy of the generated user profile
- Quality of the personalized Suno prompt
- Whether the generated songs match the user’s emotional and atmospheric preferences
- Whether CLAP-based reranking aligns with the user’s final preference

---

## Methodology

### Step 1: User Embedding Construction

- User ID: `phase2_Nancy`
- Input: Nancy’s 10 most recent songs
- Each song is encoded using the fine-tuned CLAP model
- The 10 song embeddings are used to form a **single Nancy user embedding**

---

### Step 2: Listening History Summary

Nancy’s recent listening history shows a preference for vocal-led, emotionally expressive music that blends rap, emo, hip hop, experimental textures, and atmospheric production.

#### Top Genres

- rap
- emo
- hip hop
- experimental
- emo rap
- death metal
- glitch hop
- lo-fi

#### Top Tags

- rap
- hip-hop
- emo
- cloud rap
- hip hop
- experimental
- emo rap
- post-emo
- post emo
- emotional trap

#### Representative Artists

- Lil Peep
- BRENNAN SAVAGE
- Sybyr
- JPEGMAFIA
- Joji
- Мы

#### Representative Tracks

| Artist | Track |
|--------|-------|
| BRENNAN SAVAGE | Bulletproof |
| Sybyr | 101Dalmatians |
| Lil Peep | White Girl |
| Lil Peep | Better Off (Dying) |
| JPEGMAFIA | Rainbow Six |

---

### Step 3: Audio & Mood Profile

Nancy’s input songs are mostly vocal-focused, with a small instrumental component.

| Attribute | Value |
|----------|-------|
| Dominant Mode | Vocal |
| Vocal / Language Ratio | 0.9 |
| Instrumental Ratio | 0.1 |
| Average Danceability | 0.584 |
| Average Energy | 0.629 |
| Average Valence | 0.298 |
| Average Tempo | 120.820 BPM |

#### Mood Summary

- energetic
- driving
- bittersweet
- reflective
- mid-tempo

This suggests that Nancy’s taste contains emotionally expressive and mid-tempo qualities, but the user feedback indicates that the system overemphasized darker and more abrasive elements while underrepresenting the softer, warmer, and more romantic atmosphere of her actual listening preference.

---

## Profile & Prompt Generation

### Generated User Profile

> This listener is drawn to a dark, emotionally charged blend of rap, emo rap, cloud rap, and experimental hip hop, with occasional edges of lo-fi texture and heavier, more abrasive energy. The overall taste leans toward bittersweet, reflective, and driving tracks that pair intimate, vulnerable vocals with sparse or glitchy production, then build into hard-hitting, cathartic peaks. They may be drawn to artists such as Lil Peep, BRENNAN SAVAGE, Sybyr, JPEGMAFIA, and Joji, especially when songs balance melodic sadness with restless rhythmic momentum and an underground, rough-edged aesthetic.

### Generated Suno Prompt

> Vocal-led emo rap / experimental hip hop, 120 BPM mid-tempo. Melodic, emotionally raw vocals over sparse trap drums, hazy pads, distorted 808s, glitchy textures, and occasional lo-fi grit. Bittersweet, reflective, driving energy; intimate verses with a cathartic hook, dense but not overproduced. Avoid bright pop polish and cheerful tones.

### Style Keywords

- emo rap
- experimental hip hop
- cloud rap
- bittersweet
- glitchy drums
- distorted 808s
- lo-fi grit

---

## User Feedback on Profile & Prompt

- Profile + Prompt Accuracy: **7/10**

User explanation:

> I would give it 7/10 because it captures some emotional and vocal-centered qualities of the input songs, such as bittersweet mood, reflective energy, and intimate vocals. However, since my actual input songs are more romantic and atmospheric, the profile leans too much toward dark emo rap, experimental hip hop, distorted 808s, and glitchy textures. It partially matches the emotional side of my taste, but it misses the softer, warmer, and more romantic love-song feeling.

This feedback suggests that the profile generation step captured the broad emotional direction, but did not fully capture the user’s preferred atmosphere. The system correctly identified bittersweet and vocal-centered traits, but over-weighted darker genre labels and production textures.

---

## Music Generation

Using the personalized Suno prompt, music was generated for Nancy.

The final evaluation used the top two reranked songs selected by the fine-tuned CLAP model.

- Run ID: `20260505T143847Z-phase2_Nancy-suno`
- Rerank Top-K: 2
- Manifest path:  
  `C:\Users\Nancy\Desktop\Gen4Rec\src\eval\eval_phase_2\Nancy\result\phase2_Nancy\20260505T143847Z-phase2_Nancy-suno\run_manifest.json`
- Generation artifacts root:  
  `C:\Users\Nancy\Desktop\Gen4Rec\src\eval\eval_phase_2\Nancy\result\phase2_Nancy\20260505T143847Z-phase2_Nancy-suno`
- Rerank output path:  
  `C:\Users\Nancy\Desktop\Gen4Rec\src\eval\eval_phase_2\Nancy\result\phase2_Nancy\20260505T143847Z-phase2_Nancy-suno\rerank_results.json`

---

## CLAP Reranking Results

| Rank | Track | CLAP Cosine Score |
|------|-------|-------------------|
| Top 1 | Track A / song1 | 0.6369 |
| Top 2 | Track B / song2 | 0.6012 |

---

## Generated Tracks

### Track A: song1

- CLAP cosine score: **0.6369**
- Local file:  
  `C:\Users\Nancy\Desktop\Gen4Rec\src\eval\eval_phase_2\Nancy\result\phase2_Nancy\20260505T143847Z-phase2_Nancy-suno\audio\call_01\phase2_nancy_rec_song_variant_01.mp3`

### Track B: song2

- CLAP cosine score: **0.6012**
- Local file:  
  `C:\Users\Nancy\Desktop\Gen4Rec\src\eval\eval_phase_2\Nancy\result\phase2_Nancy\20260505T143847Z-phase2_Nancy-suno\audio\call_01\phase2_nancy_rec_song_variant_02.mp3`

---

## Reranking vs Human Preference

- **Pipeline-selected track via CLAP reranking:** Track A / song1  
- **User-selected favorite:** Track A / First Song  

The CLAP reranking result is consistent with Nancy’s human preference in this case.

Track A received the higher CLAP cosine score and was also the user’s preferred song. This suggests that the fine-tuned CLAP reranking successfully identified the candidate that better matched Nancy’s recent listening profile, even though the profile itself only partially captured her intended emotional style.

---

## Human Evaluation Results

| Track | Preference Match | Sound Quality | Creativity |
|------|------------------|--------------|-----------|
| Track A / song1 | 5 | 5 | 4 |
| Track B / song2 | 3 | 5 | 5 |

---

## User Qualitative Feedback

> The generated songs fit my taste well, and both the sound and the lyrics generally matched the songs I uploaded. The second song had more noticeable changes in its overall listening experience, so I would rate it higher in creativity. However, I still personally preferred the first song overall because it felt closer to the atmosphere and style I wanted.

This response indicates that both generated songs achieved strong sound quality, but preference alignment differed. Track A was preferred because it better matched Nancy’s desired atmosphere and style, while Track B was considered more creative due to its more noticeable changes in listening experience.

Track A also received the higher CLAP cosine score, which aligns with Nancy’s preference judgment. However, Track B’s higher creativity score suggests that CLAP similarity may be better at identifying preference alignment than perceived novelty.

---

## Key Findings

### Strengths

- The generated profile captured some emotional and vocal-centered qualities of Nancy’s listening history
- The system correctly identified bittersweet mood, reflective energy, and intimate vocals as important preference signals
- CLAP-based reranking aligned with Nancy’s human preference
- Track A received the highest CLAP cosine score and was also the user-selected favorite
- Both generated tracks received perfect sound quality scores
- Track A achieved a perfect preference match score, suggesting that the generation step still produced a highly aligned candidate
- Track B showed stronger perceived creativity, indicating that the model can generate more varied musical outputs from the same personalized direction

---

### Limitations

- The profile and prompt received a moderate **7/10** accuracy rating, lower than Tony’s case
- The system overemphasized dark emo rap, experimental hip hop, distorted 808s, and glitchy textures
- The generated profile missed the softer, warmer, and more romantic love-song feeling that Nancy associated with her actual input songs
- Genre and tag-based summaries may overweight prominent labels while underrepresenting emotional nuance
- Although CLAP reranking matched human preference, the score difference between the top two tracks was relatively small, so human validation remains important
- Future prompt generation should better distinguish between “melancholic/dark” and “romantic/atmospheric” emotional styles

---

## Notes

- Nancy’s user embedding was constructed from 10 recent songs
- The input listening history was mostly vocal-based, with a **0.9 vocal/language ratio**
- The average tempo of the input songs was around **121 BPM**, supporting the mid-tempo prompt direction
- The generated prompt focused on vocal-led emo rap, experimental hip hop, glitchy textures, distorted 808s, and bittersweet emotional delivery
- The final recommendation selected by CLAP reranking was also the user’s preferred song
- This case shows that personalized music generation can produce high-quality and preference-aligned outputs even when the profile itself only partially matches the user’s intended emotional style
- Nancy’s feedback highlights the importance of capturing atmosphere-level distinctions, especially between darker emo-rap aesthetics and softer romantic emotional preferences
