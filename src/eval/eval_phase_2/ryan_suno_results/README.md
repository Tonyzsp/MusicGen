# User Study 6 – Personalized Music Generation Evaluation

## Overview

This user study evaluates the effectiveness of our personalized music generation pipeline using a real user (Ryan). The evaluation focuses on whether the system can capture Ryan’s listening preferences from recent listening history, generate a meaningful user profile and Suno prompt, and produce personalized music that aligns with human judgment.

Ryan’s case is especially useful because his listening preferences are highly fluid and genre-switching, moving across electronic, synthpop, lo-fi, experimental, emo, noise pop, and other styles. The generated profile received a high user rating, suggesting that the pipeline successfully captured the main sonic elements and emotional direction of his listening history, even though his taste changes frequently across contexts.

---

## Evaluation Metrics

We evaluate generated music along three dimensions:

- **Preference Match (1–5)**: How well the song aligns with the user's taste  
- **Sound Quality (1–5)**: Audio coherence and production quality  
- **Creativity (1–5)**: Novelty and distinctiveness  

We also collect qualitative feedback on:

- Accuracy of the generated user profile
- Quality of the personalized Suno prompt
- Whether the generated songs match the user’s genre, mood, and atmosphere preferences
- Whether CLAP-based reranking aligns with the user’s final preference

---

## Methodology

### Step 1: User Embedding Construction

- User ID: `phase2_ryan`
- Input: Ryan’s 10 most recent songs
- Each song is encoded using the fine-tuned CLAP model
- The 10 song embeddings are used to form a **single Ryan user embedding**

---

### Step 2: Listening History Summary

Ryan’s recent listening history shows a preference for emotionally expressive, vocal-led music that blends electronic, synthpop, lo-fi, experimental, new wave, emo, and noise-pop elements.

#### Top Genres

- electronic
- experimental
- synthpop
- lo-fi
- new wave
- emo
- noise pop
- black metal

#### Top Tags

- electronic
- experimental
- synthpop
- lo-fi
- new wave
- 80s
- cloud rap
- emo
- noise pop
- 2017

#### Representative Artists

- Sleigh Bells
- Depeche Mode
- Mount Eerie
- BRENNAN SAVAGE
- Rex Orange County
- The Human League

#### Representative Tracks

| Artist | Track |
|--------|-------|
| Sleigh Bells | Favorite Transgressions |
| Depeche Mode | Stories of Old |
| Mount Eerie | Lost Wisdom Pt. 2 |
| BRENNAN SAVAGE | Bulletproof |
| Rex Orange County | Green Eyes, Pt. II |

---

### Step 3: Audio & Mood Profile

Ryan’s input songs are fully vocal-focused.

| Attribute | Value |
|----------|-------|
| Dominant Mode | Vocal |
| Vocal / Language Ratio | 1.0 |
| Instrumental Ratio | 0.0 |
| Average Danceability | 0.551 |
| Average Energy | 0.578 |
| Average Valence | 0.368 |
| Average Tempo | 120.485 BPM |

#### Mood Summary

- energetic
- driving
- bittersweet
- reflective
- mid-tempo

This suggests that Ryan’s taste leans toward mid-tempo, vocal-led songs with a balance of movement and emotional reflection. His listening history contains energetic and driving qualities, but the relatively low valence also suggests a preference for bittersweet or moody emotional tones rather than purely bright pop energy.

---

## Profile & Prompt Generation

### Generated User Profile

> This listener seems drawn to emotionally charged electronic music that balances synthpop polish with lo-fi grit and experimental edges. Their taste leans toward mid-tempo, driving tracks with expressive vocals, moody melodies, and a bittersweet reflective tone. There’s a clear attraction to 80s-leaning new wave textures, cold synth layers, punchy drum programming, and occasional noise-pop distortion or indie-emo fragility. They may be drawn to artists such as Sleigh Bells, Depeche Mode, Mount Eerie, and The Human League, alongside more intimate alt-pop voices. Overall, the aesthetic feels atmospheric but kinetic: glossy enough to move, rough enough to feel personal.

### Generated Suno Prompt

> Vocal-led synthpop/electronic track with new wave sheen, lo-fi edge, and experimental noise-pop accents. Mid-tempo 120 BPM, driving drums, pulsing bass, chilly analog synths, layered harmonies, bittersweet and reflective mood. Intimate male vocal with crisp, emotive delivery. Dense but clean arrangement, 80s-inspired yet modern. Avoid overly polished pop and heavy metal guitars.

### Style Keywords

- synthpop
- new wave
- electronic
- lo-fi grit
- bittersweet
- driving mid-tempo
- experimental edge

---

## User Feedback on Profile & Prompt

- Profile Accuracy: **9/10**

User explanation:

> The generated profile is quite aligned with my taste. It has a slight 80s retro-pop feeling, which makes sense. My listening style changes a lot: one moment I may listen to rap, and the next moment I may switch to jazz or something completely different. I might not want to listen to this style today, but tomorrow I may come back to it, and the day after that my preference may change again. Still, the description is quite accurate, and the elements it identifies are aligned with my taste.

This feedback suggests that the profile generation step worked well for Ryan’s case. The system captured the main stylistic and emotional elements in his recent listening history, especially synthpop, new wave, electronic textures, lo-fi grit, and bittersweet reflective mood. However, the user also emphasized that his music preference is highly context-dependent and changes frequently, which may make a single static profile less stable over time.

---

## Music Generation

Using the personalized Suno prompt, music was generated for Ryan.

The final evaluation used the top two reranked songs selected by the fine-tuned CLAP model.

- Run ID: `20260506T043906Z-phase2_ryan-suno`
- Rerank Top-K: 2

---

## CLAP Reranking Results

| Rank | Track | CLAP Cosine Score |
|------|-------|-------------------|
| Top 1 | Track A / song1 | 0.6289 |
| Top 2 | Track B / song2 | 0.5238 |

---

## Generated Tracks

### Track A: song1

- CLAP cosine score: **0.6289**
- Local file:  
  `/Users/conny_fan/Desktop/Gen4Rec/src/eval/eval_phase_2/ryan/result/phase2_ryan/20260506T043906Z-phase2_ryan-suno/audio/call_01/phase2_ryan_rec_song_variant_02.mp3`

### Track B: song2

- CLAP cosine score: **0.5238**
- Local file:  
  `/Users/conny_fan/Desktop/Gen4Rec/src/eval/eval_phase_2/ryan/result/phase2_ryan/20260506T043906Z-phase2_ryan-suno/audio/call_01/phase2_ryan_rec_song_variant_01.mp3`

---

## Reranking vs Human Preference

- **Pipeline-selected track via CLAP reranking:** Track A / song1  
- **User-selected favorite:** Track A / First Song  

The CLAP reranking result is consistent with Ryan’s human preference in this case.

Track A received the higher CLAP cosine score and was also selected by Ryan as the preferred song. This suggests that, for Ryan’s case, the fine-tuned CLAP reranking was able to identify the generated song that better matched his recent listening profile.

---

## Human Evaluation Results

| Track | Preference Match | Sound Quality | Creativity |
|------|------------------|--------------|-----------|
| Track A / song1 | 5 | 4 | 4 |
| Track B / song2 | 4 | 4 | 4 |

---

## User Qualitative Feedback

Ryan preferred the first generated song.

The first song received a higher preference match score, with **5/5** for preference alignment, while the second song received **4/5**. Both songs received the same sound quality and creativity scores, suggesting that the main difference between the two tracks was not production quality or novelty, but how closely the song matched Ryan’s current taste.

This response indicates that the personalized prompt was effective at generating songs in Ryan’s preferred direction, especially around vocal-led electronic music, synthpop/new wave textures, lo-fi edge, and bittersweet mid-tempo energy.

---

## Key Findings

### Strengths

- The generated profile received a high **9/10** accuracy rating
- The system accurately captured Ryan’s electronic, synthpop, new wave, lo-fi, and experimental music preferences
- The profile successfully reflected the user’s interest in vocal-led, bittersweet, reflective, and mid-tempo tracks
- The generated prompt captured key musical elements such as chilly synth layers, driving drums, pulsing bass, lo-fi grit, and experimental noise-pop accents
- CLAP-based reranking aligned with Ryan’s human preference
- Track A achieved a perfect **5/5** preference match score
- Both generated songs received strong sound quality and creativity scores

---

### Limitations

- Ryan’s listening preference is highly fluid and context-dependent, so a single profile may not fully represent his taste over time
- The generated profile leaned toward 80s-inspired synthpop and retro electronic textures, which matched the current input but may not reflect every future listening mood
- Although Track A was preferred, both songs received the same creativity score, suggesting that the generation may not have created a strong enough creative distinction between candidates
- Future versions could incorporate short-term mood selection or user-controlled style sliders to better handle users with rapidly shifting listening preferences

---

## Notes

- Ryan’s user embedding was constructed from 10 recent songs
- The input listening history was fully vocal-based, with a **vocal/language ratio of 1.0**
- The average tempo of the input songs was around **120 BPM**, supporting the mid-tempo prompt direction
- The generated prompt focused on vocal-led synthpop/electronic music, new wave sheen, lo-fi edge, experimental noise-pop accents, and bittersweet reflective mood
- The final recommendation selected by CLAP reranking was also the user’s preferred song
- This case shows that the personalized pipeline can capture a user’s current listening direction, but users with highly variable taste may require more dynamic or session-based personalization
