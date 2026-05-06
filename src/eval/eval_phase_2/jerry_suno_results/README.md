# User Study 5 – Personalized Music Generation Evaluation

## Overview

This user study evaluates the effectiveness of our personalized music generation pipeline using a real user (Jerry). The evaluation focuses on whether the system can capture Jerry’s listening preferences from recent listening history, generate a meaningful user profile and Suno prompt, and produce personalized music that aligns with human judgment.

Jerry’s case is especially useful because his listening history is centered on more niche and texture-driven genres, including experimental rock, post-rock, industrial rock, ambient, and freak folk. The generated profile received a high user rating, suggesting that the pipeline successfully captured both genre-level preferences and emotion-level listening patterns.

---

## Evaluation Metrics

We evaluate generated music along three dimensions:

- **Preference Match (1–5)**: How well the song aligns with the user's taste  
- **Sound Quality (1–5)**: Audio coherence and production quality  
- **Creativity (1–5)**: Novelty and distinctiveness  

We also collect qualitative feedback on:

- Accuracy of the generated user profile
- Quality of the personalized Suno prompt
- Whether the generated songs match the user’s genre, mood, and instrumental preferences
- Whether CLAP-based reranking aligns with the user’s final preference

---

## Methodology

### Step 1: User Embedding Construction

- User ID: `phase2_jerry`
- Input: Jerry’s 10 most recent songs
- Each song is encoded using the fine-tuned CLAP model
- The 10 song embeddings are used to form a **single Jerry user embedding**

---

### Step 2: Listening History Summary

Jerry’s recent listening history shows a preference for experimental, atmospheric, and texture-driven rock music. His input songs include a balanced mix of instrumental and vocal-led tracks, suggesting that his taste is shaped by both sonic texture and emotional delivery.

#### Top Genres

- experimental
- rock
- metal
- post-rock
- industrial rock
- industrial
- ambient
- freak folk

#### Top Tags

- experimental
- instrumental
- rock
- metal
- post-rock
- industrial rock
- industrial
- ambient
- indie
- freak folk

#### Representative Artists

- Nine Inch Nails
- Animal Collective
- Död Mark
- Queen
- Siouxsie and the Banshees
- The Claypool Lennon Delirium

#### Representative Tracks

| Artist | Track |
|--------|-------|
| Nine Inch Nails | Ripe (With Decay) |
| Animal Collective | Airpipe (To a New Transition) |
| Död Mark | Drabbad av sjukdom |
| Queen | Procession |
| Siouxsie and the Banshees | Pure |

---

### Step 3: Audio & Mood Profile

Jerry’s input songs show a mixed vocal and instrumental profile.

| Attribute | Value |
|----------|-------|
| Dominant Mode | Mixed |
| Vocal / Language Ratio | 0.5 |
| Instrumental Ratio | 0.5 |
| Average Danceability | 0.500 |
| Average Energy | 0.328 |
| Average Valence | 0.262 |
| Average Tempo | 105.752 BPM |

#### Mood Summary

- soft
- balanced
- bittersweet
- reflective
- mid-tempo

This suggests that Jerry’s taste is not primarily focused on high-energy or mainstream rock. Instead, his listening preference leans toward reflective, muted, texture-heavy music with cinematic or experimental qualities.

---

## Profile & Prompt Generation

### Generated User Profile

> This listener is drawn to adventurous rock at the edge of genre: experimental, industrial, post-rock, metal, ambient, and freak folk, with a strong split between instrumental passages and vocal-led tracks. The overall feel is moody and reflective rather than aggressive, favoring mid-tempo pacing, muted dynamics, and a bittersweet atmosphere over high-energy release. They seem to appreciate textured production, shifting arrangements, and songs that balance heft with detail, from abrasive industrial grit to spacious, cinematic build-ups. They may be especially responsive to artists such as Nine Inch Nails, Animal Collective, Död Mark, Queen, and Siouxsie and the Banshees for their contrast of tension, drama, and experimentation.

### Generated Suno Prompt

> Experimental post-rock/industrial rock with ambient textures and freak-folk details, mid-tempo, moody and bittersweet. Mix instrumental sections with restrained male vocal lines; layered guitars, cracked synths, distant bass, brushed drums, and dense but airy builds. Cinematic, dark, reflective, 105 BPM. Avoid upbeat pop hooks and polished brightness.

### Style Keywords

- experimental rock
- industrial grit
- post-rock build
- ambient haze
- bittersweet
- mid-tempo
- textured production
- mixed vocals

---

## User Feedback on Profile & Prompt

- Profile Accuracy: **9/10**

User explanation:

> The system is very accurate in analyzing the style of my songs, especially in capturing genre tags. The music I listen to is indeed experimental rock and post-rock. The profile also captures the emotional side of my listening experience very accurately, which matches the emotional connection I have with this type of music. It also includes instrumental descriptions that accurately describe the characteristics of the music. Although the profile mentions many musicians that I may not necessarily know, it does a good job summarizing them in a way that helps me understand that these artists are likely aligned with the type of music I have been listening to recently.

This feedback suggests that the profile generation step worked well for Jerry’s case. The system successfully captured not only surface-level genre tags such as experimental rock and post-rock, but also the emotional, atmospheric, and instrumental qualities that are central to his listening experience.

---

## Music Generation

Using the personalized Suno prompt, multiple songs were generated for Jerry.

The final evaluation used the top two reranked songs selected by the fine-tuned CLAP model.

- Run ID: `20260505T160437Z-phase2_jerry-suno`
- Rerank Top-K: 2

---

## CLAP Reranking Results

| Rank | Track | CLAP Cosine Score |
|------|-------|-------------------|
| Top 1 | Track A / song1 | 0.7886 |
| Top 2 | Track B / song2 | 0.7685 |

---

## Generated Tracks

### Track A: song1

- CLAP cosine score: **0.7886**
- Local file:  
  `/Users/itsnotjerryh/Desktop/Github/Gen4Rec/src/eval/eval_phase_2/jerry/result/phase2_jerry/20260505T160437Z-phase2_jerry-suno/audio/call_04/phase2_jerry_rec_song_variant_02.mp3`

### Track B: song2

- CLAP cosine score: **0.7685**
- Local file:  
  `/Users/itsnotjerryh/Desktop/Github/Gen4Rec/src/eval/eval_phase_2/jerry/result/phase2_jerry/20260505T160437Z-phase2_jerry-suno/audio/call_05/phase2_jerry_rec_song_variant_01.mp3`

---

## Reranking vs Human Preference

- **Pipeline-selected track via CLAP reranking:** Track A / song1  
- **User-selected favorite:** Track A / First Song  

The CLAP reranking result is consistent with Jerry’s human preference in this case.

Track A received the highest CLAP cosine score and was also selected by Jerry as the preferred song. This suggests that, for this user, the fine-tuned CLAP reranking captured preference alignment reasonably well, especially for genre-specific and texture-driven music.

---

## Human Evaluation Results

| Track | Preference Match | Sound Quality | Creativity |
|------|------------------|--------------|-----------|
| Track A / song1 | 5 | 3 | 5 |
| Track B / song2 | 4 | 3 | 4 |

---

## User Qualitative Feedback

Jerry preferred the first generated song.

The first song received higher scores across all three evaluation dimensions. It achieved a perfect **5/5** in both preference match and creativity, suggesting that it was both closely aligned with Jerry’s taste and distinct enough to feel novel. The second song also matched his taste reasonably well, but it was slightly weaker in both preference alignment and creativity.

This response indicates that the personalized prompt was effective at generating songs in Jerry’s preferred direction, especially around experimental rock, post-rock, emotional atmosphere, and instrumental texture. However, both songs received moderate sound quality scores, suggesting that production quality remains an area for improvement.

---

## Key Findings

### Strengths

- The generated profile received a high **9/10** accuracy rating
- The system accurately captured Jerry’s genre preferences, especially experimental rock and post-rock
- The profile successfully reflected Jerry’s emotional connection to this type of music
- Instrumental descriptions were considered accurate and helpful
- The generated prompt captured important musical traits such as ambient texture, industrial grit, post-rock build, and mixed vocal/instrumental structure
- CLAP-based reranking aligned with Jerry’s human preference
- Track A achieved perfect scores in both preference match and creativity
- Both generated songs were directionally aligned with Jerry’s taste, with preference match scores of 5 and 4

---

### Limitations

- Both tracks received moderate sound quality scores of **3/5**
- While the songs matched Jerry’s taste and showed creativity, the audio coherence or production quality may still need improvement
- Some artists mentioned in the profile were unfamiliar to the user, although the profile summary still helped explain why they might be relevant
- The system may need stronger control over production quality when generating experimental or texture-heavy music
- For users with mixed instrumental and vocal preferences, future evaluation could ask more detailed questions about arrangement, build-up, atmosphere, and vocal balance

---

## Notes

- Jerry’s user embedding was constructed from 10 recent songs
- The input listening history had a balanced instrumental/vocal structure, with an **instrumental ratio of 0.5** and a **vocal/language ratio of 0.5**
- The average tempo of the input songs was around **106 BPM**, supporting the mid-tempo prompt direction
- The generated prompt focused on experimental post-rock, industrial rock, ambient textures, freak-folk details, mixed vocals, and cinematic builds
- The final recommendation selected by CLAP reranking was also the user’s preferred song
- This case shows that the personalized pipeline can capture niche genre preferences and emotionally specific listening patterns, especially for users whose taste depends on texture, atmosphere, and instrumental arrangement
