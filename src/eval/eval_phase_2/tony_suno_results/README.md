# User Study 3 – Personalized Music Generation Evaluation

## Overview

This user study evaluates the effectiveness of our personalized music generation pipeline using a real user (Tony). The evaluation focuses on whether the system can capture Tony’s music taste from recent listening history, generate a meaningful user profile and Suno prompt, and produce personalized music that aligns with human preference.

Tony’s case is especially useful because the generated profile received a perfect user rating, suggesting that the pipeline successfully captured not only genre-level preferences, but also the emotional and stylistic core of the user’s listening behavior.

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

---

## Methodology

### Step 1: User Embedding Construction

- User ID: `phase2_Tony`
- Input: Tony’s 10 most recent songs
- Each song is encoded using the fine-tuned CLAP model
- The 10 song embeddings are used to form a **single Tony user embedding**

---

### Step 2: Listening History Summary

Tony’s recent listening history shows a strong preference for vocal-led, emotionally expressive music that blends rap, emo, hip hop, and electronic textures.

#### Top Genres

- rap
- emo
- hip hop
- electronic
- minimal
- techno
- electronica
- emo rap

#### Top Tags

- rap
- cloud rap
- emo
- hip hop
- electronic
- hip-hop
- minimal
- techno
- electronica
- post-emo

#### Representative Artists

- Lil Peep
- JPEGMAFIA
- BRENNAN SAVAGE
- Matthew Dear
- Wicca Phase Springs Eternal
- Bob Moses

#### Representative Tracks

| Artist | Track |
|--------|-------|
| BRENNAN SAVAGE | Bulletproof |
| Matthew Dear | Echo |
| Lil Peep | Problems |
| Wicca Phase Springs Eternal | I REACH OUT TO YOU IN SONG |
| Bob Moses | Winter's Song |

---

### Step 3: Audio & Mood Profile

Tony’s input songs are strongly vocal-focused.

| Attribute | Value |
|----------|-------|
| Dominant Mode | Vocal |
| Vocal / Language Ratio | 1.0 |
| Instrumental Ratio | 0.0 |
| Average Danceability | 0.621 |
| Average Energy | 0.600 |
| Average Valence | 0.280 |
| Average Tempo | 116.287 BPM |

#### Mood Summary

- energetic
- driving
- bittersweet
- reflective
- mid-tempo

This suggests that Tony’s taste is not simply high-energy rap or electronic music. Instead, it leans toward emotionally detailed, vocal-centered, and bittersweet tracks with controlled momentum.

---

## Profile & Prompt Generation

### Generated User Profile

> This listener leans into a moody blend of rap, emo, hip hop, and electronic music with a strong vocal focus. Their taste points toward confessional writing, bittersweet melodies, and a restless but controlled energy, often around mid-tempo grooves. They seem drawn to atmospheric beats, minimal to layered electronic textures, and emotionally raw delivery that sits between cloud rap haze and club-adjacent pulse. The listener might enjoy artists such as Lil Peep, JPEGMAFIA, BRENNAN SAVAGE, and Wicca Phase Springs Eternal, alongside more rhythmic electronic turns like Matthew Dear or Bob Moses. Overall, the aesthetic feels intimate, bruised, and driving rather than explosive.

### Generated Suno Prompt

> Vocal-led emo rap x minimal electronic track, 115 BPM. Male vocal, emotionally raw and intimate, with melodic half-sung verses and a hooky chorus. Sparse 808s, hazy synth pads, crisp drums, subtle techno pulse, melancholic atmosphere, driving but restrained, reflective and bittersweet. Clean modern production, spacious mix, dark club-adjacent tension. Avoid bright pop polish and aggressive trap maximalism.

### Style Keywords

- emo rap
- cloud rap
- minimal electronic
- bittersweet
- mid-tempo
- hazy synths
- raw vocals

---

## User Feedback on Profile & Prompt

- Profile + Prompt Accuracy: **10/10**

User explanation:

> I give it full marks because it does not just identify surface-level labels like “rap” or “hip hop.” Instead, it accurately captures the emotional core shared by my input songs: vocal-led, melancholic, bittersweet, mid-tempo, and emotionally raw. The generated prompt also blends the emo rap and grounded rap energy of XXXTENTACION / 21 Savage while preserving the melodic and emotional progression of Lukas Graham / Coldplay, so overall it matches my listening preferences very well.

This feedback suggests that the profile generation step successfully moved beyond simple genre recognition. It captured the emotional atmosphere, vocal emphasis, and stylistic balance that Tony cares about most.

---

## Music Generation

Using the personalized Suno prompt, we generated two songs for Tony.

Both songs were evaluated by the user based on preference match, sound quality, and creativity.

---

## Human Evaluation Results

| Track | Preference Match | Sound Quality | Creativity |
|------|------------------|--------------|-----------|
| Track A | 5 | 5 | 5 |
| Track B | 4 | 5 | 4 |

---

## User Qualitative Feedback

> I am not someone who is very sensitive to technical details in music, but I care a lot about the overall atmosphere and feeling. Both generated songs matched my taste immediately, and I liked them as soon as I heard them. I was honestly very surprised by how well they captured the mood I enjoy.

This response indicates that the personalized prompt was highly effective at capturing Tony’s desired atmosphere. Even though the user does not focus heavily on technical musical details, the generated songs succeeded in the aspects that mattered most to him: mood, emotional tone, and immediate preference alignment.

---

## Key Findings

### Strengths

- The generated profile received a perfect **10/10** accuracy rating from the user
- The system captured the emotional core of Tony’s taste, not just surface-level genres
- The personalized prompt successfully combined emo rap, cloud rap, minimal electronic, and bittersweet vocal-driven elements
- Both generated songs received high preference match scores
- Both tracks received perfect sound quality scores, suggesting strong audio coherence and production quality
- Track A received full marks across all three evaluation dimensions

---

### Limitations

- The evaluation is based on two generated tracks, so more candidates would be needed for stronger conclusions
- The user emphasized atmosphere and emotional feeling more than technical music structure, so future evaluation could include more detailed questions about melody, arrangement, vocals, and production
- The current result is highly positive, but additional A/B testing against a generic or random prompt would help better isolate the effect of personalization
- Since the profile references representative artists, future versions may need to ensure that artist names are framed as possible preference signals rather than direct listening history

---

## Notes

- Tony’s user embedding was constructed from 10 recent songs
- The input listening history was fully vocal-based, with no instrumental tracks
- The average tempo of the input songs was around **116 BPM**, supporting the generated mid-tempo prompt direction
- The generated prompt focused on vocal-led emo rap, minimal electronic textures, hazy synths, and bittersweet emotional delivery
- This case shows that the personalized pipeline can effectively capture mood-level and atmosphere-level preferences, which may be more important to users than exact genre matching
