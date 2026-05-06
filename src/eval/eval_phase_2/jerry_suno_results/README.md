# User Study 5 – Personalized Music Generation Evaluation

## Overview

This user study evaluates the effectiveness of our personalized music generation pipeline using a real user (Jerry). The evaluation focuses on whether the system can generate a user profile that accurately reflects Jerry’s listening preferences, and whether the generated songs align with his human judgment.

Since only user feedback and human evaluation results are currently available, this README focuses on profile accuracy, qualitative feedback, and song-level evaluation. Additional pipeline details, generated prompt, CLAP reranking scores, or track URLs can be added later.

---

## Evaluation Metrics

We evaluate generated music along three dimensions:

- **Preference Match (1–5)**: How well the song aligns with the user's taste  
- **Sound Quality (1–5)**: Audio coherence and production quality  
- **Creativity (1–5)**: Novelty and distinctiveness  

We also collect qualitative feedback on:

- Accuracy of the generated user profile
- Whether the profile captures genre-level and emotion-level preferences
- Whether the generated songs match the user’s actual listening preference

---

## User Feedback on Profile

- Profile Accuracy: **9/10**

User explanation:

> The system is very accurate in analyzing the style of my songs, especially in capturing genre tags. The music I listen to is indeed experimental rock and post-rock. The profile also captures the emotional side of my listening experience very accurately, which matches the emotional connection I have with this type of music. It also includes instrumental descriptions that accurately describe the characteristics of the music. Although the profile mentions many musicians that I may not necessarily know, it does a good job summarizing them in a way that helps me understand that these artists are likely aligned with the type of music I have been listening to recently.

This feedback suggests that the profile generation step worked well for Jerry’s case. The system successfully captured not only surface-level genre tags such as experimental rock and post-rock, but also the emotional and instrumental qualities that are important to the user’s listening experience.

---

## Music Generation

Using the personalized profile and prompt, two songs were generated for Jerry.

The user was asked to choose which generated song they preferred.

- **User-selected favorite:** Track A / First Song

---

## Human Evaluation Results

| Track | Preference Match | Sound Quality | Creativity |
|------|------------------|--------------|-----------|
| Track A / First Song | 5 | 3 | 5 |
| Track B / Second Song | 4 | 3 | 4 |

---

## User Preference Result

Jerry preferred the first generated song.

The first song received higher scores across all three evaluation dimensions:

- Preference Match: **5/5**
- Sound Quality: **3/5**
- Creativity: **5/5**

The second song also performed reasonably well, with a preference match score of **4/5** and creativity score of **4/5**, but it was slightly less aligned with Jerry’s taste compared with the first song.

---

## Key Findings

### Strengths

- The generated profile received a high **9/10** accuracy rating
- The system accurately captured Jerry’s genre preferences, especially experimental rock and post-rock
- The profile successfully reflected the emotional connection Jerry has with this type of music
- Instrumental descriptions were considered accurate and helpful
- Track A achieved perfect scores in both preference match and creativity
- Both generated songs were directionally aligned with Jerry’s taste, with preference match scores of 5 and 4

---

### Limitations

- Both tracks received moderate sound quality scores of **3/5**
- While the songs matched Jerry’s taste and showed creativity, the audio coherence or production quality may still need improvement
- Some artists mentioned in the profile were unfamiliar to the user, although the summary still helped explain why they might be relevant
- Additional information such as generated prompt, CLAP reranking score, track URL, and pipeline-selected track is not yet available

---

## Notes

- Jerry’s evaluation currently includes profile feedback and human evaluation results only
- The generated profile was considered highly accurate in terms of genre, emotion, and instrumental characteristics
- The first generated song was preferred by the user
- This case shows that the personalized pipeline can capture genre-specific and emotion-specific listening preferences well, even when sound quality remains an area for improvement
