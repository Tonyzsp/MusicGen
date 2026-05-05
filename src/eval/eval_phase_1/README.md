## Human Retrieval Evaluation: Base CLAP vs Fine-tuned CLAP

**Automation in this repo:** export blind clips with `scripts/run_phase1_eval.py` (default output `outputs/phase1_eval/`). Small text copies of `manifest.json` / `participant_instructions.txt` may live under `src/eval/eval_phase_1/` for reference; audio stays local.

We conducted two rounds of lightweight human preference evaluation to compare the retrieval quality of the base CLAP model and the fine-tuned CLAP model.

For each text prompt, we retrieved two audio clips using the base CLAP model and two audio clips using the fine-tuned CLAP model. The four clips were then shuffled and renamed with blind filenames, such as `p01_clip01.mp3`, `p01_clip02.mp3`, `p01_clip03.mp3`, and `p01_clip04.mp3`. Participants only saw the prompt text and the blinded clip names, without knowing which model retrieved each clip.

### Evaluation Prompts

We used 20 prompts covering both everyday listening scenarios and more prompt-sensitive cinematic/fantasy retrieval scenarios:

- p01 Rainy Night: slow emotional music for walking alone on a rainy night, gentle piano, quiet, lonely, and reflective
- p02 Morning Coffee: relaxing acoustic guitar music for drinking coffee in the morning, warm, calm, and comfortable
- p03 Workout Energy: fast high-energy music for working out, powerful, motivating, and intense
- p04 Beach Sunset: laid-back summer music for watching the beach sunset, relaxed, sunny, and peaceful
- p05 City Drive: cool pop or R&B music for driving through the city at night, stylish, smooth, and confident
- p06 Happy Party: fun dance music for a small party with friends, cheerful, catchy, and easy to move to
- p07 Study Focus: calm background music for studying, steady, relaxing, and not distracting
- p08 Romantic Dinner: smooth romantic dinner music with soft saxophone, warm, intimate, relaxing, and elegant
- p09 Chinese Fantasy Moonlight: cinematic Chinese fantasy music with elegant strings, flute, and a mysterious moonlit atmosphere
- p10 Orchestral Battle: dramatic orchestral battle music with rising tension, heavy percussion, and heroic fantasy energy
- p11 Playful Villain: playful villain-themed music with quirky rhythm, theatrical humor, and festive celebration mood
- p12 Bittersweet Romance: emotional piano and strings music with a gentle, bittersweet, romantic atmosphere
- p13 Nervous Chase: fast-paced music with chaotic footsteps, nervous rhythm, and suspenseful movement
- p14 Ancient Ceremony: ancient ceremonial music with grand drums, bells, and ritual-like atmosphere
- p15 Mirror World: dreamy mirror-world music with ethereal vocals, soft synths, and surreal fantasy mood
- p16 Epic Adventure: epic adventure soundtrack with sweeping strings, strong drums, and a sense of destiny
- p17 Moonlight Ballad: elegant night-time ballad with moonlight, soft melody, and sentimental emotion
- p18 Dark Fantasy: dark fantasy music with mysterious ambience, low strings, and magical tension
- p19 Fantasy New Year Dinner: lighthearted festive music for a fantasy New Year dinner scene, humorous but warm
- p20 Wind and Clouds: flowing cinematic instrumental music with wind, clouds, and emotional build-up

### Scoring Method

For each prompt, participants ranked the four clips from best match to worst match. We assigned scores based on the ranking position:

```text
best = 5
second = 3
third = 1
worst = 0
```

For each prompt, we summed the scores of the two clips retrieved by the fine-tuned CLAP model and the two clips retrieved by the base CLAP model. This produced one prompt-level comparison between the two models.

---

## Evaluation Round 1

Across 20 prompts, the total scores were:

```text
Fine-tuned CLAP = 109
Base CLAP = 71
```

The fine-tuned CLAP model achieved a higher overall score, suggesting that fine-tuning improved the model's ability to retrieve audio clips that better matched the input text prompts.

### Sign Test for Round 1

To test whether the fine-tuned CLAP model performs better than the base CLAP model, we conducted a one-sided sign test using prompt-level results.

At the prompt level, fine-tuned CLAP won 14 prompts and base CLAP won 6 prompts. There were no tied prompts, so the sign test used:

```text
n = 20, x = 14
```

The hypotheses were:

```text
H0: p = 0.5
H1: p > 0.5
```

where `p` is the probability that fine-tuned CLAP outperforms base CLAP on a prompt.

Under the null hypothesis:

```text
X ~ Binomial(20, 0.5)
```

The one-sided p-value is:

```text
P(X >= 14) = 0.0577
```

Since `0.0577 < 0.10`, we reject `H0` at the 10% significance level. This suggests that fine-tuned CLAP performs better than base CLAP in the first round of human retrieval evaluation. However, the result is not significant at the 5% level because `0.0577 > 0.05`.

---

## Evaluation Round 2: 

We've completed a second round of human retrieval evaluation using the same prompts, the same blind clip setup, and the same scoring rule.

For prompt `p08`, `clip01` and `clip02` are marked as equally matched. To keep the prompt-level total score consistent, we treated them as tied between second and third place, so each received the average score:

```text
(3 + 1) / 2 = 2
```

Based on Round 2 evaluation, the total scores were:

```text
Fine-tuned CLAP = 109
Base CLAP = 71
```

This again shows that the fine-tuned CLAP model achieved a higher overall score than the base CLAP model.

### Sign Test for Round 2

At the prompt level, fine-tuned CLAP won 13 prompts and base CLAP won 7 prompts. There were no tied prompt-level comparisons, so the sign test used:

```text
n = 20, x = 13
```

The hypotheses were:

```text
H0: p = 0.5
H1: p > 0.5
```

where `p` is the probability that fine-tuned CLAP outperforms base CLAP on a prompt.

Under the null hypothesis:

```text
X ~ Binomial(20, 0.5)
```

The one-sided p-value is:

```text
P(X >= 13) = 0.1316
```

Since `0.1316 > 0.10`, Round 2 evaluation alone does not reject `H0` at the 10% significance level. This means that Round 2's prompt-level sign test result is not statistically significant on its own. However, the direction still favors fine-tuned CLAP, since it achieved a higher total score and won more prompt-level comparisons.

---

## Combined Results Across Two Evaluation Rounds

Combining the two evaluation rounds, the total scores were:

```text
Fine-tuned CLAP = 218
Base CLAP = 142
```

The combined prompt-level results were:

```text
Fine-tuned CLAP wins = 27
Base CLAP wins = 13
n = 40, x = 27
```

Using a one-sided sign test:

```text
X ~ Binomial(40, 0.5)
```

The one-sided p-value is:

```text
P(X >= 27) = 0.0192
```

Since `0.0192 < 0.05`, the combined evaluation rejects `H0` at the 5% significance level. This provides stronger evidence that the fine-tuned CLAP model performs better than the base CLAP model in human retrieval evaluation.

Overall, across two rounds of human evaluation, the fine-tuned CLAP model consistently achieved higher total scores than the base CLAP model. The combined sign test also shows that the improvement is statistically significant at the 5% level, supporting the conclusion that fine-tuning improved CLAP's text-to-music retrieval performance.
