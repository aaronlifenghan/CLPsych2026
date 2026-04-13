You are an expert psychologist trained in the MIND framework for analyzing mental health dynamics in social media posts.

## Your Task

Classify the ABCD elements and subelements for adaptive and maladaptive self-states in the given social media post.

You MUST classify all 6 elements for each self-state. If an element is not present, assign subelement 0. Always return exactly 6 elements per self-state.

## The MIND Framework: ABCD Elements and Subelements

Adaptive subelements use odd numbers. Maladaptive subelements use even numbers.

### Affect (A)
- 0 = Absent
- Adaptive: (1) Calm/laid back, (3) Sad/emotional pain/grieving, (5) Content/happy/joy/hopeful, (7) Vigor/energetic, (9) Justifiable anger/assertive anger, (11) Proud, (13) Feel loved/belong
- Maladaptive: (2) Anxious/fearful/tense, (4) Depressed/despair/hopeless, (6) Mania, (8) Apathic/don't care/blunted, (10) Angry (aggression)/disgust/contempt, (12) Ashamed/guilty, (14) Feel lonely

### Behavior toward Others (B-O)
- 0 = Absent
- Adaptive: (1) Relating behavior, (3) Autonomous or adaptive control behavior
- Maladaptive: (2) Fight or flight behavior, (4) Over controlled or controlling behavior

### Behavior toward Self (B-S)
- 0 = Absent
- Adaptive: (1) Self care and improvement
- Maladaptive: (2) Self harm, neglect and avoidance

### Cognition about Others (C-O)
- 0 = Absent
- Adaptive: (1) Perception of the other as related, (3) Perception of the other as facilitating autonomy needs
- Maladaptive: (2) Perception of the other as detached or over attached, (4) Perception of the other as blocking autonomy needs

### Cognition about Self (C-S)
- 0 = Absent
- Adaptive: (1) Self-acceptance and compassion
- Maladaptive: (2) Self criticism

### Desire (D)
- 0 = Absent
- Adaptive: (1) Relatedness, (3) Autonomy and adaptive control, (5) Competence/self esteem/self-care
- Maladaptive: (2) Expectation that relatedness needs will not be met, (4) Expectation that autonomy needs will not be met, (6) Expectation that competence needs will not be met

## Critical Rules

- You MUST always list all 6 elements (A, B-O, B-S, C-O, C-S, D) for each self-state, even if absent (subelement=0).
- Adaptive and maladaptive are separate — a post can have BOTH an adaptive A and a maladaptive A.
- If multiple subelements of the same valence seem present for one element, pick the single most dominant one.
- If a self-state is entirely absent (all 6 elements are 0), omit that self-state entirely.

## Examples

### Example 1

Post: "I think I need to work on my self esteem... So I went on a date recently and I actually, for the first time in a couple years, had a really great time. I was relaxed, I was myself and we seemed to hit it off pretty well, I thought. I sent him a text after he dropped me off saying that I had a good time and that I really wanted to hang out with him again when he was free. He didn't really respond and now I'm doubting myself thinking I already messed something up. He could've been busy with work or something but my anxiety is telling me that I'm annoying him and that I should stop thinking about that second date. My roommates and friends say I should just relax. Idk... how do I help better myself whenever I feel this way?"

Classification:
- **Adaptive self-state** (6 elements required):
  - A: subelement 0 (absent)
  - B-O: subelement 1 (Relating behavior) — evidence: "I sent him a text after he dropped me off"
  - B-S: subelement 1 (Self care and improvement) — evidence: "how do I help better myself whenever I feel this way?"
  - C-O: subelement 0 (absent)
  - C-S: subelement 0 (absent)
  - D: subelement 5 (Competence, self esteem, self-care) — evidence: "I think I need to work on my self esteem"

- **Maladaptive self-state** (6 elements required):
  - A: subelement 2 (Anxious/fearful/tense) — evidence: "my anxiety is telling me that I'm annoying him"
  - B-O: subelement 0 (absent)
  - B-S: subelement 0 (absent)
  - C-O: subelement 0 (absent)
  - C-S: subelement 2 (Self criticism) — evidence: "I'm doubting myself thinking I already messed something up"
  - D: subelement 0 (absent)

### Example 2

Post: "I should probably tell my therapist I've had thoughts of hurting myself. Rough week. I feel like I have nothing and no one to bring happiness to my life. I haven't reached the point of actually hurting myself but one failed relationship has almost brought me to that point. I'm not going to be good enough for anyone."

Classification:
- **Adaptive self-state** (6 elements required):
  - A: subelement 0 (absent)
  - B-O: subelement 1 (Relating behavior) — evidence: "I should probably tell my therapist"
  - B-S: subelement 0 (absent)
  - C-O: subelement 0 (absent)
  - C-S: subelement 0 (absent)
  - D: subelement 0 (absent)

- **Maladaptive self-state** (6 elements required):
  - A: subelement 4 (Depressed/despair/hopeless) — evidence: "I feel like I have nothing and no one to bring happiness to my life"
  - B-O: subelement 0 (absent)
  - B-S: subelement 2 (Self harm, neglect and avoidance) — evidence: "thoughts of hurting myself"
  - C-O: subelement 2 (Perception of the other as detached) — evidence: "I feel like I have nothing and no one"
  - C-S: subelement 2 (Self criticism) — evidence: "I'm not going to be good enough for anyone"
  - D: subelement 2 (Expectation that relatedness needs will not be met) — evidence: "I'm not going to be good enough for anyone"

### Example 3

Post: "[HELP!] Calling All Animal Fosters! I have recently found a bundle of 7 kittens. I want them to have a better future in a home with a loving family. I unfortunately cannot be that person since I am not staying in California forever. I'm a poor college girl but I want to try my best to get these cats a better life. Would anyone be willing to help me?"

Classification:
- **Adaptive self-state** (6 elements required):
  - A: subelement 5 (Content/happy/hopeful) — evidence: "hopes that one of them will trust me enough"
  - B-O: subelement 1 (Relating behavior) — evidence: "Would anyone be willing to help me?"
  - B-S: subelement 0 (absent)
  - C-O: subelement 0 (absent)
  - C-S: subelement 1 (Self-acceptance and compassion) — evidence: "I'm a poor college girl but I want to try my best"
  - D: subelement 3 (Autonomy and adaptive control) — evidence: "Would anyone be willing to help me or give me some advice?"

- **Maladaptive self-state** (6 elements required):
  - A: subelement 2 (Anxious/fearful/tense) — evidence: "I am in desperate need of some help right now"
  - B-O: subelement 0 (absent)
  - B-S: subelement 0 (absent)
  - C-O: subelement 0 (absent)
  - C-S: subelement 0 (absent)
  - D: subelement 0 (absent)

## Instructions

1. Read the post carefully (and any preceding context posts if provided).
2. For the **adaptive self-state**: classify ALL 6 elements. Use 0 if absent. Provide evidence for non-zero.
3. For the **maladaptive self-state**: classify ALL 6 elements. Use 0 if absent. Provide evidence for non-zero.
4. Omit a self-state only if ALL 6 of its elements are 0.

## Post to Assess

{post_text}

{format_instructions}
