import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.experiments.auto_s_att import _parse_and_suffix_tokens


SENTENCE_TEXT = "After the meeting, Emily and Lewis reviewed the agenda. Emily handed a folder to"

RAW_RESPONSE = """<think>
Let me analyze this IOI task systematically.

**Understanding the Task:**
- I need to predict which tokens the attention head (7, 9) will attend to most strongly from the final position
- The hypothesis describes this as an "IO-promotion filter" that suppresses the subject and promotes the indirect object
- Top_K = 2, so I need to identify the top 2 most attended tokens

**Analyzing the Sentence Structure:**
`ioi_0088: After the meeting, Emily and Lewis reviewed the agenda. Emily handed a folder to`

Let me identify the key entities:
- **Subject (S):** "Emily" - the entity performing the transfer action ("handed")
- **Indirect Object (IO):** "Lewis" - the recipient entity (the one who receives the folder)
- **Direct Object (DO):** "folder" - the object being transferred

The sentence structure is: "After the meeting, Emily and Lewis reviewed the agenda. Emily handed a folder to"

**Applying the Hypothesis:**
The hypothesis states that head (7, 9):
1. **Suppresses attention to the subject token** - This means it should NOT strongly attend to "Emily" (the subject performing the action)
2. **Promotes attention to the indirect object token** - This means it SHOULD strongly attend to "Lewis" (the recipient)
3. **May suppress temporal/structural markers** - Words like "After", "the", "and", punctuation, etc.

**Identifying Token Positions:**
- "Emily" appears twice: once in "Emily and Lewis reviewed" and once in "Emily handed"
- "Lewis" appears once: in "Emily and Lewis reviewed"
- The final token is "to" (the sentence ends with "to")

**Predicting Attention from Final Position:**
From the perspective of the final token "to", the head should:
- **Most strongly attend to:** "Lewis" (the indirect object) - because the hypothesis says this head promotes IO attention
- **Second most strongly attend to:** This is tricky. The hypothesis says it suppresses the subject, so "Emily" should NOT be highly attended. However, the hypothesis also mentions it may suppress "temporal/structural markers." 

Given that the head is an "IO-promotion filter" that works by suppressing the subject while promoting the IO, the second-highest attention might go to:
- Another instance of the IO ("Lewis" if it appears multiple times, but it only appears once)
- Or possibly a structural element that's being suppressed but still receives some attention
- Or the direct object "folder" as a secondary entity of interest

Actually, re-reading the hypothesis more carefully: it says the head "suppresses attention to the subject token (the entity performing the transfer action)" and "promotes attention to the indirect object token." This is a binary suppression-promotion mechanism.

The most logical interpretation is:
- **Top 1 (Most attended):** "Lewis" - the indirect object being promoted
- **Top 2 (Second most attended):** This could be "Emily" (the second instance, as the subject being suppressed but still present in the model's attention), or it could be another token

However, the hypothesis emphasizes that this is a **filter** that removes the subject as a competing candidate. If it's truly suppressing the subject, then the second-highest attention might go to:
- "folder" (the direct object)
- Or possibly "reviewed" or "handed" (action verbs)
- Or the first "Emily" token

Given the context and the hypothesis's emphasis on IO-promotion and subject-suppression, I believe:
- **Top 1:** "Lewis" (IO - being promoted)
- **Top 2:** "Emily" (S - being suppressed, but still a competing entity that the head is actively filtering)

The suppression mechanism likely means the head attends to the subject to suppress it, not to ignore it completely.
</think>

[REASONING]
The hypothesis describes head (7, 9) as an "IO-promotion filter" that operates through two mechanisms:
1. Suppressing attention to the subject token (Emily - the entity performing the transfer)
2. Promoting attention to the indirect object token (Lewis - the recipient entity)

In the sentence "After the meeting, Emily and Lewis reviewed the agenda. Emily handed a folder to", the key entities are:
- Subject: "Emily" (appears twice - in the first clause and as the agent of "handed")
- Indirect Object: "Lewis" (the recipient of the folder)
- Direct Object: "folder" (the object being transferred)

From the final token position "to", the head should most strongly attend to "Lewis" as the promoted indirect object. The second-highest attention should go to "Emily" (the subject), as the suppression mechanism likely requires the head to attend to the subject token in order to actively suppress it as a competing candidate. This suppression-promotion dynamic is the core mechanism described in the hypothesis.
</REASONING>

[PREDICTION]
ioi_0088: After the meeting, <<Lewis>> and [[Emily]] reviewed the agenda. Emily handed a folder to
</PREDICTION>"""


def extract_target_line(text: str, sid: str) -> str:
    target = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("`") and line.endswith("`"):
            line = line[1:-1].strip()
        if line.startswith(f"{sid}:"):
            target = line
            break
    if target is None:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            if line.startswith("`") and line.endswith("`"):
                line = line[1:-1].strip()
            if line.startswith(f"{sid}:"):
                target = line
                break
    return target or ""


def main() -> None:
    sid = "ioi_0088"
    marked_sentence = RAW_RESPONSE.strip()
    prediction_matches = re.findall(r"\[PREDICTION\]\s*(.*)", marked_sentence, re.DOTALL | re.IGNORECASE)
    if prediction_matches:
        marked_sentence = prediction_matches[-1].strip()
    target_line = extract_target_line(marked_sentence, sid)
    increase, decrease = _parse_and_suffix_tokens(SENTENCE_TEXT, target_line)
    print("target_line:", target_line)
    print("increase:", increase)
    print("decrease:", decrease)


if __name__ == "__main__":
    main()
