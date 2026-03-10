import asyncio
import sys
import json
import os
import numpy as np
import re
import random
from collections import defaultdict
import argparse
from typing import Dict

sys.path.append("/data31/private/wangziran/eap_auto/")
from api import OpenRouter


async def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run automated hypothesis generation and refinement for agr_gender middle heads."
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--typename", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--causal_dataset",
        type=str,
        required=True,
        help="Path to JSON dict with diff_vectors (sender->receiver pattern diffs).",
    )
    parser.add_argument(
        "--receiver_heads",
        type=str,
        default="",
        help="Comma-separated downstream heads (e.g., '10.7,11.10') for contextual prompts.",
    )
    parser.add_argument(
        "--receiver_descriptions_file",
        type=str,
        default="",
        help="Optional JSON file mapping receiver head strings to textual descriptions.",
    )
    parser.add_argument(
        "--target_heads",
        type=str,
        default="",
        help="Optional comma-separated target heads (C) for A->B->C prompting.",
    )
    parser.add_argument(
        "--target_descriptions_file",
        type=str,
        default="",
        help="Optional JSON file mapping target head strings to textual descriptions.",
    )
    parser.add_argument(
        "--use-abc",
        action="store_true",
        help="Use diff_vectors_abc (A->B->C) if present instead of diff_vectors.",
    )
    return parser.parse_args()


def split_dataset(full_dataset, validation_split=0.2, seed=42):
    print(f"将数据集划分为 {1-validation_split:.0%} 训练集和 {validation_split:.0%} 验证集...")
    random.seed(seed)
    sentence_ids = list(full_dataset.keys())
    random.shuffle(sentence_ids)
    split_index = int(len(sentence_ids) * (1 - validation_split))
    train_ids = sentence_ids[:split_index]
    validation_ids = sentence_ids[split_index:]
    train_dataset = {sid: full_dataset[sid] for sid in train_ids}
    validation_dataset = {sid: full_dataset[sid] for sid in validation_ids}
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(validation_dataset)}")
    return train_dataset, validation_dataset


def initialize_openrouter(model: str = "claude-sonnet-4-20250514-thinking"):
    api_key = "sk-Z3pwy4dD8WY2XZlbzch66NP5hQIoFKeU7KvI2XD8bQSyFVGO"
    return OpenRouter(model=model, api_key=api_key)


def normalize_token(token):
    return token.strip().lower()


def random_sample_sentences_from_causal_dataset(causal_data, batch_size=5):
    if not causal_data:
        return []
    sentence_ids = list(causal_data.keys())
    if len(sentence_ids) <= batch_size:
        return sentence_ids
    return random.sample(sentence_ids, batch_size)


def get_causal_effects_for_sampling(sampled_sentence_ids, causal_dataset, sender_head, top_k=1, use_abc=False):
    sender_prefix = f"{sender_head[0]}.{sender_head[1]}->"
    examples = []
    for sid in sampled_sentence_ids:
        data = causal_dataset.get(sid, {})
        tokens = data.get("tokens", [])
        sentence_text = data.get("sentence_text", "")
        if use_abc and isinstance(data.get("diff_vectors_abc"), dict):
            diffs = data.get("diff_vectors_abc", {})
        else:
            diffs = data.get("diff_vectors", {})
        all_vecs = []
        for path_key, vec in (diffs or {}).items():
            if isinstance(path_key, str) and path_key.startswith(sender_prefix) and isinstance(vec, list) and vec:
                all_vecs.append(np.array(vec, dtype=float))
        if not all_vecs or not tokens:
            continue
        min_len = min(len(v) for v in all_vecs)
        tokens = tokens[:min_len]
        all_vecs = [v[:min_len] for v in all_vecs]
        # Build suffixes based on sentence token order to distinguish duplicates.
        words = re.findall(r"\w+|[^\w\s]", sentence_text)
        global_counts = defaultdict(int)
        for w in words:
            global_counts[normalize_token(w)] += 1
        running_counts = defaultdict(int)
        suffixed_tokens = []
        for tok in tokens:
            tok = tok.strip()
            if re.search(r"_\d+$", tok):
                suffixed_tokens.append(tok)
                continue
            norm = normalize_token(tok)
            running_counts[norm] += 1
            if global_counts.get(norm, 0) > 1:
                suffixed_tokens.append(f"{tok}_{running_counts[norm]}")
            else:
                suffixed_tokens.append(tok)
        avg_vec = np.mean(all_vecs, axis=0)
        changes = sorted(zip(suffixed_tokens, avg_vec), key=lambda x: x[1], reverse=True)
        inc = [t.strip() for t, _ in changes[:top_k]]
        dec = [t.strip() for t, _ in changes[-top_k:]]
        examples.append({"key": sid, "sentence": sentence_text, "increase_tokens": inc, "decrease_tokens": dec})
    return examples


async def predict_causal_effects_for_sentences(open_router: OpenRouter, hypothesis: str, examples, sender_head, top_k, output_dir):
    results = []
    max_format_retries = 5
    token_pat = r"(?:\w+|[^\w\s])"

    def _extract_marked_line_strict(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in reversed(lines):
            has_inc = re.search(rf"<<[^>]*{token_pat}[^>]*>>", ln) is not None
            has_dec = re.search(rf"\[\[[^\]]*{token_pat}[^\]]*\]\]", ln) is not None
            if has_inc or has_dec:
                return ln
        return ""

    def _extract_token_from_marker(chunk: str) -> str:
        m = re.search(r"\w+|[^\w\s]", chunk)
        return (m.group(0).strip() if m else "")

    def _validate_and_extract(text: str, inc_candidates: list[str], dec_candidates: list[str]):
        marked_line = _extract_marked_line_strict(text)
        if not marked_line:
            return None, "no marked line found"
        # strict: must be single line (no extra text)
        if text.strip() != marked_line:
            return None, "extra text (must return only the marked sentence line)"

        inc_chunks = re.findall(r"<<(.*?)>>", marked_line)
        dec_chunks = re.findall(r"\[\[(.*?)\]\]", marked_line)
        if len(inc_chunks) != top_k or len(dec_chunks) != top_k:
            return None, f"wrong marker counts: inc={len(inc_chunks)} dec={len(dec_chunks)} expected={top_k}"
        inc_tok = _extract_token_from_marker(inc_chunks[0]) if inc_chunks else ""
        dec_tok = _extract_token_from_marker(dec_chunks[0]) if dec_chunks else ""
        if not inc_tok or not dec_tok:
            return None, "empty token inside marker"

        inc_norm = normalize_token(inc_tok)
        dec_norm = normalize_token(dec_tok)
        inc_ok = inc_norm in {normalize_token(t) for t in inc_candidates}
        dec_ok = dec_norm in {normalize_token(t) for t in dec_candidates}
        if not inc_ok or not dec_ok:
            return None, f"token not in candidates: inc='{inc_tok}' dec='{dec_tok}'"

        # Enforce suffixed tokens if candidate is suffixed
        if any(re.search(r"_\\d+$", t) for t in inc_candidates) and not re.search(r"_\\d+$", inc_tok):
            return None, "increase token missing suffix (_1/_2)"
        if any(re.search(r"_\\d+$", t) for t in dec_candidates) and not re.search(r"_\\d+$", dec_tok):
            return None, "decrease token missing suffix (_1/_2)"

        return marked_line, None

    system_prompt = (
        "You are helping interpret a transformer attention head on a gender pronoun prediction task.\n"
        "The model sees a prompt ending right before a pronoun position, and must predict the correct pronoun (he/she).\n"
        "We ran a path-patching style intervention: we corrupt the sender head and observe how attention patterns of downstream heads change.\n\n"
        "Interpretation rule:\n"
        "- If attention to a token INCREASES after corrupting the sender head, the sender head normally SUPPRESSES attention to that token.\n"
        "- If attention to a token DECREASES, the sender head normally PROMOTES attention to that token.\n\n"
        "Output format (strict):\n"
        "- Return ONLY ONE LINE: the original sentence with markers inserted.\n"
        "- Include exactly one <<...>> token (INCREASE) and exactly one [[...]] token (DECREASE).\n"
        "- Do not add any explanations, bullet points, or extra lines.\n"
        "- Only use tokens from the provided candidate lists.\n"
        "- If the candidate token includes a suffix like _1/_2, you must output the suffix.\n"
        "- Do not mark any token with both types.\n"
    )

    for ex in examples:
        sent = ex["sentence"]
        inc_candidates = ", ".join(ex.get("increase_tokens", []))
        dec_candidates = ", ".join(ex.get("decrease_tokens", []))
        base_user_prompt = (
            f"Sender head: {sender_head[0]}.{sender_head[1]}\n"
            f"Hypothesis: {hypothesis}\n\n"
            f"Sentence: {sent}\n\n"
            f"INCREASE candidates: {inc_candidates}\n"
            f"DECREASE candidates: {dec_candidates}\n\n"
            "Return ONLY ONE LINE: the sentence with exactly one <<INCREASE>> token and one [[DECREASE]] token."
        )

        last_error = None
        for attempt in range(1, max_format_retries + 1):
            user_prompt = base_user_prompt
            if last_error:
                user_prompt = (
                    base_user_prompt
                    + "\n\nYour previous output was invalid because: "
                    + last_error
                    + "\nFix it. Output ONLY the single marked sentence line."
                )
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            resp = await open_router.generate(messages=messages, output_dir=output_dir)
            raw_text = (resp.text or "").strip()
            marked_line, err = _validate_and_extract(
                raw_text, ex.get("increase_tokens", []), ex.get("decrease_tokens", [])
            )
            if err is None and marked_line:
                results.append({"key": ex["key"], "marked_sentence": marked_line})
                break
            last_error = err or "unknown formatting error"
            if attempt == max_format_retries:
                # fall back to raw text; evaluation will likely fail, but we keep the trace
                results.append({"key": ex["key"], "marked_sentence": raw_text})
    return results


def _get_suffixed_word_map(original_sentence_text: str):
    words = re.findall(r"\w+|[^\w\s]", original_sentence_text)
    global_counts = defaultdict(int)
    for w in words:
        global_counts[normalize_token(w)] += 1
    running_counts = defaultdict(int)
    suffixed_words_map = {}
    for i, word in enumerate(words):
        norm_word = normalize_token(word)
        running_counts[norm_word] += 1
        if global_counts[norm_word] > 1:
            suffixed_word = f"{word.strip()}_{running_counts[norm_word]}"
        else:
            suffixed_word = word.strip()
        suffixed_words_map[i] = suffixed_word
    return words, suffixed_words_map


def _parse_and_suffix_tokens(original_sentence_text: str, marked_sentence: str, increase_markers=("<<", ">>"), decrease_markers=("[[", "]]")):
    words, suffixed_map = _get_suffixed_word_map(original_sentence_text)

    def pick_suffix(raw_token: str):
        norm = normalize_token(raw_token)
        for i, w in enumerate(words):
            if normalize_token(w) == norm:
                return suffixed_map[i]
        return raw_token.strip()

    increase_suffixed, decrease_suffixed = [], []
    inc_matches = re.findall(r"<<(.*?)>>", marked_sentence)
    dec_matches = re.findall(r"\[\[(.*?)\]\]", marked_sentence)

    for chunk in inc_matches:
        # pick the first token-like element inside the marker (word or punctuation)
        m = re.search(r"\w+|[^\w\s]", chunk)
        if m:
            increase_suffixed.append(pick_suffix(m.group(0)))
    for chunk in dec_matches:
        m = re.search(r"\w+|[^\w\s]", chunk)
        if m:
            decrease_suffixed.append(pick_suffix(m.group(0)))

    return increase_suffixed, decrease_suffixed


def _extract_marked_sentence(text: str) -> str:
    # Heuristic: pick the last line that contains a token inside a marker.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "<<" in ln or "[[" in ln:
            token_pat = r"(?:\w+|[^\w\s])"
            has_inc = re.search(rf"<<[^>]*{token_pat}[^>]*>>", ln) is not None
            has_dec = re.search(rf"\\[\\[[^\\]]*{token_pat}[^\\]]*\\]\\]", ln) is not None
            if has_inc or has_dec:
                return ln
    return text.strip()


def evaluate_causal_predictions(predictions, ground_truth_examples, sender_head, top_k=1):
    pred_map = {p["key"]: p for p in predictions}
    total_f1_inc, total_f1_dec, count = 0.0, 0.0, 0
    feedback_lines = []
    gt_map = {ex["key"]: ex for ex in ground_truth_examples}
    for key, gt in gt_map.items():
        pred = pred_map.get(key)
        if not pred:
            continue
        sentence = gt.get("sentence", "")
        marked = _extract_marked_sentence(pred["marked_sentence"])
        inc_pred, dec_pred = _parse_and_suffix_tokens(sentence, marked)
        inc_set = {normalize_token(t) for t in inc_pred[:top_k]}
        dec_set = {normalize_token(t) for t in dec_pred[:top_k]}
        real_inc = {normalize_token(t) for t in gt.get("increase_tokens", [])[:top_k]}
        real_dec = {normalize_token(t) for t in gt.get("decrease_tokens", [])[:top_k]}
        f1_inc = (2 * len(inc_set & real_inc)) / (len(inc_set) + len(real_inc)) if (len(inc_set) + len(real_inc)) else 0
        f1_dec = (2 * len(dec_set & real_dec)) / (len(dec_set) + len(real_dec)) if (len(dec_set) + len(real_dec)) else 0
        total_f1_inc += f1_inc
        total_f1_dec += f1_dec
        count += 1
        feedback_lines.append(f"- {key}: pred_inc={list(inc_set)} real_inc={list(real_inc)} pred_dec={list(dec_set)} real_dec={list(real_dec)}")
    avg = ((total_f1_inc / count) + (total_f1_dec / count)) / 2 if count else 0.0
    return avg, "\n".join(feedback_lines)


def load_head_descriptions(path: str) -> Dict[str, str]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text())
        if isinstance(payload, dict):
            return {str(k): str(v) for k, v in payload.items()}
    except Exception:
        pass
    return {}


def build_initial_prompt(
    sender_head,
    receiver_heads,
    receiver_desc: Dict[str, str],
    target_heads: str = "",
    target_desc: Dict[str, str] | None = None,
) -> str:
    parts = []
    parts.append(
        "Task: gender pronoun prediction (he/she).\n"
        "You are analyzing a SENDER attention head and its causal influence on DOWNSTREAM heads.\n\n"
        "Intervention setup:\n"
        "- We corrupt the sender head by patching its head output (z) from a corrupted prompt into the clean prompt.\n"
        "- We then observe how downstream heads' attention patterns change.\n\n"
        "Interpretation rule:\n"
        "- If attention to a token INCREASES after corrupting the sender head, the sender normally SUPPRESSES attention: it inhibits that token.\n"
        "- If attention to a token DECREASES, the sender normally PROMOTES attention: it boosts that token.\n\n"
        "Your job:\n"
        "- Infer what the sender head attends to and how that information causally shapes downstream behavior.\n"
        "- Produce a mechanistic hypothesis explaining the head's role in predicting the correct pronoun (he/she).\n"
    )
    if receiver_heads:
        parts.append(f"\nDownstream receiver heads: {receiver_heads}\n")
        if receiver_desc:
            parts.append("Known behaviors of downstream heads (use as contextual priors):\n")
            for h in receiver_heads.split(","):
                h = h.strip()
                if not h:
                    continue
                desc = receiver_desc.get(h)
                if desc:
                    parts.append(f"- Receiver {h}: {desc}\n")
    if target_heads:
        parts.append(f"\nTarget heads (C) for A->B->C: {target_heads}\n")
        if target_desc:
            parts.append("Known behaviors of target heads (use as contextual priors):\n")
            for h in target_heads.split(","):
                h = h.strip()
                if not h:
                    continue
                desc = target_desc.get(h)
                if desc:
                    parts.append(f"- Target {h}: {desc}\n")
    parts.append("\nFormat: Provide [HYPOTHESIS]: <one paragraph>.\n")
    return "".join(parts)


def extract_hypothesis_text(text: str) -> str:
    m = re.search(r"\[HYPOTHESIS\]:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


async def refine_hypothesis(open_router: OpenRouter, old: str, feedback: str, output_dir: str) -> str:
    system = (
        "You refine a mechanistic hypothesis about an attention head for a gender pronoun prediction task.\n\n"
        "Intervention setup:\n"
        "- Sender head is corrupted (z patched from corrupted prompt into clean prompt).\n"
        "- We observe downstream attention changes.\n\n"
        "Feedback meanings:\n"
        "- The feedback lists predicted vs. gold tokens that changed (increase/decrease).\n"
        "- Use it to adjust what the sender head suppresses or promotes.\n\n"
        "Output format:\n"
        "- Return exactly one paragraph starting with [HYPOTHESIS]:\n"
        "- Do NOT output JSON, lists, or analysis."
    )
    user = (
        f"Old hypothesis:\n{old}\n\n"
        f"Feedback (pred vs gold):\n{feedback}\n\n"
        "Write an improved hypothesis that better explains BOTH the attention changes and the role in he/she prediction."
    )
    resp = await open_router.generate(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], output_dir=output_dir)
    return extract_hypothesis_text(resp.text)


async def main():
    batch_size = 5
    max_iters = 5
    model_name = "claude-sonnet-4-20250514-thinking"

    args = await parse_arguments()
    layer, head = args.layer, args.head
    sender_head = (layer, head)
    output_dir = args.output_dir

    full_causal_dataset = json.loads(open(args.causal_dataset).read())
    train_dataset, validation_dataset = split_dataset(full_causal_dataset)

    receiver_desc = load_head_descriptions(args.receiver_descriptions_file)
    target_desc = load_head_descriptions(args.target_descriptions_file)
    open_router = initialize_openrouter(model=model_name)

    # initial hypothesis from aggregate cue
    receiver_heads = args.receiver_heads
    target_heads = args.target_heads
    initial_prompt = build_initial_prompt(sender_head, receiver_heads, receiver_desc, target_heads, target_desc)
    resp = await open_router.generate(messages=[{"role": "system", "content": "You are a helpful mechanistic interpretability assistant."}, {"role": "user", "content": initial_prompt}], output_dir=output_dir)
    hypothesis = extract_hypothesis_text(resp.text)

    all_validation = []
    for it in range(1, max_iters + 1):
        print(f"\n--- Iteration {it}/{max_iters} ---")
        sampled_ids = random_sample_sentences_from_causal_dataset(train_dataset, batch_size=batch_size)
        gt = get_causal_effects_for_sampling(
            sampled_ids, train_dataset, sender_head, top_k=1, use_abc=bool(args.use_abc)
        )
        preds = await predict_causal_effects_for_sentences(open_router, hypothesis, gt, sender_head, 1, output_dir)
        causal_f1, causal_feedback = evaluate_causal_predictions(preds, gt, sender_head, top_k=1)
        print(f"Iteration {it} Causal F1: {causal_f1:.2f}")
        debug_path = os.path.join(output_dir, f"debug_causal_iter_{it}.json")
        with open(debug_path, "w") as f:
            json.dump(
                {
                    "iteration": it,
                    "causal_f1": causal_f1,
                    "predictions": preds,
                    "ground_truth": gt,
                    "feedback": causal_feedback,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # validation
        val_ids = random_sample_sentences_from_causal_dataset(validation_dataset, batch_size=batch_size)
        gt_val = get_causal_effects_for_sampling(
            val_ids, validation_dataset, sender_head, top_k=1, use_abc=bool(args.use_abc)
        )
        preds_val = await predict_causal_effects_for_sentences(open_router, hypothesis, gt_val, sender_head, 1, output_dir)
        causal_val, _ = evaluate_causal_predictions(preds_val, gt_val, sender_head, top_k=1)

        all_validation.append(
            {
                "hypothesis": hypothesis,
                "validation_scores": {"direct_attention_f1": 1.0, "causal_f1": float(causal_val)},
            }
        )

        if it == max_iters:
            break
        hypothesis = await refine_hypothesis(open_router, hypothesis, causal_feedback, output_dir)

    final = {
        "head": f"{layer}.{head}",
        "typename": args.typename,
        "all_hypotheses_validation_results": all_validation,
    }
    out_path = os.path.join(output_dir, f"final_result_round_{args.rounds}.json")
    with open(out_path, "w") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved final result to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
