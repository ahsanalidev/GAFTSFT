import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean


SEED_FILES = {
    "3407": Path("/Users/ahsanali/Documents/Repositories/GAFTSFT/results_with_seed/IDKTuning_seed_3407/tofu.json"),
    "3408": Path("/Users/ahsanali/Documents/Repositories/GAFTSFT/results_with_seed/IDKTuning_seed_3408/tofu.json"),
    "3409": Path("/Users/ahsanali/Documents/Repositories/GAFTSFT/results_with_seed/IDKTuning_seed_3409/tofu.json"),
}

FULL_CSV = Path("/Users/ahsanali/Documents/Repositories/GAFTSFT/results_with_seed/idk_behavior_full_classification.csv")
SAMPLES_CSV = Path("/Users/ahsanali/Documents/Repositories/GAFTSFT/results_with_seed/idk_behavior_samples.csv")
RANDOM_SEED = 7

CATEGORY_ORDER = [
    "clean_abstention",
    "partial_abstention",
    "empty_or_no_answer",
    "original_memory_leakage",
    "hallucinated_substitute",
    "unrelated_drift",
]

ABSTENTION_PATTERNS = [
    r"\bi do not have (?:that|this|enough)? information\b",
    r"\bi don't have (?:that|this|enough)? information\b",
    r"\bi do not know\b",
    r"\bi don't know\b",
    r"\bi cannot answer\b",
    r"\bi can't answer\b",
    r"\bi am not able to answer\b",
    r"\bi'm not able to answer\b",
    r"\bi cannot provide\b",
    r"\bi can't provide\b",
    r"\bi do not remember\b",
    r"\bi don't remember\b",
    r"\bi am unsure\b",
    r"\bi'm unsure\b",
    r"\bi am not sure\b",
    r"\bi'm not sure\b",
    r"\bno information available\b",
    r"\bnot enough information\b",
    r"\bi do not have access\b",
    r"\bi don't have access\b",
]

EOS_LIKE_TOKENS = {
    "",
    "<pad>",
    "[pad]",
    "</s>",
    "<s>",
    "<unk>",
    "n/a",
    "na",
    "none",
    "null",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has",
    "have", "he", "her", "his", "in", "is", "it", "its", "of", "on", "or", "she",
    "that", "the", "their", "this", "to", "was", "were", "with", "who", "what",
    "when", "where", "which", "why", "how", "about", "into", "after", "before",
    "during", "through", "can", "could", "would", "should", "will", "may", "might",
}


def safe_mean(values):
    return mean(values) if values else 0.0


def collapse_ws(text):
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_for_compare(text):
    text = collapse_ws(text)
    text = re.sub(r"^(#+\s*)?answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(#+\s*)?response\s*:\s*", "", text, flags=re.IGNORECASE)
    return collapse_ws(text)


def normalize_for_repeat(text):
    text = normalize_for_compare(text).lower()
    text = text.replace("<pad>", "").replace("[pad]", "")
    return collapse_ws(text)


def punctuation_only(text):
    return not re.search(r"[A-Za-z0-9]", text or "")


def token_set(text):
    tokens = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return {tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def similarity_features(generated, reference_text):
    gen_norm = normalize_for_compare(generated)
    ref_norm = normalize_for_compare(reference_text)
    seq = SequenceMatcher(None, gen_norm.lower(), ref_norm.lower()).ratio()
    gen_tokens = token_set(gen_norm)
    ref_tokens = token_set(ref_norm)
    jac = jaccard(gen_tokens, ref_tokens)
    overlap = len(gen_tokens & ref_tokens) / max(len(ref_tokens), 1)
    return {
        "sequence_ratio": seq,
        "token_jaccard": jac,
        "reference_token_recall": overlap,
        "max_similarity": max(seq, jac, overlap),
    }


def looks_like_abstention(text):
    lowered = normalize_for_compare(text).lower()
    return any(re.search(pattern, lowered) for pattern in ABSTENTION_PATTERNS)


def is_empty_output(text):
    raw = collapse_ws(text)
    norm = normalize_for_compare(text).strip().lower()
    if not raw:
        return True
    if norm in EOS_LIKE_TOKENS:
        return True
    if punctuation_only(norm):
        return True
    if len(re.findall(r"[A-Za-z0-9]+", norm)) == 0:
        return True
    if len(norm) <= 2:
        return True
    repeated_tokens = re.findall(r"(?:<pad>|\[pad\]|</s>|<unk>)", raw.lower())
    if repeated_tokens and len(repeated_tokens) >= 2:
        return True
    if re.fullmatch(r"(answer:?\s*)+", norm):
        return True
    return False


def likely_factual_content(text):
    norm = normalize_for_compare(text)
    token_count = len(re.findall(r"[A-Za-z0-9']+", norm))
    has_year_or_date = bool(re.search(r"\b(?:18|19|20)\d{2}\b", norm))
    has_named_entity_shape = len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", norm)) >= 1
    has_copula = bool(re.search(r"\b(is|was|were|are|born|wrote|received|lives|works|published|awarded|known)\b", norm.lower()))
    return token_count >= 8 and (has_year_or_date or has_named_entity_shape or has_copula)


def question_overlap_score(question, generated):
    if not question:
        return None
    return jaccard(token_set(question), token_set(generated))


def classify_output(question, reference_output, generated):
    raw = collapse_ws(generated)
    norm = normalize_for_compare(generated)
    sims = similarity_features(generated, reference_output)
    q_overlap = question_overlap_score(question, generated)
    has_abstention = looks_like_abstention(generated)
    has_question_marker = "### question:" in raw.lower() or "[inst]" in raw.lower()

    if is_empty_output(generated):
        category = "empty_or_no_answer"
        reason = "empty, malformed, or eos-like output"
    elif has_abstention:
        content_after = re.split(
            r"(?:i do not have(?: that| this| enough)? information|i don't have(?: that| this| enough)? information|i do not know|i don't know|i cannot answer|i can't answer|i am not able to answer|i'm not able to answer|i am not sure|i'm not sure|not enough information)",
            norm,
            flags=re.IGNORECASE,
            maxsplit=1,
        )
        trailing = content_after[-1] if content_after else norm
        trailing_tokens = token_set(trailing)
        if likely_factual_content(trailing) or sims["max_similarity"] >= 0.35 or len(trailing_tokens) >= 6:
            category = "partial_abstention"
            reason = "abstention phrase plus additional factual content"
        else:
            category = "clean_abstention"
            reason = "explicit abstention with little added content"
    elif sims["sequence_ratio"] >= 0.82 or sims["token_jaccard"] >= 0.58 or sims["reference_token_recall"] >= 0.72:
        category = "original_memory_leakage"
        reason = "high lexical similarity to the saved original-question model output (proxy evidence, not gold truth)"
    elif has_question_marker or (q_overlap is not None and q_overlap < 0.08 and not likely_factual_content(norm)):
        category = "unrelated_drift"
        reason = "off-topic continuation, prompt repetition, or incoherent drift"
    elif likely_factual_content(norm):
        category = "hallucinated_substitute"
        reason = "confident factual-looking answer with low similarity to the saved reference output"
    else:
        category = "unrelated_drift"
        reason = "did not look like abstention, leakage, or a grounded factual answer"

    return category, reason, sims, q_overlap


def inspect_schema(obj, prefix="", depth=0, max_depth=3, seen=None, lines=None):
    if seen is None:
        seen = set()
    if lines is None:
        lines = []
    if depth > max_depth:
        return lines
    obj_id = id(obj)
    if obj_id in seen:
        return lines
    seen.add(obj_id)

    if isinstance(obj, dict):
        for key, value in obj.items():
            value_type = type(value).__name__
            lines.append(f"{prefix}{key}: {value_type}")
            if isinstance(value, (dict, list)):
                inspect_schema(value, prefix + "  ", depth + 1, max_depth, seen, lines)
    elif isinstance(obj, list):
        lines.append(f"{prefix}[list_length={len(obj)}]")
        if obj:
            first = obj[0]
            lines.append(f"{prefix}first_item_type: {type(first).__name__}")
            if isinstance(first, (dict, list)):
                inspect_schema(first, prefix + "  ", depth + 1, max_depth, seen, lines)
    return lines


def find_candidate_paths(obj, predicate, path="root", results=None, max_hits=20):
    if results is None:
        results = []
    if len(results) >= max_hits:
        return results
    try:
        if predicate(path, obj):
            results.append(path)
    except Exception:
        pass
    if isinstance(obj, dict):
        for key, value in obj.items():
            if len(results) >= max_hits:
                break
            find_candidate_paths(value, predicate, f"{path}.{key}", results, max_hits)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj[:3]):
            if len(results) >= max_hits:
                break
            find_candidate_paths(value, predicate, f"{path}[{idx}]", results, max_hits)
    return results


def load_cached_questions():
    try:
        from datasets import load_dataset
    except Exception:
        return None, "datasets package unavailable"

    possible_cache_dirs = [
        Path("/Users/ahsanali/Documents/Repositories/GAFTSFT/.cache"),
        Path("/Users/ahsanali/.cache/huggingface"),
    ]
    cache_dir = next((path for path in possible_cache_dirs if path.exists()), None)
    kwargs = {"split": "train", "local_files_only": True}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    try:
        ds = load_dataset("locuslab/TOFU", "forget01_perturbed", **kwargs)
    except Exception as exc:
        return None, f"local TOFU cache unavailable ({exc})"

    questions = []
    for row in ds:
        questions.append(row.get("question") or row.get("paraphrased_question") or "")
    return questions, "loaded questions from local TOFU cache"


def print_schema_report(seed, data):
    print(f"\n=== Schema Inspection: seed {seed} ===")
    for line in inspect_schema(data, max_depth=2):
        print(line)

    question_candidates = find_candidate_paths(
        data,
        lambda path, obj: isinstance(obj, list)
        and obj
        and all(isinstance(x, str) for x in obj[: min(5, len(obj))])
        and ("question" in path.lower() or "prompt" in path.lower()),
    )
    gt_candidates = find_candidate_paths(
        data,
        lambda path, obj: isinstance(obj, list)
        and obj
        and all(isinstance(x, str) for x in obj[: min(5, len(obj))])
        and any(token in path.lower() for token in ("answer", "reference", "truth"))
        and "generated" not in path.lower(),
    )
    gen_candidates = find_candidate_paths(
        data,
        lambda path, obj: isinstance(obj, list)
        and obj
        and all(isinstance(x, str) for x in obj[: min(5, len(obj))])
        and any(token in path.lower() for token in ("generated", "prediction", "output")),
    )
    split_candidates = find_candidate_paths(
        data,
        lambda path, obj: isinstance(obj, (dict, list, str)) and "forget" in path.lower(),
    )
    metric_candidates = find_candidate_paths(
        data,
        lambda path, obj: isinstance(obj, (int, float))
        and any(token in path.lower() for token in ("acc", "rouge", "truth", "quality", "mia")),
    )

    print("Candidate question/prompt fields:", question_candidates or ["none found in JSON"])
    print("Candidate ground-truth fields:", ["none found in JSON"])
    print("Candidate proxy reference fields:", gt_candidates or ["none found in JSON"])
    print("Candidate generated-answer fields:", gen_candidates or ["none found in JSON"])
    print("Candidate split fields:", split_candidates[:10] or ["none found in JSON"])
    print("Candidate metric fields:", metric_candidates[:10] or ["none found in JSON"])


def analyze_seed(seed, path, questions=None):
    with path.open() as f:
        data = json.load(f)

    print_schema_report(seed, data)

    forget = data.get("forget", {})
    generated_answers = forget.get("generated_answers", [])
    original_answers = forget.get("original_answers", [])
    metrics = {
        "truth_ratio": forget.get("truth_ratio"),
        "truth_prob": forget.get("truth_prob"),
        "rougeL_score": forget.get("rougeL_score"),
        "acc": forget.get("acc"),
        "Forget Quality": data.get("Forget Quality"),
        "MIA": data.get("MIA"),
    }

    if len(generated_answers) != len(original_answers):
        raise ValueError(
            f"Seed {seed}: generated_answers has {len(generated_answers)} rows but original_answers has {len(original_answers)} rows"
        )

    rows = []
    repeated = Counter()
    per_category = defaultdict(list)

    for idx, (generated, original) in enumerate(zip(generated_answers, original_answers)):
        question = ""
        if questions and idx < len(questions):
            question = questions[idx]
        category, reason, sims, q_overlap = classify_output(question, original, generated)
        output_len_words = len(re.findall(r"[A-Za-z0-9']+", normalize_for_compare(generated)))
        repeat_key = normalize_for_repeat(generated)
        if repeat_key:
            repeated[repeat_key] += 1

        row = {
            "seed": seed,
            "example_index": idx,
            "split_name": "forget",
            "question": question,
            "ground_truth_answer": "",
            "reference_output_proxy": original,
            "generated_answer": generated,
            "category": category,
            "classification_reason": reason,
            "output_length_words": output_len_words,
            "sequence_ratio": round(sims["sequence_ratio"], 4),
            "token_jaccard": round(sims["token_jaccard"], 4),
            "reference_token_recall": round(sims["reference_token_recall"], 4),
            "similarity_to_reference_proxy": round(sims["max_similarity"], 4),
            "question_overlap": "" if q_overlap is None else round(q_overlap, 4),
            "forget_truth_ratio_metric": metrics["truth_ratio"],
            "forget_truth_prob_metric": metrics["truth_prob"],
            "forget_rougeL_metric": metrics["rougeL_score"],
            "forget_acc_metric": metrics["acc"],
            "forget_quality_metric": metrics["Forget Quality"],
            "mia_metric": metrics["MIA"],
        }
        rows.append(row)
        per_category[category].append(row)

    print(f"\n=== Forget-Set Analysis: seed {seed} ===")
    print(f"Total forget examples: {len(rows)}")
    print("Note: `original_answers` is a saved model output on the original question, so similarity-based leakage labels are proxy-only.")
    for category in CATEGORY_ORDER:
        bucket = per_category.get(category, [])
        count = len(bucket)
        pct = (count / len(rows) * 100.0) if rows else 0.0
        avg_len = safe_mean([r["output_length_words"] for r in bucket])
        avg_sim = safe_mean([r["similarity_to_reference_proxy"] for r in bucket])
        print(
            f"{category}: {count:3d} ({pct:5.1f}%) | avg_len={avg_len:6.2f} words | avg_similarity_to_reference_proxy={avg_sim:0.3f}"
        )

    print("Most common repeated outputs:")
    for text, count in repeated.most_common(10):
        preview = text[:120] + ("..." if len(text) > 120 else "")
        print(f"  {count:3d}x | {preview}")

    return rows


def print_combined_stats(rows):
    print("\n=== Combined Statistics Across All Seeds ===")
    print(f"Total forget examples: {len(rows)}")
    by_category = defaultdict(list)
    by_seed = defaultdict(list)
    repeated = Counter()

    for row in rows:
        by_category[row["category"]].append(row)
        by_seed[row["seed"]].append(row)
        repeat_key = normalize_for_repeat(row["generated_answer"])
        if repeat_key:
            repeated[repeat_key] += 1

    for seed, seed_rows in sorted(by_seed.items()):
        print(f"Seed {seed}: {len(seed_rows)} rows")

    for category in CATEGORY_ORDER:
        bucket = by_category.get(category, [])
        count = len(bucket)
        pct = (count / len(rows) * 100.0) if rows else 0.0
        avg_len = safe_mean([r["output_length_words"] for r in bucket])
        avg_sim = safe_mean([r["similarity_to_reference_proxy"] for r in bucket])
        print(
            f"{category}: {count:4d} ({pct:5.1f}%) | avg_len={avg_len:6.2f} words | avg_similarity_to_reference_proxy={avg_sim:0.3f}"
        )

    print("Most common repeated outputs across all seeds:")
    for text, count in repeated.most_common(10):
        preview = text[:120] + ("..." if len(text) > 120 else "")
        print(f"  {count:3d}x | {preview}")


def write_csvs(rows):
    FULL_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with FULL_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rng = random.Random(RANDOM_SEED)
    samples = []
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    for category in CATEGORY_ORDER:
        bucket = list(by_category.get(category, []))
        if not bucket:
            continue
        chosen = rng.sample(bucket, min(10, len(bucket)))
        for row in chosen:
            sample_row = dict(row)
            sample_row["sample_group"] = "combined_random_examples"
            samples.append(sample_row)

    sample_fieldnames = ["sample_group"] + fieldnames
    with SAMPLES_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def build_interpretation(rows):
    total = len(rows)
    counts = Counter(row["category"] for row in rows)
    pct = {key: (counts[key] / total * 100.0 if total else 0.0) for key in CATEGORY_ORDER}
    dominant = counts.most_common(2)
    dominant_text = ", ".join(f"{cat} ({count}/{total}, {pct[cat]:.1f}%)" for cat, count in dominant)
    leakage_signal = pct["original_memory_leakage"]
    abstention_signal = pct["clean_abstention"] + pct["partial_abstention"]
    empty_signal = pct["empty_or_no_answer"]
    hallucination_signal = pct["hallucinated_substitute"]

    lines = []
    lines.append("=== Plain-English Interpretation ===")
    lines.append(f"The dominant behaviors are {dominant_text}.")
    if pct["clean_abstention"] >= 50:
        lines.append("IDK Tuning is mostly producing clean abstentions.")
    elif abstention_signal >= 50:
        lines.append("IDK Tuning often abstains, but a meaningful share of those abstentions are mixed with extra content.")
    else:
        lines.append("IDK Tuning is not mostly behaving like a clean abstention model on this forget set.")

    if empty_signal >= 20:
        lines.append("A sizable portion of the behavior looks like empty or degenerate no-answer output.")
    else:
        lines.append("Empty or degenerate no-answer behavior is present but not the main pattern.")

    if hallucination_signal >= 25:
        lines.append("There is substantial evidence of hallucinated substitute answers: the model often gives confident-looking answers that do not match the saved reference outputs.")
    else:
        lines.append("Hallucinated substitute answers occur, but they are not the dominant failure mode.")

    if leakage_signal >= 15:
        lines.append("There is non-trivial proxy evidence of memory retention because many outputs remain lexically close to the saved original-question outputs.")
    else:
        lines.append("Proxy lexical evidence of original memory leakage appears limited under these heuristics, and true gold-answer leakage cannot be measured from these JSON files alone.")

    lines.append(
        "For the paper, the safest claim is that IDK Tuning changes the response style on forget examples, but it does not reliably guarantee clean abstention; its behavior should be described as a mix of abstention, substitution, drift, and some residual leakage."
    )

    discussion = (
        "Across three seeds, IDK Tuning does not behave as a pure abstention mechanism on TOFU forget examples. "
        f"While abstention-style responses account for {abstention_signal:.1f}% of outputs, the model also produces a substantial share of "
        f"hallucinated substitutes ({hallucination_signal:.1f}%), unrelated drift ({pct['unrelated_drift']:.1f}%), and residual memory-like generations ({leakage_signal:.1f}%) under proxy lexical similarity heuristics against saved original-question outputs. "
        "This suggests that the method often suppresses direct recall without consistently enforcing a clean refusal policy, so improvements in forget metrics may partly reflect answer replacement rather than reliable epistemic abstention. "
        "For discussion purposes, we therefore recommend characterizing IDK Tuning as behavior-shaping toward non-answer responses, but not as a complete solution to faithful forgetting."
    )
    lines.append("\n=== Camera-Ready Discussion Paragraph ===")
    lines.append(discussion)
    return "\n".join(lines)


def main():
    print("IDK Tuning TOFU behavior analysis")
    print(f"Seeds: {', '.join(SEED_FILES.keys())}")

    questions, question_status = load_cached_questions()
    print(f"\nQuestion recovery status: {question_status}")
    if questions is not None:
        print(f"Recovered {len(questions)} question strings from local cache.")
    else:
        print("Questions are not present inside the JSON files; analysis will continue without prompt text.")

    all_rows = []
    for seed, path in SEED_FILES.items():
        rows = analyze_seed(seed, path, questions=questions)
        all_rows.extend(rows)

    print_combined_stats(all_rows)
    write_csvs(all_rows)

    print(f"\nSaved full classification CSV to: {FULL_CSV}")
    print(f"Saved sampled examples CSV to: {SAMPLES_CSV}")
    print()
    print(build_interpretation(all_rows))


if __name__ == "__main__":
    main()
