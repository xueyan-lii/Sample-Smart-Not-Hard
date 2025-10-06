import argparse
import json
import os
from typing import Dict, List, Optional, Tuple
from collections import Counter

from lm_eval.filters.extraction import RegexFilter
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
from gsm8k_eval_passk import norm
import matplotlib.pyplot as plt


def load_samples_json(json_path: str) -> Tuple[dict, List[dict]]:
    """Load the GSM8K-like samples JSON that gsm8k_eval_passk.py consumes.

    Expects structure: { "samples": { "gsm8k_fewshot": [ { "resps": [...] }, ... ] } }
    Returns the loaded root and the samples list reference to mutate.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples_container = data["samples"]
    samples = samples_container["gsm8k_fewshot"][:250]
    return data, samples


def read_low_conf_jsonl_sequence(jsonl_path: str) -> List[Optional[List[int]]]:
    """Read JSONL where each line is a list (or null) of low_conf_ranks for one response.

    Returns a list aligned with the flattened order of responses in json2.
    """
    ranks_sequence: List[Optional[List[int]]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON on line {line_num} of {jsonl_path}: {e}"
                ) from e
            # Expect each line to be either a list of ints or null/None
            if obj is None:
                ranks_sequence.append(None)
            elif isinstance(obj, list):
                ranks_sequence.append(obj)
            else:
                raise ValueError(
                    f"Line {line_num} of {jsonl_path} expected list or null, got: {type(obj)}"
                )
    return ranks_sequence


def assign_ranks_sequentially(samples: List[dict], ranks_sequence: List[Optional[List[int]]]) -> None:
    """Assign ranks to samples sequentially, assuming ranks_sequence matches the total number
    of responses across all samples in the same order.
    """
    # Compute total response slots
    total_slots = 0
    for sample in samples:
        resps = sample.get("resps", [])
        if isinstance(resps, list) and len(resps) > 0 and isinstance(resps[0], list):
            total_slots += len(resps[0])
        elif isinstance(resps, list):
            total_slots += len(resps)

    if total_slots != len(ranks_sequence):
        raise ValueError(
            f"Number of lines in ranks JSONL ({len(ranks_sequence)}) does not match total number of responses in JSON ({total_slots})."
        )

    idx = 0
    for sample in samples:
        resps_raw = sample.get("resps", [])
        if isinstance(resps_raw, list) and len(resps_raw) > 0 and isinstance(resps_raw[0], list):
            resps = resps_raw[0]
        elif isinstance(resps_raw, list):
            resps = resps_raw
        else:
            resps = []

        k = len(resps)
        sample_ranks = ranks_sequence[idx : idx + k]
        idx += k
        sample["ranks"] = sample_ranks


def compute_accuracy_by_rank_length(samples: List[dict]) -> Dict[int, Tuple[int, int]]:
    """Compute (correct_count, total_count) per rank length using gsm8k_eval_passk's logic.

    Only considers responses where a ranks list is available (i.e., not None).
    """
    rf = RegexFilter(regex_pattern=r"(?<=[Tt]he answer is )[^.\\]+")
    rank_len_to_counts: Dict[int, Tuple[int, int]] = {}

    for sample in samples:
        if "resps" not in sample or "target" not in sample or "ranks" not in sample:
            continue

        resps_raw = sample["resps"]
        resps = resps_raw[0]
        
        ranks_list = sample.get("ranks", [])

        # Extract answers using the same regex filter logic
        extracted = rf.apply([resps], [{}])[0]
        extracted_norm = [norm(a) for a in extracted]

        gt = norm(sample["target"])

        for ans_norm, ranks in zip(extracted_norm, ranks_list):
            if ranks is None:
                continue
            k = len(ranks)
            bin_k = k if 0 <= k < 20 else 20
            correct = 1 if ans_norm == gt else 0
            correct_count, total_count = rank_len_to_counts.get(bin_k, (0, 0))
            rank_len_to_counts[bin_k] = (correct_count + correct, total_count + 1)

    return rank_len_to_counts


def plot_accuracy(rank_len_to_counts: Dict[int, Tuple[int, int]], out_path: str) -> None:
    xs = list(range(0, len(rank_len_to_counts)))
    ys = []
    for k in xs:
        correct_count, total_count = rank_len_to_counts.get(k, (0, 0))
        ys.append((correct_count / total_count) if total_count else 0.0)
    print("Accuracies used for plotting (k: accuracy), where 10 represents 10+:")
    print({k: f"{v:.4g}" for k, v in zip(xs, ys)})

    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Number of low-confidence tokens")
    plt.ylabel("Average accuracy")
    plt.title("Accuracy vs. low-confidence token count (10=10+)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(xs)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")


def compute_k_sampled_accuracy(samples: List[dict]) -> Dict[int, Tuple[int, int]]:
    """Compute (correct_count, total_count) per sampled-k.

    - For each response, compute correctness using the same extraction/norm logic.
    - For each k in the ranks list of that response, attribute the response's correctness to bin k.
    - Bins: 1..9 and 10 represents 10+ (i.e., any k >= 10 goes to bin 10).
    """
    rf = RegexFilter(regex_pattern=r"(?<=[Tt]he answer is )[^.\\]+")
    k_to_counts: Dict[int, Tuple[int, int]] = {i: (0, 0) for i in range(1, 11)}  # 10 means 10+

    for sample in samples:
        if "resps" not in sample or "target" not in sample or "ranks" not in sample:
            continue

        resps_raw = sample["resps"]
        if isinstance(resps_raw, list) and len(resps_raw) > 0 and isinstance(resps_raw[0], list):
            resps = resps_raw[0]
        else:
            resps = resps_raw if isinstance(resps_raw, list) else []

        ranks_list = sample.get("ranks", [])
        if not isinstance(ranks_list, list):
            continue

        # Align lengths defensively
        min_len = min(len(resps), len(ranks_list))
        if min_len == 0:
            continue
        resps = resps[:min_len]
        ranks_list = ranks_list[:min_len]

        # Extract answers and normalize
        extracted = rf.apply([resps], [{}])[0]
        extracted_norm = [norm(a) for a in extracted]

        # Align again if needed
        min_len = min(len(extracted_norm), len(resps))
        if min_len == 0:
            continue
        resps = resps[:min_len]
        ranks_list = ranks_list[:min_len]
        extracted_norm = extracted_norm[:min_len]

        gt = norm(sample["target"])

        for ans_norm, ranks in zip(extracted_norm, ranks_list):
            if ranks is None:
                continue
            correct = 1 if ans_norm == gt else 0
            for r in ranks:
                if not isinstance(r, int):
                    continue
                bin_k = r if 1 <= r <= 9 else 10
                c, t = k_to_counts.get(bin_k, (0, 0))
                k_to_counts[bin_k] = (c + correct, t + 1)

    return k_to_counts


def count_rank_length_occurrences(samples: List[dict]) -> Dict[int, int]:
    """Count how many responses have a given number (0..10) of low-confidence tokens.

    Only counts entries where ranks is present (not None). Lengths > 10 are ignored.
    """
    counts: Dict[int, int] = {i: 0 for i in range(0, 11)}
    for sample in samples:
        ranks_list = sample.get("ranks")
        if not isinstance(ranks_list, list):
            continue
        for ranks in ranks_list:
            if ranks is None:
                continue
            klen = len(ranks)
            if 0 <= klen <= 10:
                counts[klen] += 1
    return counts


def compute_k_sampled_accuracy_len1(samples: List[dict]) -> Dict[int, Tuple[int, int]]:
    """Compute (correct_count, total_count) per sampled-k, restricted to responses where len(ranks) == 1.

    - Uses same extraction/norm logic as other computations.
    - Bins: 1..9 and 10 represents 10+ (i.e., any k >= 10 goes to bin 10).
    """
    rf = RegexFilter(regex_pattern=r"(?<=[Tt]he answer is )[^.\\]+")
    k_to_counts: Dict[int, Tuple[int, int]] = {i: (0, 0) for i in range(1, 11)}  # 10 means 10+

    for sample in samples:
        if "resps" not in sample or "target" not in sample or "ranks" not in sample:
            continue

        resps_raw = sample["resps"]
        if isinstance(resps_raw, list) and len(resps_raw) > 0 and isinstance(resps_raw[0], list):
            resps = resps_raw[0]
        else:
            resps = resps_raw if isinstance(resps_raw, list) else []

        ranks_list = sample.get("ranks", [])
        if not isinstance(ranks_list, list):
            continue

        min_len = min(len(resps), len(ranks_list))
        if min_len == 0:
            continue
        resps = resps[:min_len]
        ranks_list = ranks_list[:min_len]

        extracted = rf.apply([resps], [{}])[0]
        extracted_norm = [norm(a) for a in extracted]

        min_len = min(len(extracted_norm), len(resps))
        if min_len == 0:
            continue
        resps = resps[:min_len]
        ranks_list = ranks_list[:min_len]
        extracted_norm = extracted_norm[:min_len]

        gt = norm(sample["target"])

        for ans_norm, ranks in zip(extracted_norm, ranks_list):
            if ranks is None or len(ranks) != 1:
                continue
            k_val = ranks[0]
            if not isinstance(k_val, int):
                continue
            correct = 1 if ans_norm == gt else 0
            bin_k = k_val if 1 <= k_val <= 9 else 10
            c, t = k_to_counts.get(bin_k, (0, 0))
            k_to_counts[bin_k] = (c + correct, t + 1)

    return k_to_counts


def compute_mean_rank_binned_accuracy(samples: List[dict], bin_edges: Optional[List[int]] = None) -> Dict[str, Tuple[int, int]]:
    """Compute (correct_count, total_count) per mean-rank bin.

    - For each response, compute mean(ranks) and correctness using the same extraction/norm logic.
    - Bins are defined by ascending bin_edges (e.g., [5,10,15,20]) producing labels: "1-5", "5-10", ..., "20+".
    """
    rf = RegexFilter(regex_pattern=r"(?<=[Tt]he answer is )[^.\\]+")

    # Default edges
    if not bin_edges:
        bin_edges = [5, 10, 15, 20]

    # Build labels and initialize counts
    labels: List[str] = []
    lower = 1
    for edge in bin_edges:
        labels.append(f"{lower}-{edge}")
        lower = edge
    labels.append(f"{bin_edges[-1]}+")

    bin_to_counts: Dict[str, Tuple[int, int]] = {label: (0, 0) for label in labels}

    for sample in samples:
        if "resps" not in sample or "target" not in sample or "ranks" not in sample:
            continue

        resps_raw = sample["resps"]
        if isinstance(resps_raw, list) and len(resps_raw) > 0 and isinstance(resps_raw[0], list):
            resps = resps_raw[0]
        else:
            resps = resps_raw if isinstance(resps_raw, list) else []

        ranks_list = sample.get("ranks", [])
        if not isinstance(ranks_list, list):
            continue

        # Align lengths defensively
        min_len = min(len(resps), len(ranks_list))
        if min_len == 0:
            continue
        resps = resps[:min_len]
        ranks_list = ranks_list[:min_len]

        # Extract answers and normalize
        extracted = rf.apply([resps], [{}])[0]
        extracted_norm = [norm(a) for a in extracted]

        # Align again if needed
        min_len = min(len(extracted_norm), len(resps))
        if min_len == 0:
            continue
        resps = resps[:min_len]
        ranks_list = ranks_list[:min_len]
        extracted_norm = extracted_norm[:min_len]

        gt = norm(sample["target"])

        for ans_norm, ranks in zip(extracted_norm, ranks_list):
            if ranks is None or not isinstance(ranks, list) or len(ranks) == 0:
                continue
            # Compute mean rank
            int_ranks = [r for r in ranks if isinstance(r, int)]
            if not int_ranks:
                continue
            #mean_rank = sum(int_ranks) / len(int_ranks)
            mean_rank = max(int_ranks)

            # Determine bin based on edges
            label = None
            lower_bound = 1
            for edge in bin_edges:
                if lower_bound <= mean_rank <= edge:
                    label = f"{lower_bound}-{edge}"
                    break
                lower_bound = edge
            if label is None:
                label = f"{bin_edges[-1]}+"

            correct = 1 if ans_norm == gt else 0
            c, t = bin_to_counts.get(label, (0, 0))
            bin_to_counts[label] = (c + correct, t + 1)

    return bin_to_counts


def compute_conditioned_majority_voted_accuracy(samples: List[dict], mean_threshold: float = 2.0) -> Tuple[int, int]:
    """Compute majority-voted accuracy using only responses whose mean rank <= mean_threshold.

    For each question (sample):
      - Keep a response if its ranks list is present and the arithmetic mean of integer ranks <= threshold.
      - From the kept responses, extract answers, normalize, choose the most common as the majority vote.
      - Count it correct if it matches the normalized ground-truth target.

    Returns (correct_count, total_questions_considered), where total_questions_considered
    counts only questions with at least one kept response.
    """
    rf = RegexFilter(regex_pattern=r"(?<=[Tt]he answer is )[^.\\]+")

    maj_correct = 0
    n_considered = 0

    for sample in samples:
        if "resps" not in sample or "target" not in sample or "ranks" not in sample:
            continue

        resps_raw = sample["resps"]
        if isinstance(resps_raw, list) and len(resps_raw) > 0 and isinstance(resps_raw[0], list):
            resps = resps_raw[0]
        else:
            resps = resps_raw if isinstance(resps_raw, list) else []

        ranks_list = sample.get("ranks", [])
        if not isinstance(ranks_list, list) or not isinstance(resps, list):
            continue

        # Align defensively
        min_len = min(len(resps), len(ranks_list))
        if min_len == 0:
            continue

        filtered_resps: List[str] = []
        for i in range(min_len):
            ranks = ranks_list[i]
            if ranks is None or not isinstance(ranks, list) or len(ranks) == 0:
                continue
            int_ranks = [r for r in ranks if isinstance(r, int)]
            if not int_ranks:
                continue
            mean_rank = sum(int_ranks) / len(int_ranks)
            if mean_rank <= mean_threshold:
                filtered_resps.append(resps[i])

        if not filtered_resps:
            continue

        extracted = rf.apply([filtered_resps], [{}])[0]
        extracted_norm = [norm(a) for a in extracted]
        if not extracted_norm:
            continue

        c = Counter(extracted_norm)
        maj_ans, _ = c.most_common(1)[0]
        gt = norm(sample["target"])
        maj_correct += int(maj_ans == gt)
        n_considered += 1

    return maj_correct, n_considered


def main() -> None:
    json2_path = os.environ.get("SAMPLES_JSON", json2_path)
    json1_path = os.environ.get("RANKS_JSONL", json1_path)

    data, samples = load_samples_json(json2_path)
    ranks_sequence = read_low_conf_jsonl_sequence(json1_path)

    # New: total number of rank occurrences across all rows
    total_rank_occurrences = sum((len(r) for r in ranks_sequence if isinstance(r, list)), 0)
    print("Total number of low-confidence rank occurrences (sum of all rows' lengths):")
    print(total_rank_occurrences)

    assign_ranks_sequentially(samples, ranks_sequence)

    # Compute accuracy per rank length and plot
    rank_len_to_counts = compute_accuracy_by_rank_length(samples)
    print("Number of occurrences of low confidence tokens: (number correct, number of occurrences of this number of low confidence tokens)")
    print(rank_len_to_counts)
    out_dir = os.path.abspath(os.environ.get("OUT_DIR", "./outputs"))
    os.makedirs(out_dir, exist_ok=True)
    out_plot = os.path.join(out_dir, "accuracy_vs_low_conf_tokens_seq.png")
    plot_accuracy(rank_len_to_counts, out_plot)

    # Compute k-sampled accuracy and print
    k_to_counts = compute_k_sampled_accuracy(samples)
    print("Rank k sampled: (number correct, number of occurrences of rank k sampled)")
    print(k_to_counts)
    # Also print average accuracy per k (1..9 and 10 for 10+)
    k_to_accuracy = {k: ((c / t) if t else 0.0) for k, (c, t) in sorted(k_to_counts.items())}
    print("Rank k sampled: average accuracy of rank k sampled")
    print({k: f"{v:.4g}" for k, v in sorted(k_to_accuracy.items())})

    # Count rank length occurrences and print
    counts = count_rank_length_occurrences(samples)
    print(counts)

    # Compute per-k accuracy for rank length == 1
    k_to_counts_len1 = compute_k_sampled_accuracy_len1(samples)
    print("Rank k sampled (len 1): (number correct, number of occurrences of rank k sampled)")
    print(k_to_counts_len1)
    # Also print average accuracy per k (1..9 and 10 for 10+)
    k_to_accuracy_len1 = {k: ((c / t) if t else 0.0) for k, (c, t) in sorted(k_to_counts_len1.items())}
    print("Rank k sampled (len 1): average accuracy of rank k sampled")
    print({k: f"{v:.4g}" for k, v in sorted(k_to_accuracy_len1.items())})

    # Compute mean-rank binned accuracy and print
    mean_rank_bin_edges = [i * 2 for i in range(1, 11)]
    bin_to_counts = compute_mean_rank_binned_accuracy(samples, mean_rank_bin_edges)
    print("Mean-rank binned: (number correct, number of occurrences of mean-rank bin)")
    # Build ordered labels from edges
    bin_order = [f"{1}-{mean_rank_bin_edges[0]}"] + [
        f"{mean_rank_bin_edges[i-1]}-{mean_rank_bin_edges[i]}" for i in range(1, len(mean_rank_bin_edges))
    ] + [f"{mean_rank_bin_edges[-1]}+"]
    print({k: bin_to_counts.get(k, (0, 0)) for k in bin_order})
    # Also print average accuracy per bin (in fixed order)
    bin_to_accuracy = {
        k: ((bin_to_counts.get(k, (0, 0))[0] / bin_to_counts.get(k, (0, 0))[1]) if bin_to_counts.get(k, (0, 0))[1] else 0.0)
        for k in bin_order
    }
    print("Mean-rank binned: average accuracy of mean-rank bin")
    print({k: f"{v:.4g}" for k, v in bin_to_accuracy.items()})

    # Conditioned majority-voted accuracy (mean rank <= 2)
    
    cond_correct, cond_total = compute_conditioned_majority_voted_accuracy(samples, mean_threshold=3.0)
    print("Conditioned majority-voted accuracy (mean rank <= 2):")
    print({
        "correct": cond_correct,
        "total": cond_total,
        "accuracy": (f"{(cond_correct / cond_total):.4g}" if cond_total else "0"),
    })
    


if __name__ == "__main__":
    main() 