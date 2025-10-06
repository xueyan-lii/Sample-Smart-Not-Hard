import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from lm_eval.filters.extraction import RegexFilter
from gsm8k_eval_passk import norm, top_nonblank

def evaluate_json_file(json_path, source="torchtune", max_k=128):
    """
    Evaluate a single JSON file and return majority voted accuracy for k=1 to max_k
    """
    if source == "torchtune": 
        with open(json_path, "r") as f:
            data = json.load(f)
        samples = data["samples"]["gsm8k_fewshot_128"][:1313] #consider the only unique set
    elif source == "lm_eval":
        samples = []
        with open(json_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line.strip())
                samples.append({"resps":json_obj["resps"],"target":json_obj["target"]})
    
    ks = list(range(1, max_k + 1))
    rf = RegexFilter(regex_pattern=r"(?<=[Tt]he answer is )[^.\\]+")
    maj_correct = {k: 0 for k in ks}
    n_q = 0

    for s in samples:
        gt = norm(s["target"])
        resps = s["resps"][0]
        # extract answers once, then slice for ks
        extracted = rf.apply([resps], [{}])[0]
        extracted_norm = [norm(a) for a in extracted]

        # per-k accuracies
        for k in ks:
            sub = extracted_norm[:min(k, len(extracted_norm))]
            if not sub:
                continue
            c = Counter(sub)
            maj_ans, _ = top_nonblank(c)
            maj_correct[k] += int(maj_ans == gt)

        n_q += 1
    
    # Calculate accuracy for each k
    maj_acc = {k: (maj_correct[k] / n_q if n_q else 0.0) for k in ks}
    return maj_acc, n_q

def calculate_confidence_bin_accuracies(json_paths, source="torchtune", max_k=128):
    """
    Compute majority-voted accuracy curves per confidence bin without plotting.

    Args:
        json_paths: List of 10 JSON file paths corresponding to confidence bins
        source: Source format ("torchtune" or "lm_eval")
        max_k: Maximum number of samples to evaluate (default 128)

    Returns:
        A dict with keys:
            - "k_values": list of k from 1..max_k
            - "bins": list of dicts with keys {"label", "accuracies", "n_q", "path"}
    """
    bin_labels = [
        "(0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", "(0.4,0.5]",
        "(0.5,0.6]", "(0.6,0.7]", "(0.7,0.8]", "(0.8,0.9]", "(0.9,1.0]"
    ]

    k_values = list(range(1, max_k + 1))
    results = {"k_values": k_values, "bins": []}

    for i, json_path in enumerate(json_paths):
        print(f"Processing confidence bin {i+1}/10: {bin_labels[i]}")
        try:
            maj_acc, n_q = evaluate_json_file(json_path, source, max_k)
            accuracies = [maj_acc[k] for k in k_values]
            results["bins"].append({
                "label": bin_labels[i],
                "accuracies": accuracies,
                "n_q": n_q,
                "path": json_path,
            })
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            results["bins"].append({
                "label": bin_labels[i],
                "accuracies": [0.0 for _ in k_values],
                "n_q": 0,
                "path": json_path,
                "error": str(e),
            })

    return results

def plot_confidence_bins(json_paths, source="torchtune", max_k=128):
    """
    Plot majority voted accuracy vs number of samples for 10 confidence bins
    
    Args:
        json_paths: List of 10 JSON file paths corresponding to confidence bins
        source: Source format ("torchtune" or "lm_eval")
        max_k: Maximum number of samples to evaluate (default 128)
    """
    bin_data = calculate_confidence_bin_accuracies(json_paths, source, max_k)
    plot_confidence_bins_from_data(bin_data, max_k=max_k)

def plot_confidence_bins_from_data(bin_data, max_k=128):
    """
    Plot majority-voted accuracy vs number of samples using precomputed data.

    Args:
        bin_data: Dict returned by calculate_confidence_bin_accuracies
        max_k: Maximum number of samples to show on x-axis
    """
    # Colors for each line
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    plt.figure(figsize=(12, 8))

    k_values = bin_data.get("k_values", list(range(1, max_k + 1)))

    for i, bin_entry in enumerate(bin_data.get("bins", [])):
        label = bin_entry.get("label", f"bin_{i}")
        accuracies = bin_entry.get("accuracies", [])
        n_q = bin_entry.get("n_q", 0)

        plt.plot(
            k_values,
            accuracies,
            color=colors[i % len(colors)],
            label=f"{label} (n={n_q})",
            linewidth=2,
        )

    plt.xlabel('Number of Samples (k)', fontsize=12)
    plt.ylabel('Majority Voted Accuracy', fontsize=12)
    plt.title('Majority Voted Accuracy vs Number of Samples by Confidence Bin', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(1, max_k)
    plt.ylim(0, 1)

    plt.tight_layout()

    out_path = os.getenv('CONF_BIN_PLOT_PATH', 'confidence_bin_plot.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function - you need to provide the 10 JSON file paths here
    """
    # TODO: Replace these with your actual JSON file paths
    # The order should correspond to confidence bins: (0,0.1], (0.1,0.2], ..., (0.9,1.0]
    json_paths = [
    ]
    
    print("The order should correspond to confidence bins: (0,0.1], (0.1,0.2], ..., (0.9,1.0]")
    
    # Uncomment the line below once you've updated the paths
    plot_confidence_bins(json_paths, source="torchtune", max_k=128)

if __name__ == "__main__":
    main()
