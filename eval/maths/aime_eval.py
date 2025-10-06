#print each question and last answer strings for manual eval
import json, sys
from collections import Counter
from typing import List, Tuple, Dict, Any

from aime_utils import (
    remove_boxed,
    last_boxed_only_string,
    strip_string,
    is_equiv
)

def process_result(response: str) -> str:
    indices = [pos for pos, char in enumerate(response) if char == "$"]
    if len(indices) <= 1:
        answer = response
    else:
        answer = response[indices[0] + 1 : indices[-1]]

    # Prefer boxed extraction if present (AIME often uses \boxed{})
    boxed_answer = last_boxed_only_string(response)
    if boxed_answer is not None:
        try:
            boxed_content = remove_boxed(boxed_answer)
            if boxed_content is not None:
                answer = boxed_content
        except (AssertionError, IndexError):
            pass
    return answer


def top_nonblank(counter: Counter) -> Tuple[str, int]:
    return counter.most_common(1)[0] if counter else ("", 0)



def main(paths: List[str]) -> None:
    ks = [1, 4, 8, 16, 32]

    maj_correct = {k: 0 for k in ks}
    ora_correct = {k: 0 for k in ks}
    n_q = 0

    maj_prop_sum = 0.0
    uniq_cnt_sum = 0
    correct_total = 0
    total_resps = 0

    samples: List[dict] = []

    # Merge responses per-question across all provided files
    question_to_data: Dict[str, Dict[str, Any]] = {}

    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        samples_dict = data.get("samples", {})
        for _, file_samples in samples_dict.items():
            for s in file_samples:
                doc = s.get("doc", {})
                key = json.dumps(doc, sort_keys=True, ensure_ascii=False)

                gt_value = ""
                for gt_key in ["answer", "Answer", "final_answer", "FinalAnswer", "target", "Target"]:
                    if gt_key in doc and str(doc[gt_key]).strip() != "":
                        gt_value = str(doc[gt_key])
                        break

                resps = s.get("resps", [[]])
                resps_list = resps[0] if resps and isinstance(resps, list) else []

                if key not in question_to_data:
                    question_to_data[key] = {"gt": gt_value, "resps": []}
                question_to_data[key]["resps"].extend(resps_list)

    # Compute metrics over merged responses for each question
    for qdata in question_to_data.values():
        gt = qdata["gt"]
        raw_resps = qdata["resps"]
        extracted_resps = [process_result(a) for a in raw_resps]

        # per-k accuracies
        for k in ks:
            sub = extracted_resps[:min(k, len(extracted_resps))]
            if not sub:
                continue
            c = Counter(sub)
            maj_ans, _ = top_nonblank(c)
            maj_correct[k] += int(is_equiv(maj_ans, gt))
            ora_correct[k] += int(any(is_equiv(a, gt) for a in sub))

        # stats over all available samples for this question
        c_all = Counter(extracted_resps)
        if len(extracted_resps) > 0:
            maj_ans_all, maj_cnt_all = top_nonblank(c_all)
            maj_prop_sum += (maj_cnt_all / len(extracted_resps)) if maj_ans_all != "" else 0.0
        uniq_cnt_sum += len(c_all)
        correct_total += c_all.get(gt, 0)
        total_resps += len(extracted_resps)

        n_q += 1

    maj_acc = {k: (maj_correct[k] / n_q if n_q else 0.0) for k in ks}
    ora_acc = {k: (ora_correct[k] / n_q if n_q else 0.0) for k in ks}
    maj_prop_avg = (maj_prop_sum / n_q) if n_q else 0.0
    uniq_cnt_avg = (uniq_cnt_sum / n_q) if n_q else 0.0
    correct_prop_over_all = (correct_total / total_resps) if total_resps else 0.0

    out = {
        "majority_voted_accuracy": maj_acc,
        "oracle_accuracy": ora_acc,
        "avg_majority_proportion_over_all_samples": maj_prop_avg,
        "avg_num_unique_answers": uniq_cnt_avg,
        "proportion_correct_over_all_samples": correct_prop_over_all,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    
    #base
    paths = ['/is/sg2/xli/vllm/eval_results/gpt-oss-20b_20250924_005701.json']
    main(paths)
