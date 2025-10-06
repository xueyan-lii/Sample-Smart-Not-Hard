import json, re, sys
from collections import Counter, defaultdict
from lm_eval.filters.extraction import RegexFilter

# keep only digits, minus, and dot; drop commas/spaces/others
def norm(s: str) -> str:
    s = s.lower().replace(",", "")
    return re.sub(r"[^0-9\.\-]+", "", s).strip()

# choose the most common non-blank answer; returns (answer, count) or ("", 0) if none
def top_nonblank(counter: Counter) -> tuple[str, int]:
    #if counter.most_common(1)[0][0] == "":
    #    print(":<")
    #for ans, cnt in counter.most_common():
    #    if ans != "":
    #        return ans, cnt
    #return "", 0
    return counter.most_common(1)[0]

def main(path,source):
    if source == "torchtune": 
        with open(path, "r") as f:
            data = json.load(f)
        samples = data["samples"]["gsm8k_fewshot_128"][:1312] #consider the only unique set
    if source == "lm_eval":
        samples = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line.strip())
                samples.append({"resps":json_obj["resps"],"target":json_obj["target"]})
        #print(samples[0])
    #ks = [1, 20, 40, 80]
    ks = [1,4, 8,16,32,64,128]
    rf = RegexFilter(regex_pattern=r"(?<=[Tt]he answer is )[^.\\]+")
    maj_correct = {k: 0 for k in ks}
    ora_correct = {k: 0 for k in ks}
    n_q = 0

    maj_prop_sum = 0.0
    uniq_cnt_sum = 0
    correct_total = 0
    total_resps = 0

    for s in samples:
        gt = norm(s["target"])
        resps = s["resps"][0]
        # extract answers once, then slice for ks
        extracted = rf.apply([resps], [{}])[0]
        extracted_norm = [norm(a) for a in extracted]
        #print(resps)
        #print(extracted_norm)

        # per-k accuracies
        for k in ks:
            sub = extracted_norm[:min(k, len(extracted_norm))]
            if not sub:
                continue
            c = Counter(sub)
            maj_ans, _ = top_nonblank(c)
            maj_correct[k] += int(maj_ans == gt)
            ora_correct[k] += int(gt in c)

        # stats over all available samples for this question
        c_all = Counter(extracted_norm)
        if len(extracted_norm) > 0:
            maj_ans_all, maj_cnt_all = top_nonblank(c_all)
            maj_prop_sum += (maj_cnt_all / len(extracted_norm)) if maj_ans_all != "" else 0.0
        uniq_cnt_sum += len(c_all)
        correct_total += c_all.get(gt, 0)
        total_resps += len(extracted_norm)

        # optional per-question dictionary of unique answers with counts and correctness (built per instructions)
        per_q = {ans: {"count": cnt, "is_correct": int(ans == gt)} for ans, cnt in c_all.items()}
        #for ans, info in sorted(per_q.items(), key=lambda kv: kv[1]["count"], reverse=True):
        #    print(ans, info)

        n_q += 1
    print(n_q)
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
    main("Qwen2.5-1.5B-Instruct_20250913_150412.json","torchtune")
