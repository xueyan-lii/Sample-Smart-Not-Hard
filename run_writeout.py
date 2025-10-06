# run_eval.py
import argparse
import json
import os
from datetime import datetime
from typing import List, Tuple, Any, Dict

import torch
from vllm import LLM, SamplingParams
import math

# your custom logits processor
from dynamic_topk import CombinedSamplingLogitsProcessor
from yaml.nodes import ScalarNode, SequenceNode, MappingNode  # type: ignore

from lm_eval.evaluator import evaluate
from lm_eval.tasks import get_task_dict, TaskManager
from lm_eval.utils import make_table

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# ------------ vLLM → lm-eval adapter ------------
class LMEvalVLLM:
    """
    Minimal wrapper implementing the request methods lm-eval calls.
    Here we implement only generate_until for generative tasks.
    """
    def __init__(
        self,
        model_id: str,
        *,
        max_model_len: int | None = None,
        # vLLM parallel/engine options
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = False,
        # unified sampler args (forwarded via extra_args to CombinedSamplingLogitsProcessor)
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        greedy_threshold: float | None = None,
        prob_threshold: float | None = None,
        dynamic_top_k: List[int] | None = None,
        edt_var: List[float] | None = None,
        hewitt_epsilon: float | None = None,
        max_gen_toks_default: int = 256,
        output_dir: str | None = None,
    ):
        # Store sampler args to pass through to SamplingParams.extra_args
        self.sampler_args: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "greedy_threshold": greedy_threshold,
            "prob_threshold": prob_threshold,
            "dynamic_top_k": dynamic_top_k,
            "edt_var": edt_var,
            "hewitt_epsilon": hewitt_epsilon,
        }

        logits_processors = [CombinedSamplingLogitsProcessor]
        self.llm = LLM(
            model=model_id,
            max_model_len=max_model_len,
            logits_processors=logits_processors,  # pass CLASS, not instance
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
        )

        # lm-eval harness expects these attributes for sharding/caching/logging
        self.rank = 0
        self.world_size = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # counters for low-confidence steps across all generations in this run
        self.low_conf_step_count: int = 0
        self.total_generated_steps: int = 0
        self.low_conf_threshold: float = 1.0
        self.all_sampled_threshold: float = 0.1
        self.tokenizer = self.llm.get_tokenizer()
        self.eos_id = self.tokenizer.eos_token_id
        # Prepare ranks log paths
        self.output_dir = output_dir
        self.low_conf_ranks_log_path: str | None = None
        self.all_sampled_ranks_log_path: str | None = None
        if self.output_dir is not None:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except Exception:
                pass
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.low_conf_ranks_log_path = os.path.join(self.output_dir, f"low_conf_ranks_{ts}.jsonl")
            self.all_sampled_ranks_log_path = os.path.join(self.output_dir, f"all_sampled_ranks_{ts}.jsonl")

        # Defaults for SamplingParams (do not use per-sampler fields here)
        self.max_gen_toks_default = max_gen_toks_default

    # ---- the method evaluator.py will call for generative tasks ----
    def generate_until(self, requests: List[Tuple[Any, ...]]):
        """
        Each entry is typically (context:str, until:list[str]|str|None, maybe_kwargs:dict?).
        We build per-request SamplingParams so max tokens / stops can differ.
        """
       
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        until = all_gen_kwargs[0].pop("until")
        
        # TEMP DEBUG: show a preview of the prompt context and stops
        try:
            print("[TEMP DEBUG] Number of requests:", len(context))
            for idx, ctx in enumerate(list(context)[:3]):
                safe_ctx = ctx if isinstance(ctx, str) else str(ctx)
                preview = safe_ctx[-1000:] if len(safe_ctx) > 1000 else safe_ctx
                print(f"\n[TEMP DEBUG] Request {idx} context tail (last 1000 chars or full):\n{preview}")
            print(f"\n[TEMP DEBUG] Stop strings: {until}")
        except Exception:
            pass
        
        # Build extra_args for the CombinedSamplingLogitsProcessor
        extra_args: Dict[str, Any] = {k: v for k, v in self.sampler_args.items() if v is not None}

        sp = SamplingParams(
            # Do NOT pass temperature/top_p/top_k/min_p here; handled by Combined processor
            max_tokens=self.max_gen_toks_default,
            stop=until,
            extra_args=extra_args,   # per-request knobs for combined processor
            # request top-1 logprobs so we can measure the maximum probability per step
            logprobs=1,
        )


        # vLLM accepts a list of SamplingParams matching prompt list
        outs = self.llm.generate(context, sp)
        # update low-confidence step counters across all returned generations
        self._update_low_confidence_counters(outs)
        # also collect and log low-confidence ranks per sequence
        self._get_low_confidence_ranks(outs)

        # Return generated strings in the order lm-eval expects
        return [o.outputs[0].text for o in outs]
        
    def _update_low_confidence_counters(self, request_outputs: List[Any]) -> None:
        """Update counters using per-token probabilities from vLLM outputs.

        Expects each sequence output to provide:
          - token_ids: List[int]
          - logprobs: List[Dict[int, Logprob]] where Logprob has a .logprob field
        We compute the per-step maximum probability as exp(max logprob over available entries).
        """

        # TEMP DEBUG: Print all generated tokens and their probabilities
        print("=" * 50)
        print("TEMPORARY DEBUG OUTPUT - Generated tokens and probabilities:")
        print("=" * 50)
        
        for req_idx, req_out in enumerate(request_outputs):
            for seq_idx, seq_out in enumerate(req_out.outputs):
                token_ids = seq_out.token_ids
                step_logprobs = seq_out.logprobs
                if token_ids is None or step_logprobs is None:
                    continue

                print(f"\nRequest {req_idx}, Sequence {seq_idx}:")
                print("-" * 30)
                
                num_steps = len(step_logprobs)
                for i in range(num_steps):
                    tok_id = token_ids[i]
                    step_lp_dict = step_logprobs[i]
                    
                    if step_lp_dict and tok_id in step_lp_dict:
                        chosen_logprob = step_lp_dict[tok_id].logprob
                        chosen_prob = math.exp(chosen_logprob)
                        token_text = self.tokenizer.decode([tok_id])
                        #print(f"  Step {i}: token_id={tok_id} '{token_text}' prob={chosen_prob:.6f}")
                
                low_count = 0
                step_count = 0

                for i in range(num_steps):
                    tok_id = token_ids[i]
                    step_lp_dict = step_logprobs[i]

                    # Use the step's top-1 probability (max over the distribution),
                    # not the chosen token's probability.
                    step_count += 1
                    if step_lp_dict:
                        top_entry = max(step_lp_dict.values(), key=lambda x: x.logprob)
                        top_prob = math.exp(top_entry.logprob)
                        if top_prob < self.low_conf_threshold:
                            low_count += 1

                self.low_conf_step_count += low_count
                self.total_generated_steps += step_count
        
        print("=" * 50)
        print("END TEMPORARY DEBUG OUTPUT")
        print("=" * 50)

    # Stubbed to make failures obvious if a task calls them
    def loglikelihood(self, _requests):
        raise NotImplementedError("This script only supports generate_until tasks.")
    def loglikelihood_rolling(self, _requests):
        raise NotImplementedError("This script only supports generate_until tasks.")
    
    def _get_low_confidence_ranks(self, request_outputs: List[Any]) -> None:
        # Stopping conditions and pattern tracking, mirroring eleuther_eval.py
        stop_token_ids = {14582, 271, 198, 382, 624}  # qwen2.5 7b instruct
        pattern_a = [785, 4226, 374, 220]   # qwen2.5 7b instruct
        pattern_b = [1782, 4226, 374, 220]  # qwen2.5 7b instruct

        # Collect ranks per sequence across all request outputs in this batch
        per_seq_low_conf_ranks: List[List[int]] = []
        per_seq_all_sampled_ranks: List[List[int]] = []
        per_seq_last_tokens: List[List[int]] = []
        per_seq_tracking_active: List[bool] = []
        per_seq_post_pattern_remaining: List[int] = []

        # Initialize tracking structures lazily per sequence
        for req_out in request_outputs:
            for seq_out in req_out.outputs:
                token_ids = seq_out.token_ids
                step_logprobs = seq_out.logprobs
                if token_ids is None or step_logprobs is None:
                    continue

                seq_low_conf_ranks: List[int] = []
                seq_all_ranks: List[int] = []
                last_tokens: List[int] = []
                tracking_active: bool = True
                post_pattern_remaining: int = -1

                num_steps = len(step_logprobs)
                for i in range(num_steps):
                    tok_id = token_ids[i]
                    # Maintain last-4 tokens for pattern detection
                    last_tokens.append(tok_id)
                    if len(last_tokens) > 4:
                        del last_tokens[0]

                    # Pattern detection: if matched and not already counting down, start 3-step countdown
                    if tracking_active and post_pattern_remaining < 0 and len(last_tokens) == 4:
                        if last_tokens == pattern_a or last_tokens == pattern_b:
                            post_pattern_remaining = 3

                    # Fetch chosen token's entry from returned logprobs if available
                    step_lp_dict = step_logprobs[i]
                    chosen_entry = step_lp_dict.get(tok_id)
                    
                    # Use the step's top-1 probability for low-confidence detection
                    # to align with SelectiveGreedy's thresholding. If the top-1 is
                    # unconfident, the rank should be 1 after masking.
                    if step_lp_dict:
                        top_entry = max(step_lp_dict.values(), key=lambda x: x.logprob)
                        top_prob = math.exp(top_entry.logprob)
                        if tracking_active and (math.exp(chosen_entry.logprob) < self.all_sampled_threshold):
                            seq_all_ranks.append(chosen_entry.rank)
                        
                        if tracking_active and top_prob < self.low_conf_threshold:
                            # record rank of the chosen token (should be 1 when selective greedy fires)
                            if chosen_entry is not None:
                                seq_low_conf_ranks.append(chosen_entry.rank)
                    
                    # Stop tracking immediately if a stop token is sampled
                    if tok_id in stop_token_ids:
                        tracking_active = False
                        post_pattern_remaining = -1

                    # Handle post-pattern countdown: after 3 more tokens, disable tracking
                    if tracking_active and post_pattern_remaining >= 0:
                        post_pattern_remaining -= 1
                        if post_pattern_remaining == 0:
                            tracking_active = False
                            post_pattern_remaining = -1

                per_seq_low_conf_ranks.append(seq_low_conf_ranks)
                per_seq_all_sampled_ranks.append(seq_all_ranks)
                per_seq_last_tokens.append(last_tokens)
                per_seq_tracking_active.append(tracking_active)
                per_seq_post_pattern_remaining.append(post_pattern_remaining)

        # Write out ranks only, one list per line (JSONL)
        if self.low_conf_ranks_log_path is not None and per_seq_low_conf_ranks:
            try:
                with open(self.low_conf_ranks_log_path, "a", encoding="utf-8") as f:
                    for ranks in per_seq_low_conf_ranks:
                        f.write(json.dumps(ranks, ensure_ascii=False) + "\n")
            except Exception:
                pass

        if self.all_sampled_ranks_log_path is not None and per_seq_all_sampled_ranks:
            try:
                with open(self.all_sampled_ranks_log_path, "a", encoding="utf-8") as f:
                    for ranks in per_seq_all_sampled_ranks:
                        f.write(json.dumps(ranks, ensure_ascii=False) + "\n")
            except Exception:
                pass



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run lm-eval with vLLM + Combined Sampler")

    p.add_argument("--task-config", type=str, default="gsm8k_fewshot.yaml")
    p.add_argument("--limit", type=int, default=None, help="Per-task example cap")

    p.add_argument("--output-dir", type=str, default="eval_results")

    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-seq-length", type=int, default=4096)

    # vLLM parallel/engine knobs
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size (use >1 for multi-GPU TP)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5, help="vLLM GPU memory utilization fraction")
    p.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code to vLLM")

    # unified sampler knobs (all optional; combined in one processor)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--min-p", type=float, default=None)
    # support both names; --selective-greedy is an alias for --greedy-threshold
    p.add_argument("--greedy-threshold", type=float, default=None)
    p.add_argument("--selective-greedy", type=float, default=None)
    p.add_argument("--prob-threshold", type=float, default=None)
    p.add_argument("--dynamic-top-k", type=str, default=None, help="Comma-separated 10 integers for dynamic top-k bins")
    p.add_argument("--edt-var", type=str, default=None, help="Comma-separated N,theta,T0 for EDT")
    p.add_argument("--hewitt-epsilon", type=float, default=None)
    p.add_argument("--max-gen-toks", type=int, default=256)

    return p.parse_args()



def _parse_list_arg(arg_val: str | None, expect_len: int | None = None, cast=float):
    if arg_val is None:
        return None
    parts = [s.strip() for s in arg_val.split(",") if s.strip()]
    try:
        values = [cast(x) for x in parts]
    except Exception:
        return None
    if expect_len is not None and len(values) != expect_len:
        return None
    return values


def main() -> None:
    args = _parse_args()
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse --task-config. Please install pyyaml.")

    class _IgnoreUnknownTagLoader(yaml.SafeLoader):  # type: ignore
        pass

    def _ignore_unknown_constructor(loader, tag_suffix, node):  # type: ignore
        if isinstance(node, ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, SequenceNode):
            return loader.construct_sequence(node)
        return loader.construct_mapping(node)

    _IgnoreUnknownTagLoader.add_multi_constructor("!", _ignore_unknown_constructor)  # type: ignore

    # Resolve task name and include_path from YAML
    if not os.path.isabs(args.task_config):
        raise ValueError("--task-config must be an absolute path")
    with open(args.task_config, "r", encoding="utf-8") as f:
        task_yaml = yaml.load(f, Loader=_IgnoreUnknownTagLoader)
    task_name = str(task_yaml.get("task"))
    if not task_name:
        raise ValueError("Task YAML must include a 'task' field with the task name.")
    include_path = os.path.dirname(args.task_config)

    # Parse composite flags
    dynamic_top_k = _parse_list_arg(args.dynamic_top_k, expect_len=10, cast=int)
    edt_var = _parse_list_arg(args.edt_var, expect_len=3, cast=float)
    # Alias: if --selective-greedy provided, use it as greedy-threshold when not set
    greedy_threshold = args.greedy_threshold if args.greedy_threshold is not None else args.selective_greedy

    # Build the vLLM-backed LM wrapper
    lm = LMEvalVLLM(
        model_id=args.model,
        max_model_len=args.max_seq_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        greedy_threshold=greedy_threshold,
        prob_threshold=args.prob_threshold,
        dynamic_top_k=dynamic_top_k,
        edt_var=edt_var,
        hewitt_epsilon=args.hewitt_epsilon,
        max_gen_toks_default=args.max_gen_toks,
        output_dir=args.output_dir,
    )

    # Build tasks
    task_manager = TaskManager(include_path=include_path)
    task_dict = get_task_dict([task_name], task_manager)

    # Run evaluation
    print(f"Running evaluation on tasks: {[task_name]} with model: {args.model}")
    output = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=args.limit,
        write_out=True,
        confirm_run_unsafe_code=True,  # keep if your tasks require it
    )
    print("\n" + make_table(output) + "\n")

    
    # Log aggregate low-confidence steps across all generations
    print(f"Low-confidence steps (max prob < {lm.low_conf_threshold:.2f}): {lm.low_conf_step_count}/{lm.total_generated_steps}={lm.low_conf_step_count / lm.total_generated_steps:.4f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_name_sanitized = args.model.split("/")[-1].replace(" ", "_")
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"{model_name_sanitized}_{now_str}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved evaluation output to: {out_path}")
    print(f"Low-confidence ranks log path: {lm.low_conf_ranks_log_path}")
    print(f"All-sampled ranks log path: {lm.all_sampled_ranks_log_path}")
    

if __name__ == "__main__":
    main()
