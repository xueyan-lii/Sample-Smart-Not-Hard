# run_eval.py
import argparse
import os
import json
from typing import List, Tuple, Any, Dict

import torch
from vllm import LLM, SamplingParams

# your custom logits processor
from dynamic_topk import CombinedSamplingLogitsProcessor

from lm_eval.evaluator import evaluate
from lm_eval.tasks import get_task_dict, TaskManager
from lm_eval.utils import make_table
import time
from datetime import timedelta, datetime
# Define a YAML loader that ignores unknown tags (e.g., !function) so we can read 'task'
from yaml.nodes import ScalarNode, SequenceNode, MappingNode  # type: ignore
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


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
        seed: int | None = None,
        max_gen_toks_default: int = 256,
        confidence_bin_only: int | None = None,
        cal_acc: List[float] | None = None,
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
            "confidence_bin_only": confidence_bin_only,
            "cal_acc": cal_acc,
        }

        logits_processors = [CombinedSamplingLogitsProcessor]
        self.llm = LLM(
            model=model_id,
            max_model_len=max_model_len,
            logits_processors=logits_processors,  # pass CLASS, not instance
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            #enforce_eager=True,
        )

        # lm-eval harness expects these attributes for sharding/caching/logging
        self.rank = 0
        self.world_size = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = seed

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
        
        # Build extra_args for the CombinedSamplingLogitsProcessor
        extra_args: Dict[str, Any] = {k: v for k, v in self.sampler_args.items() if v is not None}

        sp = SamplingParams(
            # Do NOT pass temperature/top_p/top_k/min_p here; handled by Combined processor
            max_tokens=self.max_gen_toks_default,
            stop=until,
            seed=self.seed,
            extra_args=extra_args,   # per-request knobs for combined processor
        )

        # vLLM accepts a list of SamplingParams matching prompt list
        outs = self.llm.generate(prompts=context, sampling_params=sp, use_tqdm=False)
        # Return generated strings in the order lm-eval expects
        return [o.outputs[0].text for o in outs]
        

    # Stubbed to make failures obvious if a task calls them
    def loglikelihood(self, _requests):
        raise NotImplementedError("This script only supports generate_until tasks.")
    def loglikelihood_rolling(self, _requests):
        raise NotImplementedError("This script only supports generate_until tasks.")



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run lm-eval with vLLM + Combined Sampler")

    p.add_argument("--task-config", type=str, default="gsm8k_fewshot.yaml")
    p.add_argument("--limit", type=int, default=None, help="Per-task example cap")
    p.add_argument("--output-dir", type=str, default=None)


    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--max-seq-length", type=int, default=9096)

    # vLLM parallel/engine knobs
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size (use >1 for multi-GPU TP)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="vLLM GPU memory utilization fraction")
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
    p.add_argument("--max-gen-toks", type=int, default=512)
    p.add_argument("--confidence-bin-only", type=int, default=None, help="If set to 1-10, only sample when pmax is in that bin; otherwise force greedy")
    p.add_argument("--seed", type=int, default=None, help="Random seed for vLLM SamplingParams")
    p.add_argument("--cal-acc", type=str, default=None, help="Comma-separated A,B,thr for calibrated accuracy threshold: y=10^A * x^B >= thr")
    
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
    task_name = task_yaml.get("task")
    if not task_name:
        raise ValueError("Task YAML must include a 'task' field with the task name.")
    include_path = os.path.dirname(args.task_config)

    # Parse composite flags
    dynamic_top_k = _parse_list_arg(args.dynamic_top_k, expect_len=10, cast=int)
    edt_var = _parse_list_arg(args.edt_var, expect_len=3, cast=float)
    # Alias: if --selective-greedy provided, use it as greedy-threshold when not set
    greedy_threshold = args.greedy_threshold if args.greedy_threshold is not None else args.selective_greedy
    cal_acc = _parse_list_arg(args.cal_acc, expect_len=3, cast=float)

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
        seed=args.seed,
        max_gen_toks_default=args.max_gen_toks,
        confidence_bin_only=args.confidence_bin_only,
        cal_acc=cal_acc,
    )

    # Build tasks
    task_manager = TaskManager(include_path=include_path)
    task_dict = get_task_dict(task_name, task_manager)

    # Run evaluation
    print(f"Running evaluation on tasks: {[task_name]} with model: {args.model}")
    print(f"Parameters used: {args}")
    time_start = time.time()
    output = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=args.limit,
        write_out=True,
        confirm_run_unsafe_code=True,  # keep if your tasks require it
    )
    if args.output_dir is not None:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            model_name_sanitized = args.model.split("/")[-1].replace(" ", "_")
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(args.output_dir, f"{model_name_sanitized}_{now_str}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False, default=str)
            print("Written output to: ", out_path)
        except Exception:
            pass
    print("\n" + make_table(output) + "\n")
    time_end = time.time()
    formatted_time = str(timedelta(seconds=time_end - time_start))
    print(f"Time taken: {formatted_time}")
if __name__ == "__main__":
    main()
