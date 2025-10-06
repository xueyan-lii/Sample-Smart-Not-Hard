# dynamic_topk.py
from typing import Optional, Dict, Tuple, List
import torch
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    LogitsProcessor, BatchUpdate, MoveDirectionality
)
class SelectiveGreedyLogitsProcessor(LogitsProcessor):

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.threshold_count: int = 0

        self.threshold_cpu_tensor = torch.zeros((max_num_reqs, ),
                                                dtype=torch.float32,
                                                device="cpu",
                                                pin_memory=is_pin_memory)
        self.threshold_cpu = self.threshold_cpu_tensor.numpy()

        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            # Pre-allocated device tensor
            self.threshold_device: torch.Tensor = torch.empty((max_num_reqs, ),
                                                              dtype=torch.float32,
                                                              device=device)
        else:
            self.threshold_device = self.threshold_cpu_tensor
        # Current slice of the device tensor
        self.threshold: torch.Tensor = self.threshold_device[:0]

    def is_argmax_invariant(self) -> bool:
        """Selective Greedy preserves greedy argmax when not set"""
        return True

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        needs_update = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            # Pull threshold from SamplingParams.extra_args if provided
            thresh_val = 0.0
            extra_args = getattr(params, "extra_args", None)
            if extra_args:
                val = extra_args.get("selgreedy_thresh")
                if val is not None:
                    try:
                        thresh_val = float(val)
                    except Exception:
                        thresh_val = 0.0

            prev = self.threshold_cpu[index]
            if prev != thresh_val:
                needs_update = True
                self.threshold_cpu[index] = thresh_val
                if (thresh_val > 0.0) and (prev <= 0.0):
                    self.threshold_count += 1
                elif (thresh_val <= 0.0) and (prev > 0.0):
                    self.threshold_count -= 1

        if self.threshold_count:
            # Process removed requests.
            if batch_update.removed:
                needs_update = True
                for index in batch_update.removed:
                    if self.threshold_cpu[index] > 0.0:
                        self.threshold_cpu[index] = 0.0
                        self.threshold_count -= 1

            # Process moved requests, unidirectional (a->b) and swap (a<->b).
            for adx, bdx, direct in batch_update.moved:
                a_val, b_val = self.threshold_cpu[adx], self.threshold_cpu[bdx]
                if a_val != b_val:
                    needs_update = True
                    self.threshold_cpu[bdx] = a_val
                    if direct == MoveDirectionality.SWAP:
                        self.threshold_cpu[adx] = b_val
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if a_val > 0.0:
                        self.threshold_cpu[adx] = 0.0
                    if b_val > 0.0:
                        self.threshold_count -= 1

        # Update tensors if needed.
        size = batch_update.batch_size
        if self.threshold_count and (needs_update or self.threshold.shape[0] != size):
            self.threshold = self.threshold_device[:size]
            if self.use_double_tensor:
                self.threshold.copy_(self.threshold_cpu_tensor[:size],
                                     non_blocking=True)
            self.threshold.unsqueeze_(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.threshold_count:
            return logits

        # Compute softmax and its maximum per row
        probs = torch.nn.functional.softmax(logits, dim=-1)
        maxp, argmax_ids = probs.max(dim=-1)

        row_thresholds = self.threshold.squeeze(1) if self.threshold.dim() == 2 else self.threshold
        rows = (maxp < row_thresholds).nonzero(as_tuple=True)[0]
        if rows.numel() > 0:
            logits[rows] = float("-inf")
            logits[rows, argmax_ids[rows]] = 0.0
        return logits
    
class SelectivePLogitsProcessor(LogitsProcessor):

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.min_p_count: int = 0

        self.min_p_cpu_tensor = torch.zeros((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=is_pin_memory)
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()

        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            # Pre-allocated device tensor
            self.min_p_device: torch.Tensor = torch.empty((max_num_reqs, ),
                                                          dtype=torch.float32,
                                                          device=device)
        else:
            self.min_p_device = self.min_p_cpu_tensor
        # Current slice of the device tensor
        self.min_p: torch.Tensor = self.min_p_device[:0]

    def is_argmax_invariant(self) -> bool:
        """Min-p never impacts greedy sampling"""
        return True

    def get_min_p_by_index(self, index: int) -> float:
        return float(self.min_p_cpu[index])

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        needs_update = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            # Prefer value from extra_args (e.g., when using custom processor),
            # otherwise fall back to SamplingParams.min_p
            min_p_val = 0.0
            extra_args = getattr(params, "extra_args", None)
            if extra_args is not None:
                try:
                    v = extra_args.get("selp_min_p")
                    if v is not None:
                        min_p_val = float(v)
                    else:
                        min_p_val = float(getattr(params, "min_p", 0.0))
                except Exception:
                    min_p_val = float(getattr(params, "min_p", 0.0))
            else:
                min_p_val = float(getattr(params, "min_p", 0.0))

            min_p_before = float(self.min_p_cpu[index])
            if min_p_before != min_p_val:
                needs_update = True
                self.min_p_cpu[index] = min_p_val
                if (min_p_val > 0.0) and (min_p_before <= 0.0):
                    self.min_p_count += 1
                elif (min_p_val <= 0.0) and (min_p_before > 0.0):
                    self.min_p_count -= 1

        if self.min_p_count:
            # Process removed requests.
            if batch_update.removed:
                needs_update = True
                for index in batch_update.removed:
                    if self.min_p_cpu[index]:
                        self.min_p_cpu[index] = 0
                        self.min_p_count -= 1

            # Process moved requests, unidirectional (a->b) and swap (a<->b).
            for adx, bdx, direct in batch_update.moved:
                min_p_a, min_p_b = self.min_p_cpu[adx], self.min_p_cpu[bdx]
                if min_p_a != min_p_b:
                    needs_update = True
                    self.min_p_cpu[bdx] = min_p_a
                    if direct == MoveDirectionality.SWAP:
                        self.min_p_cpu[adx] = min_p_b
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if min_p_a:
                        self.min_p_cpu[adx] = 0
                    if min_p_b:
                        self.min_p_count -= 1

        # Update tensors if needed.
        size = batch_update.batch_size
        if self.min_p_count and (needs_update or self.min_p.shape[0] != size):
            self.min_p = self.min_p_device[:size]
            if self.use_double_tensor:
                self.min_p.copy_(self.min_p_cpu_tensor[:size],
                                 non_blocking=True)
            self.min_p.unsqueeze_(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits

        # Convert logits to probability distribution
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Identify invalid tokens using threshold comparison (broadcasts over rows)
        invalid_token_mask = probs < self.min_p
        if invalid_token_mask.any():
            # Mask out invalid tokens by setting logits to -inf
            logits[invalid_token_mask] = float("-inf")
            # Ensure at least one token remains per row; if a row is fully masked, keep argmax
            all_masked_rows = invalid_token_mask.all(dim=-1).nonzero(as_tuple=True)[0]
            if all_masked_rows.numel() > 0:
                _, argmax_ids = probs.max(dim=-1)
                logits[all_masked_rows, argmax_ids[all_masked_rows]] = 0.0
        return logits

# ---------------- Combined Processor -----------------
class CombinedSamplingLogitsProcessor(LogitsProcessor):
    """
    A single logits processor that composes multiple sampling constraints without
    intervening renormalizations. Supports:
      - temperature (scalar)
      - top_k (constant)
      - top_p (nucleus sampling)
      - min_p (keep tokens with prob >= max_prob * min_p)
      - greedy_threshold (force greedy when max softmax prob < threshold)
      - prob_threshold (keep tokens with prob >= threshold)
      - dynamic_top_k (length-10 list mapping pmax bins to per-row k)
      - edt_var (N, theta, T0) for entropy-dependent temperature (overrides temperature)
      - hewitt_epsilon (epsilon threshold as in Hewitt truncation)
      - confidence_bin_only (1-10): only apply sampling when pmax falls in the chosen bin; otherwise force greedy

    Values are read from SamplingParams.extra_args.
    """
    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.device = device
        self.use_double_tensor = torch.device(device).type != "cpu"

        # Per-request scalar settings (float32 tensors on CPU with optional pinned memory)
        def _alloc_cpu():
            return torch.zeros((max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=is_pin_memory)

        self._temperature_cpu = _alloc_cpu()  # 0 means use 1.0
        self._temperature_set_cpu = torch.zeros((max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=is_pin_memory)
        self._top_k_cpu = torch.zeros((max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=is_pin_memory)
        self._top_p_cpu = _alloc_cpu()
        self._min_p_cpu = _alloc_cpu()
        self._greedy_thr_cpu = _alloc_cpu()
        self._prob_thr_cpu = _alloc_cpu()
        self._hewitt_eps_cpu = _alloc_cpu()
        # Calibration: per-request A, B, and threshold for y=10^A * x^B >= thr
        self._cal_A_cpu = _alloc_cpu()
        self._cal_B_cpu = _alloc_cpu()
        self._cal_thr_cpu = _alloc_cpu()
        # EDT per-request parameters (N, theta, T0)
        self._edt_N_cpu = _alloc_cpu()
        self._edt_theta_cpu = _alloc_cpu()
        self._edt_T0_cpu = _alloc_cpu()
        # Confidence bin gating (int32; 0 means inactive)
        self._only_bin_cpu = torch.zeros((max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=is_pin_memory)

        # Per-batch slices on device
        def _maybe_device_like(cpu_tensor: torch.Tensor, dtype=None):
            if self.use_double_tensor:
                return torch.empty_like(cpu_tensor, device=device, dtype=dtype or cpu_tensor.dtype)
            return cpu_tensor

        self._temperature_dev = _maybe_device_like(self._temperature_cpu)
        self._temperature_set_dev = torch.empty_like(self._temperature_set_cpu, device=device) if self.use_double_tensor else self._temperature_set_cpu
        self._top_k_dev = torch.empty_like(self._top_k_cpu, device=device) if self.use_double_tensor else self._top_k_cpu
        self._top_p_dev = _maybe_device_like(self._top_p_cpu)
        self._min_p_dev = _maybe_device_like(self._min_p_cpu)
        self._greedy_thr_dev = _maybe_device_like(self._greedy_thr_cpu)
        self._prob_thr_dev = _maybe_device_like(self._prob_thr_cpu)
        self._hewitt_eps_dev = _maybe_device_like(self._hewitt_eps_cpu)
        self._cal_A_dev = _maybe_device_like(self._cal_A_cpu)
        self._cal_B_dev = _maybe_device_like(self._cal_B_cpu)
        self._cal_thr_dev = _maybe_device_like(self._cal_thr_cpu)
        self._edt_N_dev = _maybe_device_like(self._edt_N_cpu)
        self._edt_theta_dev = _maybe_device_like(self._edt_theta_cpu)
        self._edt_T0_dev = _maybe_device_like(self._edt_T0_cpu)
        self._only_bin_dev = torch.empty_like(self._only_bin_cpu, device=device) if self.use_double_tensor else self._only_bin_cpu

        # Current batch views (will be sliced in update_state)
        self.temperature: torch.Tensor = self._temperature_dev[:0]
        self.temperature_set: torch.Tensor = self._temperature_set_dev[:0]
        self.top_k: torch.Tensor = self._top_k_dev[:0]
        self.top_p: torch.Tensor = self._top_p_dev[:0]
        self.min_p: torch.Tensor = self._min_p_dev[:0]
        self.greedy_thr: torch.Tensor = self._greedy_thr_dev[:0]
        self.prob_thr: torch.Tensor = self._prob_thr_dev[:0]
        self.hewitt_eps: torch.Tensor = self._hewitt_eps_dev[:0]
        self.edt_N: torch.Tensor = self._edt_N_dev[:0]
        self.edt_theta: torch.Tensor = self._edt_theta_dev[:0]
        self.edt_T0: torch.Tensor = self._edt_T0_dev[:0]
        self.only_bin: torch.Tensor = self._only_bin_dev[:0]
        self.cal_A: torch.Tensor = self._cal_A_dev[:0]
        self.cal_B: torch.Tensor = self._cal_B_dev[:0]
        self.cal_thr: torch.Tensor = self._cal_thr_dev[:0]

        # Non per-request settings (shared for the batch) read from last seen request
        self.dynamic_top_k: Optional[List[int]] = None

        # Active counts for quick checks
        self._active_any: bool = False

    def is_argmax_invariant(self) -> bool:
        # When nothing is enabled, we keep logits unchanged.
        return True

    def _read_extra_args(self, extra_args: Optional[Dict]) -> Dict:
        if not extra_args:
            return {}
        return {
            "temperature": extra_args.get("temperature"),
            "top_k": extra_args.get("top_k"),
            "top_p": extra_args.get("top_p"),
            "min_p": extra_args.get("min_p"),
            # support both names for greedy threshold
            "greedy_threshold": extra_args.get("greedy_threshold", extra_args.get("selgreedy_thresh", extra_args.get("selective_greedy"))),
            "prob_threshold": extra_args.get("prob_threshold"),
            "dynamic_top_k": extra_args.get("dynamic_top_k"),
            "edt_var": extra_args.get("edt_var"),
            "hewitt_epsilon": extra_args.get("hewitt_epsilon"),
            "confidence_bin_only": extra_args.get("confidence_bin_only", extra_args.get("sample_only_bin", extra_args.get("confidence_only_bin"))),
            "cal_acc": extra_args.get("cal_acc"),
        }

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        needs_update = False
        # Process added requests: snapshot per-request scalar parameters
        for index, params, _, _ in batch_update.added:
            values = self._read_extra_args(getattr(params, "extra_args", None))

            def _set_if_present(cpu_arr: torch.Tensor, key: str, cast_fn):
                nonlocal needs_update
                val = values.get(key)
                if val is not None:
                    try:
                        cpu_val = cast_fn(val)
                    except Exception:
                        return
                    prev = float(cpu_arr[index])
                    if cpu_val != prev:
                        needs_update = True
                        cpu_arr[index] = cpu_val

            _set_if_present(self._temperature_cpu, "temperature", float)
            # Track whether temperature was explicitly provided so that temperature==0 implies greedy
            if values.get("temperature") is not None:
                if int(self._temperature_set_cpu[index].item()) != 1:
                    self._temperature_set_cpu[index] = 1
                    needs_update = True
            # ints for top_k
            val_top_k = values.get("top_k")
            if val_top_k is not None:
                try:
                    v = int(val_top_k)
                    if int(self._top_k_cpu[index].item()) != v:
                        needs_update = True
                        self._top_k_cpu[index] = v
                except Exception:
                    pass
            _set_if_present(self._top_p_cpu, "top_p", float)
            _set_if_present(self._min_p_cpu, "min_p", float)
            _set_if_present(self._greedy_thr_cpu, "greedy_threshold", float)
            _set_if_present(self._prob_thr_cpu, "prob_threshold", float)
            _set_if_present(self._hewitt_eps_cpu, "hewitt_epsilon", float)

            # EDT per-request (N, theta, T0)
            if values.get("edt_var") is not None:
                try:
                    ev = list(map(float, values["edt_var"]))
                    if len(ev) == 3:
                        if float(self._edt_N_cpu[index]) != ev[0]:
                            self._edt_N_cpu[index] = ev[0]
                            needs_update = True
                        if float(self._edt_theta_cpu[index]) != ev[1]:
                            self._edt_theta_cpu[index] = ev[1]
                            needs_update = True
                        if float(self._edt_T0_cpu[index]) != ev[2]:
                            self._edt_T0_cpu[index] = ev[2]
                            needs_update = True
                except Exception:
                    pass

            # Calibration per-request (A, B, thr)
            if values.get("cal_acc") is not None:
                try:
                    cal = list(map(float, values["cal_acc"]))
                    if len(cal) == 3:
                        if float(self._cal_A_cpu[index]) != cal[0]:
                            self._cal_A_cpu[index] = cal[0]
                            needs_update = True
                        if float(self._cal_B_cpu[index]) != cal[1]:
                            self._cal_B_cpu[index] = cal[1]
                            needs_update = True
                        if float(self._cal_thr_cpu[index]) != cal[2]:
                            self._cal_thr_cpu[index] = cal[2]
                            needs_update = True
                except Exception:
                    pass

            # Shared settings (take the last seen non-None)
            if values.get("dynamic_top_k") is not None:
                try:
                    dtk = list(map(int, values["dynamic_top_k"]))
                    if len(dtk) == 10:
                        self.dynamic_top_k = dtk
                        needs_update = True
                except Exception:
                    pass

            # Confidence bin only (per-request int 1..10; 0 disables)
            val_only_bin = values.get("confidence_bin_only")
            if val_only_bin is not None:
                try:
                    ibin = int(val_only_bin)
                    if 1 <= ibin <= 10:
                        if int(self._only_bin_cpu[index].item()) != ibin:
                            self._only_bin_cpu[index] = ibin
                            needs_update = True
                except Exception:
                    pass

        # Handle removed requests: reset to 0
        if batch_update.removed:
            needs_update = True
            for index in batch_update.removed:
                self._temperature_cpu[index] = 0.0
                self._temperature_set_cpu[index] = 0
                self._top_k_cpu[index] = 0
                self._top_p_cpu[index] = 0.0
                self._min_p_cpu[index] = 0.0
                self._greedy_thr_cpu[index] = 0.0
                self._prob_thr_cpu[index] = 0.0
                self._hewitt_eps_cpu[index] = 0.0
                self._edt_N_cpu[index] = 0.0
                self._edt_theta_cpu[index] = 0.0
                self._edt_T0_cpu[index] = 0.0
                self._only_bin_cpu[index] = 0
                self._cal_A_cpu[index] = 0.0
                self._cal_B_cpu[index] = 0.0
                self._cal_thr_cpu[index] = 0.0

        # Process moved requests
        for adx, bdx, direct in batch_update.moved:
            # move/copy scalar values
            def _move_scalar(arr: torch.Tensor):
                a_val, b_val = arr[adx].item(), arr[bdx].item()
                if a_val != b_val:
                    nonlocal needs_update
                    needs_update = True
                    arr[bdx] = a_val
                    if direct == MoveDirectionality.SWAP:
                        arr[adx] = b_val
                    elif direct == MoveDirectionality.UNIDIRECTIONAL:
                        arr[adx] = 0

            _move_scalar(self._temperature_cpu)
            _move_scalar(self._temperature_set_cpu)
            _move_scalar(self._top_k_cpu)
            _move_scalar(self._top_p_cpu)
            _move_scalar(self._min_p_cpu)
            _move_scalar(self._greedy_thr_cpu)
            _move_scalar(self._prob_thr_cpu)
            _move_scalar(self._hewitt_eps_cpu)
            _move_scalar(self._edt_N_cpu)
            _move_scalar(self._edt_theta_cpu)
            _move_scalar(self._edt_T0_cpu)
            _move_scalar(self._only_bin_cpu)
            _move_scalar(self._cal_A_cpu)
            _move_scalar(self._cal_B_cpu)
            _move_scalar(self._cal_thr_cpu)

        # Update current batch slices
        size = batch_update.batch_size
        if needs_update or self.temperature.shape[0] != size:
            self.temperature = self._temperature_dev[:size]
            self.temperature_set = self._temperature_set_dev[:size]
            self.top_k = self._top_k_dev[:size]
            self.top_p = self._top_p_dev[:size]
            self.min_p = self._min_p_dev[:size]
            self.greedy_thr = self._greedy_thr_dev[:size]
            self.prob_thr = self._prob_thr_dev[:size]
            self.hewitt_eps = self._hewitt_eps_dev[:size]
            self.edt_N = self._edt_N_dev[:size]
            self.edt_theta = self._edt_theta_dev[:size]
            self.edt_T0 = self._edt_T0_dev[:size]
            self.only_bin = self._only_bin_dev[:size]
            self.cal_A = self._cal_A_dev[:size]
            self.cal_B = self._cal_B_dev[:size]
            self.cal_thr = self._cal_thr_dev[:size]
            if self.use_double_tensor:
                self.temperature.copy_(self._temperature_cpu[:size], non_blocking=True)
                self.temperature_set.copy_(self._temperature_set_cpu[:size], non_blocking=True)
                self.top_k.copy_(self._top_k_cpu[:size], non_blocking=True)
                self.top_p.copy_(self._top_p_cpu[:size], non_blocking=True)
                self.min_p.copy_(self._min_p_cpu[:size], non_blocking=True)
                self.greedy_thr.copy_(self._greedy_thr_cpu[:size], non_blocking=True)
                self.prob_thr.copy_(self._prob_thr_cpu[:size], non_blocking=True)
                self.hewitt_eps.copy_(self._hewitt_eps_cpu[:size], non_blocking=True)
                self.edt_N.copy_(self._edt_N_cpu[:size], non_blocking=True)
                self.edt_theta.copy_(self._edt_theta_cpu[:size], non_blocking=True)
                self.edt_T0.copy_(self._edt_T0_cpu[:size], non_blocking=True)
                self.only_bin.copy_(self._only_bin_cpu[:size], non_blocking=True)
                self.cal_A.copy_(self._cal_A_cpu[:size], non_blocking=True)
                self.cal_B.copy_(self._cal_B_cpu[:size], non_blocking=True)
                self.cal_thr.copy_(self._cal_thr_cpu[:size], non_blocking=True)
            # Track if anything is active
            self._active_any = (
                (self.temperature.abs().sum() > 0)
                or (((self.temperature == 0) & (self.temperature_set > 0)).any())
                or (self.top_k.abs().sum() > 0)
                or (self.top_p.abs().sum() > 0)
                or (self.min_p.abs().sum() > 0)
                or (self.greedy_thr.abs().sum() > 0)
                or (self.prob_thr.abs().sum() > 0)
                or (self.hewitt_eps.abs().sum() > 0)
                or (self.edt_N.abs().sum() > 0)
                or (self.edt_theta.abs().sum() > 0)
                or (self.edt_T0.abs().sum() > 0)
                or (self.dynamic_top_k is not None)
                or (self.only_bin.abs().sum() > 0)
                or (self.cal_thr.abs().sum() > 0)
            )

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        min_entropy = torch.finfo(entropy.dtype).eps
        return torch.clamp(entropy, min=min_entropy)

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        # EDT (per-request) overrides scalar temperature for rows where provided
        if self.edt_N.numel() > 0 and (self.edt_N > 0).any():
            entropy = self._compute_entropy(logits)
            # Ensure valid bases and parameters per row
            N = torch.where(self.edt_N <= 0, torch.ones_like(self.edt_N), self.edt_N)
            theta = self.edt_theta
            T0 = torch.where(self.edt_T0 <= 0, torch.ones_like(self.edt_T0), self.edt_T0)
            exp_arg = (theta / entropy) * torch.log(torch.clamp(N, min=1e-9))
            exp_arg = torch.clamp(exp_arg, min=1e-5)
            T_eff = T0 * torch.exp(exp_arg)
            logits = logits / T_eff.unsqueeze(1)
            return logits
        # Per-row scalar temperature; 0.0 means 1.0
        temps = self.temperature
        if temps.numel() > 0:
            temps = torch.where(temps <= 0, torch.ones_like(temps), temps)
            logits = logits / temps.unsqueeze(1)
        return logits

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # Fast path: nothing to do
        if not self._active_any:
            return logits

        # 1) Temperature/EDT scaling for current logits tensor (raw for all constraints)
        logits = self._apply_temperature(logits)

        # 2) Compute probabilities once from raw (scaled) logits and sort
        probs = torch.nn.functional.softmax(logits, dim=-1)
        B, V = probs.shape
        device = logits.device

        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        pmax = sorted_probs[:, 0]

        # Initialize per-row keep K as V (no restriction)
        keep_k = torch.full((B,), V, dtype=torch.int64, device=device)

        # 3) Constant top-k -> K = top_k (or V if inactive/zero), clamp to [1, V]
        if self.top_k.numel() > 0:
            row_k = self.top_k.clamp(min=0, max=V).to(dtype=torch.int64)
            k_topk = torch.where(row_k > 0, row_k.clamp(min=1), torch.full_like(row_k, V))
            keep_k = torch.minimum(keep_k, k_topk)

        # 4) Dynamic top-k based on pmax bins -> K = chosen_k
        if self.dynamic_top_k is not None:
            bin_base = torch.clamp((pmax * 10).floor(), max=9).to(torch.int64)
            bin_idx = 9 - bin_base
            dtk = torch.tensor(self.dynamic_top_k, device=device, dtype=torch.int64)
            chosen_k = dtk.index_select(0, bin_idx).clamp(min=1, max=V)
            keep_k = torch.minimum(keep_k, chosen_k)

        # 5) prob_threshold absolute -> K = count(probs >= thr)
        if self.prob_thr.numel() > 0 and self.prob_thr.max() > 0:
            thr = self.prob_thr.clamp(min=0.0, max=1.0)
            counts = (sorted_probs >= thr.unsqueeze(1)).sum(dim=1).to(torch.int64)
            k_prob_thr = torch.clamp(counts, min=1)
            keep_k = torch.minimum(keep_k, k_prob_thr)

        # 6) min_p relative to max prob -> K = count(probs >= min_p * pmax)
        if self.min_p.numel() > 0 and self.min_p.max() > 0:
            mp = self.min_p.clamp(min=0.0, max=1.0)
            thresh = pmax * mp
            counts = (sorted_probs >= thresh.unsqueeze(1)).sum(dim=1).to(torch.int64)
            k_minp = torch.clamp(counts, min=1)
            keep_k = torch.minimum(keep_k, k_minp)

        # 7) top_p nucleus -> K = count(cumprobs <= p) + 1
        if self.top_p.numel() > 0 and self.top_p.max() > 0:
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            p = self.top_p.clamp(min=torch.finfo(probs.dtype).eps, max=1.0)
            k_topp = (cumprobs <= p.unsqueeze(1)).sum(dim=-1).to(torch.int64) + 1
            keep_k = torch.minimum(keep_k, k_topp)

        # 8) Hewitt epsilon -> K = count(probs >= thresh_row)
        if self.hewitt_eps.numel() > 0 and self.hewitt_eps.max() > 0:
            entropy = self._compute_entropy(logits)
            eps = self.hewitt_eps
            thresh_row = torch.minimum(eps, torch.sqrt(eps) * torch.exp(-entropy))
            counts = (sorted_probs >= thresh_row.unsqueeze(1)).sum(dim=1).to(torch.int64)
            k_hewitt = torch.clamp(counts, min=1)
            keep_k = torch.minimum(keep_k, k_hewitt)

        # 9) Greedy threshold -> if pmax < thr then K = 1 else no restriction
        if self.greedy_thr.numel() > 0 and self.greedy_thr.max() > 0:
            need_force = pmax < self.greedy_thr
            k_greedy = torch.where(need_force, torch.ones_like(keep_k), torch.full_like(keep_k, V))
            keep_k = torch.minimum(keep_k, k_greedy)

        # 10) Confidence-bin gating -> force greedy outside desired bin
        if self.only_bin.numel() > 0 and self.only_bin.max() > 0:
            bin_numbers = torch.clamp((pmax * 10).floor(), max=9).to(torch.int64) + 1
            desired_bins = self.only_bin.to(torch.int64)
            outside = (desired_bins > 0) & (bin_numbers != desired_bins)
            keep_k = torch.where(outside, torch.ones_like(keep_k), keep_k)

        # 11) Calibration derived from best fit line. log(acc)=A+Blog(p)
        if self.cal_thr.numel() > 0 and self.cal_thr.max() > 0 and self.cal_A.numel() > 0 and self.cal_B.numel() > 0:
            calA = self.cal_A.unsqueeze(1)
            calB = self.cal_B.unsqueeze(1)
            calThr = self.cal_thr.unsqueeze(1)
            sorted_acc = torch.pow(10.0, calA) * torch.pow(sorted_probs, calB)
            counts = (sorted_acc >= calThr).sum(dim=1).to(torch.int64)
            k_cal = torch.clamp(counts, min=1)
            keep_k = torch.minimum(keep_k, k_cal)

        # 12) Explicit temperature == 0 -> force greedy sampling (K=1)
        if self.temperature.numel() > 0 and self.temperature_set.numel() > 0:
            force_rows = (self.temperature_set > 0) & (self.temperature == 0)
            if force_rows.any():
                keep_k = torch.where(force_rows, torch.ones_like(keep_k), keep_k)

        # Final clamp for safety
        keep_k = keep_k.clamp(min=1, max=V)

        # Build final mask: keep top-K per row based on sorted indices
        idx = torch.arange(V, device=device).unsqueeze(0).expand(B, V)
        keep_sorted = idx < keep_k.unsqueeze(1)
        keep_mask = torch.zeros_like(sorted_probs, dtype=torch.bool)
        keep_mask.scatter_(1, sorted_idx, keep_sorted)

        if (~keep_mask).any():
            logits[~keep_mask] = float("-inf")

        # Guarantee at least one token remains per row (should already hold)
        all_masked = torch.isinf(logits).all(dim=-1)
        if all_masked.any():
            argmax_ids = sorted_idx[:, 0]
            rows = all_masked.nonzero(as_tuple=True)[0]
            logits[rows] = float("-inf")
            logits[rows, argmax_ids[rows]] = 0.0

        return logits