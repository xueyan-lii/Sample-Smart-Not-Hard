import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import numpy as np

class CalibratedTopK:
    """
    Bin by p_max into 10 confidence bins, average probs by rank in each bin,
    then choose a single rank threshold per bin: the largest rank r with
    avg_prob_at_rank[r-1] >= p.
    """
    def __init__(self, num_bins: int = 10, V: int | None = None, bin_edges: np.ndarray | None = None):
        self.num_bins = num_bins
        self.V = V
        self.bin_edges = np.linspace(0.0, 1.0, num_bins + 1) if bin_edges is None else np.asarray(bin_edges)
        self.sum_probs = None   # shape [B, V]
        self.counts = None      # shape [B]
        self.avg_by_bin = None  # shape [B, V]

    def _ensure_shapes(self, V: int):
        if self.V is None:
            self.V = V
        if self.sum_probs is None:
            self.sum_probs = np.zeros((self.num_bins, self.V), dtype=float)
            self.counts = np.zeros(self.num_bins, dtype=int)

    def fit(self, probs_list: list[np.ndarray]):
        """
        probs_list: iterable of 1D arrays (sorted desc), one per simulated sample.
        """
        for p in probs_list:
            V = len(p)
            self._ensure_shapes(V)
            # enforce same V (pad/truncate if needed)
            if V != self.V:
                if V < self.V:
                    q = np.zeros(self.V, float); q[:V] = p
                else:
                    q = p[:self.V]
            else:
                q = p
            pmax = float(q[0])
            # bin index 0..num_bins-1
            b = np.digitize(pmax, self.bin_edges, right=False) - 1
            b = max(0, min(self.num_bins - 1, b))
            self.sum_probs[b] += q
            self.counts[b] += 1

        # averages (zeros if bin empty)
        self.avg_by_bin = np.zeros_like(self.sum_probs)
        nz = self.counts > 0
        self.avg_by_bin[nz] = self.sum_probs[nz] / self.counts[nz, None]
        return self

    def keep_count(self, probs: np.ndarray, p: float = 0.1) -> int:
        """
        For the bin of this sample's p_max, return the calibrated rank threshold.
        """
        assert self.avg_by_bin is not None, "Call fit() first with your simulated logits."
        pmax = float(probs[0])
        b = np.digitize(pmax, self.bin_edges, right=False) - 1
        b = max(0, min(self.num_bins - 1, b))
        if self.counts[b] == 0:
            return 1  # fallback if no data in bin
        avg = self.avg_by_bin[b]  # shape [V]
        idx = np.where(avg >= p)[0]
        return int(idx.max() + 1) if idx.size > 0 else 1


def make_keep_count_calibrated_topk(probs_list: list[np.ndarray], num_bins: int = 10):
    """
    Convenience factory: fit once, get a drop-in function keep_count_calibrated_topk(probs, p)
    suitable for your sampler pipeline.
    """
    cal = CalibratedTopK(num_bins=num_bins).fit(probs_list)
    def keep_count_calibrated_topk(probs: np.ndarray, p: float = 0.1) -> int:
        return cal.keep_count(probs, p=p)
    return keep_count_calibrated_topk

# ---------------------- Simulation ----------------------
V = 15  # nominal vocab size
R = V
y_plot = np.linspace(0.0, 1.0, 301)
Y_EPS = 1e-13

def s_of_y(y, s_min=0.3, s_span=3.0):
    return s_min + s_span * y

def simulate_probs_for_y(y, V=500):
    y = float(np.clip(y, 1.0/V + 1e-9, 1.0 - 1e-9))  # keep feasible and stable

    # Base sharpness that increases with y (peaky at high y)
    s_base = 0.3 + 3.0 * y

    # Finite geometric tail sum S(s) over ranks 2..V (weights e^{-s * j}, j=0..V-2)
    def tail_sum(s):
        q = np.exp(-s)
        # sum_{j=0}^{V-2} q^j
        return (1.0 - q**(V-1)) / (1.0 - q + 1e-18)

    need = (1.0 - y) / y          # minimal sum so that A <= y
    S_base = tail_sum(s_base)

    # If base tail can't hold the mass under the cap y, flatten (reduce s) via bisection
    if S_base < need:
        lo, hi = 1e-6, s_base
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if tail_sum(mid) >= need:
                lo = mid
            else:
                hi = mid
        s = lo
    else:
        s = s_base

    # Build probabilities: p1=y, tail decays with s and sums to (1-y), with each tail term <= y
    ranks = np.arange(V - 1)
    w = np.exp(-s * ranks)
    A = (1.0 - y) / w.sum()
    tail = A * w
    # By construction A <= y (equality only when S == need); keep it strictly below:
    tail = np.minimum(tail, np.nextafter(y, 0.0))

    p = np.empty(V, dtype=float)
    p[0] = y
    p[1:] = tail
    # Already nonincreasing, but sort just in case of numerical ties
    return np.sort(p)[::-1]


# ---------------------- Samplers ----------------------
def keep_count_topk(probs, k=10):
    return min(k, probs.size)

def keep_count_topp(probs, p_threshold=0.9):
    csum = np.cumsum(probs)
    idx = int(np.searchsorted(csum, p_threshold, side="left"))
    return min(idx + 1, probs.size)

def keep_count_minp_rel(probs, min_p_rel=0.05):
    if probs.size == 0:
        return 0
    p_max = float(probs[0])
    return int(np.sum(probs > (min_p_rel * p_max)))

def keep_count_epsilon_abs(probs, epsilon_abs=1e-3):
    p_max = float(probs[0])
    if probs.size == 0:
        return 0
    return int(np.sum(probs >= epsilon_abs)) if p_max >= epsilon_abs else 1

#def keep_count_calibrated_topk(probs, p=0.1):
#    p_max = float(probs[0])
#    return int(np.sum(probs >= p)) if p_max >= p else 1

def keep_count_greedy_threshold(probs, p=0.3):
    if probs.size == 0:
        return 0
    p_max = float(probs[0])
    return probs.size if p_max >= p else 1

# New: keep everything (no sampling)
def keep_count_all(probs):
    return probs.size

# New: Top-k + Top-p (stricter of both)
def keep_count_topk_topp(probs, k=10, p_threshold=0.9):
    return min(keep_count_topk(probs, k=k), keep_count_topp(probs, p_threshold=p_threshold))

# ---------------------- Compute boundaries ----------------------
def boundary_from_sampler(keep_fn, **kwargs):
    kept_fracs = []
    for y in y_plot:
        probs = simulate_probs_for_y(y)
        k = keep_fn(probs, **kwargs)
        kept_fracs.append(k / R)
    return np.array(kept_fracs)

cal_set = [simulate_probs_for_y(y) for y in y_plot]  # sorted descending

# Create the calibrated function
keep_count_calibrated_topk = make_keep_count_calibrated_topk(cal_set, num_bins=10)

# Then use it in boundary_from_sampler(...)
frac_ctopk = boundary_from_sampler(keep_count_calibrated_topk, p=0.05)

frac_topk   = boundary_from_sampler(keep_fn=keep_count_topk, k=10)
frac_topp   = boundary_from_sampler(keep_fn=keep_count_topp, p_threshold=0.9)
frac_minp   = boundary_from_sampler(keep_fn=keep_count_minp_rel, min_p_rel=0.1)
frac_eps    = boundary_from_sampler(keep_fn=keep_count_epsilon_abs, epsilon_abs=0.05)
frac_gthres = boundary_from_sampler(keep_fn=keep_count_greedy_threshold, p=0.3)
#frac_ctopk  = boundary_from_sampler(keep_fn=keep_count_calibrated_topk, p=0.05)

# New boundaries
frac_all    = boundary_from_sampler(keep_fn=keep_count_all)
frac_combo  = boundary_from_sampler(keep_fn=keep_count_topk_topp, k=10, p_threshold=0.9)
frac_cal_eps = boundary_from_sampler(keep_count_epsilon_abs, epsilon_abs=0.05)

# ---------------------- Plot panels ----------------------
# Top row: 6 subplots
# Bottom row: 6 equal-width slots; we will plot in the last three
from matplotlib.gridspec import GridSpec

titles_top = [
    "No restrictions",
    "Top-k",
    "Top-p",
    "Min-p",
    "$\\epsilon$-sampling",
    "   Top-k + Top-p",
]
fracs_top = [
    frac_all,
    frac_topk,
    frac_topp,
    frac_minp,
    frac_eps,
    frac_combo,
]

titles_bottom = [
    "Greedy-Threshold",
    "       Calibrated-TopK",
    "   Calibrated-$\\epsilon$",
]
fracs_bottom = [
    frac_gthres,
    frac_ctopk,
    frac_cal_eps,
]

fig = plt.figure(figsize=(12, 5.0), dpi=300)
gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.0)

# Subgrids for each row
gs_top = gs[0, :].subgridspec(1, 6)
gs_bottom = gs[1, :].subgridspec(1, 6)

axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(6)]
# Create six bottom axes for equal sizing; use the last three for our panels
axes_bottom_full = [fig.add_subplot(gs_bottom[0, i]) for i in range(6)]
axes_bottom_positions = [2, 3, 4]
axes_bottom = [axes_bottom_full[i] for i in axes_bottom_positions]
# Hide unused bottom slots
for idx in range(6):
    if idx not in axes_bottom_positions:
        axes_bottom_full[idx].set_visible(False)

# Plot top row
for i, ax in enumerate(axes_top):
    # Remove first figure in the first row
    if i == 0:
        ax.set_visible(False)
        continue
    ax.fill_betweenx(y_plot, 0, fracs_top[i], alpha=0.4)
    ax.plot(fracs_top[i], y_plot, linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=7, length=2, pad=2)
    # Show "Top/Tail" only on first and last USED of the first row
    if i in (1, 1):
        ax.set_xticks([0, 1], labels=["Largest\ntoken", "Smallest\ntoken"], fontsize=12)
    else:
        ax.set_xticks([0, 1], labels=["", ""]) 
    ax.set_yticks([])
    ax.grid(axis='y', linestyle=":", linewidth=0.3, alpha=0.5)
    ax.set_title(titles_top[i], fontsize=14, pad=8)

# Plot bottom row
for j, ax in enumerate(axes_bottom):
    # Green fill and boundary for the last two bottom figures
    if j in (1, 2):
        ax.fill_betweenx(y_plot, 0, fracs_bottom[j], alpha=0.4, color='green')
        ax.plot(fracs_bottom[j], y_plot, linewidth=0.8, color='green')
    else:
        ax.fill_betweenx(y_plot, 0, fracs_bottom[j], alpha=0.4)
        ax.plot(fracs_bottom[j], y_plot, linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=7, length=2, pad=1)
    # Keep xticks consistent
    if j in (0, 2):
        ax.set_xticks([0, 1], labels=["", ""], fontsize=12)
    else:
        ax.set_xticks([0, 1], labels=["", ""]) 
    ax.set_yticks([])
    ax.grid(axis='y', linestyle=":", linewidth=0.3, alpha=0.5)
    ax.set_title(titles_bottom[j], fontsize=14, pad=10)

# Attach y-label to the first visible top axis (index 1)
axes_top[1].set_ylabel("Confidence", fontsize=14, labelpad=2)

# Legend for boundary meanings
legend_handles = [
    Patch(facecolor='C0', edgecolor='#333563', alpha=0.4, label='token probabilities'),
    Patch(facecolor='green', edgecolor='#1e5e0f', alpha=0.4, label='token correctness'),
]
fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.75, 0.30), fontsize=13, frameon=False, title='Truncation threshold source', title_fontsize=14)

plt.tight_layout()

fig.canvas.draw()
# Compute right edge of epsilon (top row, col 5) and left edge of Greedy (bottom row, first used col)
ax_eps_top = axes_top[4]
ax_greedy_bottom = axes_bottom[0]
x_right_eps = ax_eps_top.get_position().x1   # right edge of Epsilon panel (top row)
x_left_gth  = ax_greedy_bottom.get_position().x0   # left edge of Greedy panel (bottom row)

# Place "Ours" label to the LEFT, horizontally
fig.text(0.33, 0.3, "Ours", fontsize=19, va="center", ha="right", color='black', fontweight='bold')

fig.savefig("figure0.pdf", bbox_inches='tight')

plt.show()
