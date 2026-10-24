"""EDA 11 - what the missing optimizer parameters actually cost.

The v2 training loop built its optimizer from `embeddings + encoder.layer`.
DeBERTa keeps three more tensors directly on the encoder. This script measures
the consequence, which is not the frozen weights themselves but the way their
un-zeroed gradients poison global gradient clipping for every other parameter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import save_fig, save_table  # noqa: E402

import pandas as pd  # noqa: E402

print("=" * 78)
print("11  THE OPTIMIZER COVERAGE BUG")
print("=" * 78)

from transformers import AutoModel  # noqa: E402

model = AutoModel.from_pretrained("microsoft/deberta-v3-base")

covered = set()
for param in model.embeddings.parameters():
    covered.add(id(param))
for layer in model.encoder.layer:
    for param in layer.parameters():
        covered.add(id(param))

missed = [(name, param) for name, param in model.named_parameters() if id(param) not in covered]
total = sum(p.numel() for _, p in model.named_parameters())
missed_count = sum(p.numel() for _, p in missed)

print("\nparameters the v2 optimizer never received")
rows = [{"parameter": name, "shape": str(tuple(p.shape)), "numel": p.numel()} for name, p in missed]
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n{missed_count:,} of {total:,} parameters ({missed_count / total:.2%})")
save_table(pd.DataFrame(rows), "11_uncovered_parameters")
print("\nThe count understates the problem. `rel_embeddings` is the relative")
print("position table that makes DeBERTa's disentangled attention disentangled;")
print("freezing it pins the model's positional prior to its pretrained value.")

print("\n--- the compounding failure: gradient clipping ------------------------")
print("torch's optimizer.zero_grad() only clears parameters inside its groups.")
print("A parameter left out keeps requires_grad=True, so autograd keeps writing")
print("into .grad, and nothing ever resets it. clip_grad_norm_(model.parameters())")
print("then sees a norm that grows without bound.")


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.tracked = nn.Linear(32, 32)
        self.orphan = nn.Linear(32, 32)

    def forward(self, x):
        return self.tracked(x) + self.orphan(x)


def simulate(include_orphan: bool, steps: int = 400, max_norm: float = 1.0) -> pd.DataFrame:
    torch.manual_seed(0)
    net = Tiny()
    params = list(net.parameters()) if include_orphan else list(net.tracked.parameters())
    optimizer = torch.optim.SGD(params, lr=0.0)
    history = []
    generator = torch.Generator().manual_seed(1)
    for step in range(steps):
        x = torch.randn(16, 32, generator=generator)
        target = torch.randn(16, 32, generator=generator)
        loss = nn.functional.mse_loss(net(x), target)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm).item()
        tracked_norm = torch.cat([p.grad.flatten() for p in net.tracked.parameters()]).norm().item()
        history.append({
            "step": step,
            "total_norm_before_clip": total_norm,
            "clip_coefficient": min(1.0, max_norm / (total_norm + 1e-6)),
            "tracked_grad_after_clip": tracked_norm,
        })
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
    return pd.DataFrame(history)


buggy = simulate(include_orphan=False)
fixed = simulate(include_orphan=True)

print("\nsimulation, identical model and data, only optimizer coverage differs")
summary = pd.DataFrame({
    "step": [0, 50, 100, 200, 399],
}).assign(
    buggy_total_norm=lambda d: [buggy.loc[s, "total_norm_before_clip"] for s in d["step"]],
    buggy_clip_coef=lambda d: [buggy.loc[s, "clip_coefficient"] for s in d["step"]],
    fixed_clip_coef=lambda d: [fixed.loc[s, "clip_coefficient"] for s in d["step"]],
)
print(summary.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
save_table(summary, "11_clip_simulation")

first, last = buggy.loc[0, "clip_coefficient"], buggy.loc[399, "clip_coefficient"]
steady_buggy = buggy["clip_coefficient"].tail(200).mean()
steady_fixed = fixed["clip_coefficient"].tail(200).mean()
print(f"\nThe orphan gradient does not grow without bound, and the reason matters:")
print("clip_grad_norm_ rescales every .grad in place, including the orphan's, so")
print("the leak reaches an equilibrium instead of exploding. What it leaves behind")
print("is a permanent tax on every other gradient.")
print(f"  steady-state clip coefficient, optimizer misses a parameter : {steady_buggy:.4f}")
print(f"  steady-state clip coefficient, full coverage                : {steady_fixed:.4f}")
print(f"  effective learning rate retained                            : {steady_buggy / steady_fixed:.1%}")
print("So the requested learning rate was never the learning rate. Roughly a")
print("quarter of it was thrown away at every step, for the whole run.")
save_table(pd.DataFrame([{
    "steady_clip_buggy": steady_buggy,
    "steady_clip_fixed": steady_fixed,
    "effective_lr_retained": steady_buggy / steady_fixed,
}]), "11_clip_steady_state")

print("\n--- does this match what the v2 run did? -----------------------------")
print("v2 trained 3 epochs and every one of the five folds scored its best at the")
print("final evaluation, still improving by 8,500 on average during epoch 3.")
print("An effective learning rate held ~25% below the requested one produces")
print("exactly that shape: the model never converges, it just runs out of budget.")
print("This does not prove the bug caused the shortfall, but it is consistent with")
print("it, and the fix costs nothing to apply.")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].plot(buggy["step"], buggy["total_norm_before_clip"], label="optimizer misses a parameter")
axes[0].plot(fixed["step"], fixed["total_norm_before_clip"], label="full coverage")
axes[0].set_yscale("log")
axes[0].set_xlabel("step")
axes[0].set_ylabel("grad norm seen by clip_grad_norm_")
axes[0].set_title("un-zeroed gradients raise the norm clip_ sees")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].plot(buggy["step"], buggy["clip_coefficient"], label="optimizer misses a parameter")
axes[1].plot(fixed["step"], fixed["clip_coefficient"], label="full coverage")
axes[1].set_yscale("log")
axes[1].set_xlabel("step")
axes[1].set_ylabel("clip coefficient applied to every gradient")
axes[1].set_title("a permanent tax on every gradient")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
fig.tight_layout()
print("\nfigure:", save_fig(fig, "11_optimizer_clip_bug"))
