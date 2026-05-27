"""srna_layer.py — v5 Path D sRNA attention bias module.

Mirrors v4 `EngineeringEdgeBias` semantics but with **inhibitory default**:
  - sRNA→target edges default to negative bias (init_bias = -0.5)
  - Override attention mask at sRNA positions (do not add to -inf entries)
  - Per-edge learnable scalar so the model can refine each interaction

Edge polarity convention:
  - sRNAs in `srna_target_edges_v1.tsv` carry an `effect` column ('-' or '+').
    For '-' edges init = -|init_bias| (silencing).
    For '+' edges init = +|init_bias| (activation; e.g. RprA -> rpoS).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SrnaEdgeBias(nn.Module):
    def __init__(self,
                 edges: list[tuple[int, int]],
                 effects: list[str] | None = None,
                 init_magnitude: float = 0.5,
                 n_genes: int = 4722):
        super().__init__()
        self.edges = list(edges)
        self.n_genes = n_genes
        self.is_active = True

        n = len(self.edges)
        if n > 0:
            if effects is None:
                effects = ["-"] * n
            assert len(effects) == n, "effects length must match edges"
            init = torch.tensor(
                [(-init_magnitude if e == "-" else +init_magnitude)
                 for e in effects], dtype=torch.float32)
            self.biases = nn.Parameter(init)
            i_idx = torch.tensor([e[0] for e in self.edges], dtype=torch.long)
            j_idx = torch.tensor([e[1] for e in self.edges], dtype=torch.long)
        else:
            self.biases = nn.Parameter(torch.zeros(0))
            i_idx = torch.empty(0, dtype=torch.long)
            j_idx = torch.empty(0, dtype=torch.long)
        self.register_buffer("i_idx", i_idx)
        self.register_buffer("j_idx", j_idx)

    def toggle(self, on: bool):
        self.is_active = on

    def get_bias_and_override(self, device):
        zero = torch.zeros(self.n_genes, self.n_genes, device=device)
        if (not self.is_active) or len(self.edges) == 0:
            return zero, torch.zeros_like(zero, dtype=torch.bool)
        bias = zero.clone()
        bias.index_put_((self.i_idx, self.j_idx), self.biases, accumulate=False)
        override = torch.zeros_like(zero, dtype=torch.bool)
        override[self.i_idx, self.j_idx] = True
        return bias, override

    def edge_table_str(self) -> str:
        lines = []
        for k, (i, j) in enumerate(self.edges):
            b = self.biases[k].item()
            lines.append(f"  srna edge {k}: ({i:>5} -> {j:>5}) bias = {b:+.4f}")
        return "\n".join(lines)


def make_srna_pre_hook(srna_bias: SrnaEdgeBias):
    diag = {"called": False}

    def pre_hook(module, args, kwargs):
        if "attn_mask" in kwargs:
            existing = kwargs.pop("attn_mask")
        elif len(args) >= 2:
            existing = args[1]
            args = (args[0],) + args[2:]
        else:
            existing = None

        device = args[0].device
        bias, override = srna_bias.get_bias_and_override(device)

        if existing is None:
            new_mask = bias
        else:
            new_mask = torch.where(override, bias, existing)

        if not diag["called"]:
            diag["called"] = True
            n_override = int(override.sum().item())
            print(f"[srna hook] first call: n_edges_active={n_override}, "
                  f"is_active={srna_bias.is_active}", flush=True)

        kwargs["attn_mask"] = new_mask
        return args, kwargs

    return pre_hook


def register_srna_hooks(transformer_layers, srna_bias: SrnaEdgeBias):
    """Attach the sRNA pre-hook to each layer's attn module.

    `transformer_layers` is an iterable of layer objects each exposing `.attn`.
    Returns list of hook handles for later removal.
    """
    handles = []
    pre_hook = make_srna_pre_hook(srna_bias)
    for layer in transformer_layers:
        h = layer.attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
        handles.append(h)
    return handles


class SrnaGate(nn.Module):
    """Multiplicative (in TPM) / additive (in log_tpm) sRNA post-prediction gate.

    For each (srna_idx, target_idx, effect) edge:
        log_target_pred += W_edge * sign(effect) * log_srna_pred

    On TPM scale this corresponds to target_tpm *= exp(W * sign * log_srna_tpm)
    which models sRNA-mediated mRNA stability modulation (Hfq mechanism).

    Per-edge learnable W_edge initialized to init_weight (positive). Negative
    polarity is carried by `signs` buffer so sign flips do not require negating
    the weight itself (stays in a single sign for cleaner gradients).
    """

    def __init__(self,
                 edges: list[tuple[int, int]],
                 effects: list[str] | None,
                 n_genes: int,
                 init_weight: float = 0.5):
        super().__init__()
        self.edges = list(edges)
        self.n_genes = n_genes
        self.is_active = True
        n = len(self.edges)
        if effects is None:
            effects = ["-"] * n
        assert len(effects) == n

        if n > 0:
            self.weights = nn.Parameter(torch.full((n,), float(init_weight)))
            signs = torch.tensor(
                [-1.0 if e == "-" else 1.0 for e in effects],
                dtype=torch.float32)
            srna_idx = torch.tensor([e[0] for e in self.edges], dtype=torch.long)
            tgt_idx = torch.tensor([e[1] for e in self.edges], dtype=torch.long)
        else:
            self.weights = nn.Parameter(torch.zeros(0))
            signs = torch.zeros(0)
            srna_idx = torch.empty(0, dtype=torch.long)
            tgt_idx = torch.empty(0, dtype=torch.long)
        self.register_buffer("signs", signs)
        self.register_buffer("srna_idx", srna_idx)
        self.register_buffer("tgt_idx", tgt_idx)

    def toggle(self, on: bool):
        self.is_active = on

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """pred [B, G] log_tpm. Returns pred with sRNA modulation applied."""
        if (not self.is_active) or len(self.edges) == 0:
            return pred
        # Read sRNA predictions (use original pred, not modulated)
        srna_vals = pred.index_select(1, self.srna_idx)         # [B, E]
        # Compute per-edge contribution: w * sign * srna_val
        contribs = srna_vals * (self.weights * self.signs).unsqueeze(0)
        # Scatter-add to target positions (multiple edges to same target sum up)
        out = pred.clone()
        out.index_add_(1, self.tgt_idx, contribs)
        return out

    def edge_table_str(self) -> str:
        lines = []
        for k, (si, ti) in enumerate(self.edges):
            w = self.weights[k].item()
            s = self.signs[k].item()
            lines.append(f"  gate {k}: ({si:>5} -> {ti:>5}) "
                         f"w={w:+.3f} polarity={'+' if s > 0 else '-'}")
        return "\n".join(lines)
