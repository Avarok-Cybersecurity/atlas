# SPDX-License-Identifier: AGPL-3.0-only

"""The decision head: a hidden state and a candidate set in, a distribution out.

WHY NOT A CLASSIFIER
--------------------
The obvious head is a linear layer to a fixed number of classes. It cannot
work here. The contract is 2 to 255 candidates, supplied per request, whose
text is not known when the head is trained -- a fixed output width would pin
the model to one schema and make "dynamic candidates" false.

So the head SCORES candidates rather than enumerating them. The query
representation (the backbone's hidden state at the answer position) and each
candidate's representation (its token embedding, from the frozen input
embedding table) are projected into a shared space and compared:

    score_k = <W_q h, W_c e_k> / sqrt(d_proj) + b_k_bias

and the distribution is the softmax over candidates. The width of the output
is the number of candidates passed in, so the same trained head serves a
2-option boolean and a 200-option choice without retraining. A candidate the
head has never seen still gets a score, because what is scored is its
embedding, not its index.

The scaling by sqrt(d_proj) is not decoration: without it the dot products
grow with projection width and the softmax saturates before training starts,
which shows up as a head that outputs near-one-hot distributions from step one
and never calibrates.

ORDERED LEVELS
--------------
`score` questions have ordered labels ("low" < "medium" < "high"), and a plain
softmax treats them as unrelated. The head therefore optionally adds a
learned ordinal bias built from a scalar "severity" projection, which lets it
place mass on ADJACENT levels cheaply -- the failure mode of a categorical
head on ordered data is putting its second-most mass on a distant level, which
is both wrong and obviously wrong to a human reader.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DecisionHead(nn.Module):
    """Bilinear candidate scorer over a frozen backbone's hidden state."""

    def __init__(self, d_model: int, d_proj: int = 256, ordinal: bool = True):
        super().__init__()
        self.q = nn.Linear(d_model, d_proj, bias=False)
        self.c = nn.Linear(d_model, d_proj, bias=False)
        self.scale = 1.0 / math.sqrt(d_proj)
        # A per-candidate scalar from its own embedding: lets the head learn
        # "this label is a priori rare" without an index-keyed parameter.
        self.bias = nn.Linear(d_model, 1)
        self.ordinal = ordinal
        if ordinal:
            # One scalar per (query, candidate) capturing "how far along the
            # order" the query sits; combined with the candidate's position it
            # forms a distance penalty. Only used for ordered label sets.
            self.sev = nn.Linear(d_model, 1)
            self.ord_w = nn.Parameter(torch.tensor(0.0))

    def forward(self, h: torch.Tensor, emb: torch.Tensor, ordered: torch.Tensor | None = None):
        """
        h       : [B, d_model]      hidden state at the answer position
        emb     : [B, K, d_model]   candidate embeddings (padded with zeros)
        ordered : [B] bool          whether this row's labels are ordered
        returns : [B, K] logits (padded positions are -inf)
        """
        qh = self.q(h)                              # [B, P]
        ce = self.c(emb)                            # [B, K, P]
        logits = torch.einsum("bp,bkp->bk", qh, ce) * self.scale
        logits = logits + self.bias(emb).squeeze(-1)

        if self.ordinal and ordered is not None and ordered.any():
            b, k = logits.shape
            # Position of each candidate within its set, normalised to [0, 1].
            pos = torch.arange(k, device=logits.device, dtype=logits.dtype)
            pos = pos.unsqueeze(0).expand(b, k) / max(k - 1, 1)
            sev = torch.sigmoid(self.sev(h))        # [B, 1] in [0, 1]
            penalty = -self.ord_w.abs() * (pos - sev) ** 2
            logits = logits + torch.where(ordered.unsqueeze(1), penalty, torch.zeros_like(penalty))
        return logits


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log-softmax over the valid candidates only.

    Padded rows must not receive mass. Adding -inf before the softmax is the
    only version that is correct AND stable; zeroing probabilities afterwards
    and renormalising gives the same answer in exact arithmetic but leaks
    gradient into the pad columns.
    """
    logits = logits.masked_fill(~mask, float("-inf"))
    return torch.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# v2: per-candidate encoding, invariant by construction
# ---------------------------------------------------------------------------


class DecisionHeadV2(nn.Module):
    """Scores candidates that were each encoded as their OWN sequence.

    WHY THIS EXISTS
    ---------------
    `DecisionHead` scores candidates against one shared hidden state, produced
    from a prompt that lists the options in order. That inherits the prompt's
    ordering: measured on the untuned read, 78% of answers flip when the
    options are merely permuted, and slot 0 draws up to 1.65x its share of the
    probability mass. No amount of calibration repairs a model that is
    reporting presentation.

    NanoJev's answer is structural rather than statistical. Each candidate gets
    its own forward pass -- prefix + that one candidate -- so there is no "the
    option listed first" for the model to prefer, because the options never
    appear in a list. This head consumes those per-candidate states:

        score_k = w . LayerNorm(h_k)

    which is permutation-EQUIVARIANT by construction. Permuting the inputs
    permutes the outputs, and the softmax over them is unchanged, so the
    predicted distribution is permutation-INVARIANT. Not approximately, not
    after augmentation: exactly, by shape.

    THE SET TERM, AND WHY IT KEEPS THE PROPERTY
    -------------------------------------------
    Scoring candidates independently loses comparative information -- "which is
    safest" needs the alternatives. An optional multi-head attention over the
    candidate axis restores it, and preserves invariance because attention over
    a SET (with a key-padding mask and no positional encoding) is itself
    equivariant. The output projection starts at zero, so the set term begins
    as a no-op and has to earn its contribution.

    The cost is K forward passes of the backbone per question instead of one.
    That is the real price of the guarantee, and it is why serving this wants
    prefix reuse: every candidate shares the state and question tokens.
    """

    def __init__(self, d_model: int, d_set: int = 128, set_head: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.scalar = nn.Linear(d_model, 1)
        nn.init.normal_(self.scalar.weight, std=0.02)
        nn.init.zeros_(self.scalar.bias)
        self.set_head = set_head
        if set_head:
            # The +1 input feature carries log K, so the set term can behave
            # differently for a 2-way and a 200-way question without any
            # per-index parameter.
            self.project = nn.Linear(d_model + 1, d_set)
            self.attn = nn.MultiheadAttention(d_set, 4, dropout=0.0, batch_first=True)
            self.out = nn.Linear(d_set, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h    : [B, K, d_model]  per-candidate hidden states (padded)
        mask : [B, K] bool      which candidates are real
        returns [B, K] logits
        """
        h = self.norm(h)
        z = self.scalar(h).squeeze(-1).float()
        if self.set_head:
            k = h.shape[1]
            log_k = mask.sum(-1).clamp(min=1).float().log()[:, None, None].expand(-1, k, 1)
            u = self.project(torch.cat([h, log_k.to(h.dtype)], dim=-1))
            mixed, _ = self.attn(u, u, u, key_padding_mask=~mask, need_weights=False)
            z = z + self.out(torch.tanh(u + mixed)).squeeze(-1).float()
        return z
