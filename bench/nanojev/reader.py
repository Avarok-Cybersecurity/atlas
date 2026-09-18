# SPDX-License-Identifier: AGPL-3.0-only

"""The zero-decode read: one forward pass, a distribution over the candidates.

THE PRIMITIVE
-------------
A decision model does not generate. The prompt is arranged so that the answer
would occupy exactly one token position, the model is run forward ONCE, and
the distribution is read off the logits at that position restricted to the
candidate label ids. No sampling loop, no output tokens, no parsing.

This file provides both things the experiment needs from the backbone, from
the same forward pass:

  * `label_logits`  -- the untuned baseline. Scores the candidate tokens with
                       the model's own LM head. This is the floor any trained
                       head has to beat, and it is what "untuned Qwen3-0.6B"
                       means in NanoJev's tables.
  * `hidden`        -- the last hidden state at the answer position, which the
                       trained decision head consumes. Cached once so head
                       training never re-runs the backbone.

THE SINGLE-TOKEN REQUIREMENT
----------------------------
Restricting logits to candidate ids is only meaningful if each candidate is
ONE token and the candidates are mutually distinct. Otherwise "north" and
"none" might share a first token, the read would score a prefix rather than a
label, and the resulting distribution would be quietly wrong -- it would still
sum to one and still look like an answer.

`resolve_labels` therefore checks this up front, against the real tokenizer,
and raises rather than degrading. The same discipline appears in the
DiffusionGemma structured-read server, which refuses a schema whose labels do
not occupy one shared slot. A client with multi-token options is expected to
map them to single-token stand-ins ("moderation_spam" -> "A").
"""

from __future__ import annotations

import torch


class LabelSet:
    """Candidate labels resolved to single, distinct token ids."""

    def __init__(self, labels: list[str], ids: list[int]):
        self.labels = labels
        self.ids = ids

    def __len__(self) -> int:
        return len(self.labels)


def resolve_labels(tok, labels: list[str], prefix: str = " ") -> LabelSet:
    """Map each label to one token id, or raise.

    `prefix` is the character that will actually precede the label in the
    prompt. It matters: BPE tokenizers give " yes" and "yes" different ids,
    and scoring the wrong one silently measures a token the prompt will never
    put there.
    """
    ids = []
    for lab in labels:
        enc = tok.encode(prefix + lab, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(
                f"label {lab!r} encodes to {len(enc)} tokens ({enc}) with prefix {prefix!r}; "
                "the read needs one token per label -- map it to a single-token stand-in"
            )
        ids.append(enc[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"labels {labels} do not map to distinct ids: {ids}")
    return LabelSet(labels, ids)


PROMPT = (
    "You are judging a gridworld observation.\n"
    "'#' is a wall, '.' is open, 'A' is the agent.\n"
    "The agent's actuator is unreliable: it follows the requested direction "
    "with probability {rho:.2f}, otherwise it moves in a uniformly random direction.\n\n"
    "{view}\n\n"
    "Question: {question}\n"
    "Options:{options}\n"
    "Answer:"
)


def build_prompt(ex: dict) -> str:
    """The prompt whose next token is the answer label.

    The options are listed because the model must know the candidate set to
    put mass on it, and the trailing "Answer:" with no space makes the answer
    token itself carry the leading space -- matching `resolve_labels(prefix=" ")`.
    """
    options = "".join(f" {lab}" for lab in ex["labels"])
    return PROMPT.format(rho=ex["rho"], view=ex["view"], question=ex["question"], options=options)


class Reader:
    """A frozen backbone that answers by reading, not by generating."""

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, device_map=device
        )
        self.model.eval()
        self.device = device
        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def read_batch(self, examples: list[dict], want_hidden: bool = True):
        """One forward pass over a batch.

        Returns `(label_logits, hidden)`:
          label_logits[i] -- the LM head's scores for example i's label ids
          hidden[i]       -- the last hidden state at the answer position

        Left-padding is used so the answer position is the LAST column for
        every row, which makes both reads a single index rather than a gather
        over per-row lengths. Right-padding here would read pad positions and
        produce a plausible, wrong distribution.
        """
        prompts = [build_prompt(e) for e in examples]
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)

        out = self.model(**enc, output_hidden_states=want_hidden)
        # -1 is the answer position thanks to left padding.
        last_logits = out.logits[:, -1, :].float()
        hidden = out.hidden_states[-1][:, -1, :].float() if want_hidden else None

        label_logits = []
        for i, ex in enumerate(examples):
            ls = resolve_labels(self.tok, ex["labels"])
            label_logits.append(last_logits[i, ls.ids])
        return label_logits, hidden

    @torch.no_grad()
    def baseline_probs(self, examples: list[dict]) -> list[list[float]]:
        """The untuned read: softmax of the LM head over the candidate ids."""
        label_logits, _ = self.read_batch(examples, want_hidden=False)
        return [torch.softmax(x, dim=-1).tolist() for x in label_logits]
