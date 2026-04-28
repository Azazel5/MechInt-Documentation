"""
Minimal example (Jupyter recommended).

Run inside a notebook cell:

    %pip install -r requirements.txt
    python -m circuit_discovery.experiments.innovations.example_attention_head_inspector

This uses TransformerLens to grab a single head's attention pattern and opens the inspector.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from circuit_discovery.experiments.innovations.head_inspector import make_attention_head_inspector


def main() -> None:
    model = HookedTransformer.from_pretrained("gpt2-small")
    prompt = "When John and Mary went to the shops, John gave the bag to"
    tokens = model.to_tokens(prompt, prepend_bos=True)
    str_tokens = model.to_str_tokens(tokens[0])

    _, cache = model.run_with_cache(tokens)

    layer = 9
    head = 6
    pattern = cache["pattern", layer][0, head]  # [q, k]

    # Example "scalar panel": log-prob of top-1 at each position (optional)
    logits = model(tokens)[0]  # [pos, vocab]
    logprobs = logits.log_softmax(dim=-1)
    top1_logprob = logprobs.max(dim=-1).values.detach().cpu().numpy()

    make_attention_head_inspector(
        tokens=str_tokens,
        pattern_qk=pattern,
        title=f"GPT2-small L{layer}H{head} attention",
        extra_scalars={"top1_logprob": top1_logprob},
    ).show()


if __name__ == "__main__":
    # This will display nicely only in Jupyter; in pure python it will still construct widgets,
    # but you won't see them unless your environment renders ipywidgets.
    main()

