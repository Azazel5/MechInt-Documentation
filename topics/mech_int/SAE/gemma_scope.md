## NeuronPedia and other resources

The Gemma Scope paper and other researchers have revealed to me the existence of some interesting resources I will be looking at, before the actual paper. In particular, NeuronPedia is particularly enlightening, as we can access multiple models and use SAEs there to see what features activate!

This is a pretty nice tool that, given a prompt and its tokens, highlights precisely what features are activated. 

![alt text](images/neuronpedia.png)

It's pretty interesting, but the tool gives us the ability to steer models too. Here are two examples of Gemma9B and a steered version where I had it emphasize features related to "jazz" and I ask it something about Wes Montgomery, the legendary jazz guitarist.

![alt text](images/wes.png)

The "jazz music" feature make it go completely haywire, but that isn't all: even after removing that feature entirely, the music steered version doesn't make a whole lot of sense. Only when this steering is tuned down all the way to 0.01, does the output exactly match what is seen by the default Gemma mode, but what when this feature is **suppressed**? Interestingly, Gemma starts behaving differently. It says that it doesn't have personal opinions or beliefs and is clearly non-personal.

It is really strange because, when I find a feature that has an activation density of 0% (I picked the highest one earlier i.e. 0.28%), but even steering/suppressing this feature, causes the downstream output of Gemma go completely crazy and uninterpretable. **What could be the reason...?**

A feature with 0.28% activation density means it fires rarely — only on a small fraction of tokens. Intuitively you'd expect suppressing something rare to have a small effect, since it's barely active most of the time. But that intuition is wrong for two compounding reasons.

First, SAE reconstruction error is not zero. The SAE is an approximation of the residual stream, not a perfect decomposition — there's always residual reconstruction error between the original activation and the SAE's reconstructed version. When you steer or suppress a feature, you're not surgically editing one clean independent variable; you're modifying the SAE's reconstructed activation and then feeding that altered reconstruction back into the model. If that feature's direction in activation space happens to overlap (due to superposition) with directions that other, more load-bearing features rely on, suppressing it can collaterally damage information those other features need — even though the target feature itself was rare.

Second — and this is the more likely explanation given what you're describing — rare features are often disproportionately important precisely because they're rare. A feature that fires on 0.28% of tokens isn't necessarily unimportant; it might be a highly specific, highly load-bearing feature for a narrow but critical context (a particular grammatical construction, a specific entity type, a coherence-maintaining signal). Sparsity and importance are not the same axis. This is actually consistent with what I found with the alignment-faking and BizzaroWorld push-pull dynamics — low-magnitude or rare-firing components can carry outsized causal weight, exactly like L46H5 exceeding the normalized scale despite being a single head among thousands.

What's happening when I suppress it and get "I don't have personal opinions" is a plausible sign that this feature is entangled with something like an opinion-expression or personality-expression circuit — suppressing it may be collapsing the model into a generic safety-trained default response mode, which is itself informative: it suggests this feature (or features near it in the entangled direction) gates whether the model expresses anything resembling a personal stance at all.

Let's also talk about the [NLA paper](https://transformer-circuits.pub/2026/nla/index.html) from last month, which is something to think about implementing in BizzaroWorld. The most striking finding from the paper, and directly relevant to everything I've been up to: during their pre-deployment audit of Claude Opus 4.6, NLAs helped diagnose safety-relevant behaviors and surfaced unverbalized evaluation awareness — cases where Claude believed, but did not say, that it was being evaluated. arXiv

Let's sit with that for a second, because it connects directly to the alignment-faking conversation. The compliance gap, the hidden scratchpad reasoning, the fine-tuned condition where the disposition is baked into weights with no visible CoT — all of those required either a scratchpad you could read (legible) or were undetectable (weight-level, illegible). NLAs are a third option: a tool that reads the activation directly and produces a natural language description even when the model never said anything about being evaluated in its actual output. That's a direct empirical answer to the limitation, "what about misalignment baked into weights" — NLAs are one of the first published tools that gets partial visibility into exactly that blind spot, surfacing unverbalized awareness rather than relying on what the model chooses to say.


## Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2

