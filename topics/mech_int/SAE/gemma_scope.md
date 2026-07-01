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

SAEs are way more expensive and difficult to train, as compared to steering vectors or probing. On top of that, [transcoders may be more interpretable](https://arxiv.org/html/2501.18823v2), but they'd need to be trained from scratch, which I will absolutely do. An idea for later.

Given activations from a language model, SAEs decomposes and reconstructs them uses the encoder-decoder architecture. This decomposition is made sparse and non-zero by the choice of the activation function and regularization. The bicken et al initial paper used ReLU and L1 regularization to encourage sparsity. There are other successful methods to do the same, and the one that is followed in this paper is JumpReLU. 

## JumpReLU SAEs

This uses a different activation function and the L0 penalty. There is some hyperparameter tuning to be done here, like in all good ML, for finding the appropriate $\epsilon$ value for the parameters of the model. The training data are activations generated from the same distribution of the pretraining text that the Gemma models are trained on (with the exception of the IT models).

In the Gemma scope paper, the SAEs have been trained at particular locations, such as post-attention head or MLP layer, at certain locations, including an aggregated attention head SAE. Some transcoders have been trained as well. This is actually pretty interestingly rigorous, as described by the picture below:

![alt text](images/residual_saes.png)

> Megatron sharding seems to be a SOTA way to use maximal tensor parallelism!

It is tough to evaluate the quality of the SAEs learned latents and it's an ongoign area of research and debate. it seems simple reconstruction error isn't sufficient because reconstruction error and interpretability are two different axes that can move independently. In fact, the [transcoder paper](https://arxiv.org/html/2501.18823v2) is direct evidence of this. Sparser latents are generally more interpretable, but higher sparsity also tends to increase reconstruction error — that's the fundamental tradeoff. 

You can trivially minimize reconstruction error by being dense, not sparse. An SAE with enough latents and weak sparsity penalty will reconstruct activations almost perfectly — it's basically just learning identity. But that defeats the entire purpose, which is decomposing superposed polysemantic activations into individually meaningful, monosemantic features. Perfect reconstruction with 10,000 active latents per token tells you nothing interpretable; it's just a fancy autoencoder, not a feature dictionary.

Reconstruction error doesn't tell you whether individual features mean anything. You could have low reconstruction error with features that are themselves still polysemantic — multiple unrelated concepts packed into a single latent, with the SAE's decoder compensating by adjusting magnitudes. The reconstruction comes out fine even though the decomposition completely failed at its actual job, which is monosemanticity.

The transcoder paper references this directly: the reconstruction error of a sparse coder can be viewed as "dark matter" containing features not captured by the latents. Some residual error reflects real missing structure, but some is just noise or genuinely non-feature-like computation the model performs — so even using reconstruction error as a proxy for "are we missing features" is unreliable, since you can't distinguish meaningful gaps from irreducible noise just by looking at the error magnitude.

Feature absorption is a separate failure mode that low reconstruction error doesn't catch. This is exactly what the transcoder paper measures separately: a more general feature like "starts with the letter L" can absorb into a more specific feature like "the token lion," preventing the general feature from firing where it intuitively should. The SAE reconstructs fine in both cases, but the conceptual decomposition is broken — you'd interpret the features wrong even though numerically the autoencoder is doing its job.

This is why the field developed separate evaluation axes — autointerp scores (does an LLM, shown activating examples, correctly guess what the feature means), sparse probing (does the encoder's representation help a downstream classifier), absorption scores, and the reconstruction-vs-interpretability Pareto curve itself, rather than any single number. The transcoder paper's whole methodology — fuzzing, detection, simulation scores, plus CE loss increase, plus absorption, all reported separately — exists precisely because no single metric, including reconstruction error, captures whether an SAE is actually doing useful interpretability work. You need multiple imperfect proxies triangulated together, which is itself a sign the field hasn't converged on ground truth for "what makes a good decomposition" yet.

If this is the case, this paper's mode of evaluation has a huge problem, focusing solely on reconstruction error through multiple angles: reconstruction-fidelity metrics. They're not redundant (delta LM loss captures causal/functional impact that FVU misses, which is itself a meaningful distinction), but they're still both answering the same underlying question: "how well does this SAE reconstruct what was there," not "are the individual features it found actually monosemantic and interpretable."

## Results

Reconstruction error itself is higher on the residual stream SAEs compared to the individual attention head or MLP sublayer SAEs. This result was also validated by the initial activation patching experiments on BizzaroWorld, where it was measured that the residual stream dominates factual recall, often times 40x as much, compared to the attention/MLP sublayers.

Wider SAEs reconstruct the original LLM activations the best, but Bricken et a. found a phenomena which they called **feature splitting**, which occurs when singular latents in narrow SAEs are split inot multiple *specialized* latents in wider SAEs, which is not preferred.

The actual interpretability of the discovered latents were, after all, analyzed, but not in a way that is satisfactory. They used human and LLM explanation. [This paper](https://arxiv.org/pdf/2407.14435) was the DeepMind paper from which inspiration is drawn for this. And this is another interesting one that talks about [the composition of features in NNs](https://arxiv.org/pdf/2407.14662).

> In AI and data science, FVU stands for Fraction of Variance Unexplained. It is a statistical metric used to measure how much of the variation in a dataset is not accounted for by a machine learning model or an algorithmic process. A lower FVU means the model is accurately capturing the underlying data.

> Interestingly, reducing the models' floating point precision at inference time is a pretty good tactic that has negligible impact.

Splicing the Gemma 9B based SAEs to reconstruct activations generated from Gemma 9B-IT is pretty darn successful, resulting almost no difference between the delta loss curves 

