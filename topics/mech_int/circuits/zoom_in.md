## Zoom In: An Introduction to Circuits

The post that started it all, the mechanistic interpretability analysis into CNNs and ImageNet, analysis into how CNNs gradually build line detectors, curve detectors, building into polysemantic neurons. Re-reading this post now, I can clearly see that some of the statements made in this blog post were overstatements. Statements such as, "you can literally read algorithms off the weights of the neural networks" or "there's often a symmetry around the circuitry", made me very excited. But after having done some experiments of my own and seeing what OGs of the field/the field itself is saying now: these findings are difficult to generalize to LLMs.

## The Problem

CNNs and how they work with images are well known. In fact, computer vision has been loud in saying how it is neurons interact with each other to work here. However, transformers are different beasts altogether. We're no longer trying to find interpretability within one layer of one neural network. In the example of LLaMa-70B, there are 80 layers of repeating 64 query attention heads (but only 8 KV layers because they use Grouped Query Attention i.e. GQA). So, the number of neurons you would need to profile to replicate what this post did (which is exactly what I wanted to do in the beginning) is... well, very challenging. Like, extremely challenging. And that doesn't even lead to a satisfactory answer because people like Neel Nanda have already tried to do so, albeit for a much smaller model i.e. GPT-2 and other models supported now too: [Neuroscope](https://neuroscope.io/)

## Why study this?

Although most of the things discussed in this post are very computer vision-y in nature (edge detectors, position invariant object detection, high-low frequency detectors), it is interesting to keep in mind all the parallels, especially as I am attempting to read most interesting developments in the field over the last couple years, as I attempt to create the next idea for [bizzaro world](https://github.com/Azazel5/Bizzaro-World), as I have just finished experimentation on Gemma27B-IT.

## Position Invariance

Interestingly, there is some aspect of whatever it is that LLMs and suppression heads do here. For instance, sometimes, some collection of neurons catch an object pointed left and the other collection will find it pointed right. These are "unioned" together is what they claim, but they themselves say there are some XOR like properties present. Sloppy. What they mean is they have discovered something in between union and XOR: the two pathways detect left and right curves independently, which is union-like in that either can activate the unit, but then they inhibit each other, which means when both are present simultaneously the output is suppressed relative to either alone. That's not a union, that's closer to a soft XOR or a winner-take-all dynamic.

This actually maps onto something directly relevant to your own findings — your L46H4 and L46H5 in Gemma 12B sitting adjacent with opposing signs is structurally analogous. One load-bearing, one hyper-suppressive, in the same layer. That's potentially the same push-pull inhibition dynamic the Distill paper is describing here, just at the attention head level rather than the neuron level in a CNN. 

> The load-bearing and suppressive heads at the same layer aren't independent, they're implementing a push-pull dynamic that has the same functional signature as the CNN circuit's soft XOR — strong response to one signal, suppression when both compete. That is, prior work in CNN circuits identified push-pull inhibition as a mechanism for invariant feature detection ([distill zoom-in](https://distill.pub/2020/circuits/zoom-in/#claim-2-curves)). We observe an analogous structure at the attention head level in Gemma 12B-IT — L46H4 and L46H5 at the same layer with opposing signs of -0.61 and +1.11, where the suppressive head's magnitude exceeds the normalized scale ceiling, suggesting active competition rather than passive absence of signal. Whether this implements a similar soft XOR dynamic at the circuit level is an open question, but the structural parallel is striking. 

## Superposition

Some neurons are noted near the middle layers of InceptionV1 which looks at multiple different things at ones: polysemanticity via superposition! At this point the researchers are suprised that the network spreads it spreads feature over a number of neurons that are doing something else (detecting dogs), rather than creating pure car detectors at the next layer.

This seems intuitive enough to me: almost as if the CNN is realizing the best way to be able to see is to combine knowledge across domains, not hyper-specialize, be a generalist. Be a polymath. Be a liberal arts learner, across a variety of knowledge because perhaps an unexpected piece of knowledge from somewhere else can come in handy?! **Polisemantic neurons are deliberate.**

GD doesn't distinguish between "I'm doing this to save space" and "I'm doing this because it works better." It just minimizes loss. If polysemantic representations both compress efficiently AND generalize better, GD would find them for both reasons simultaneously and there's no way to decompose the contribution of each from the outside. The field has been telling one half of that story as if it were the whole story.
The stronger version of your claim is actually falsifiable in principle: if polysemanticity were purely compression, you'd expect models trained with abundant capacity (vastly overparameterized, no bottleneck pressure) to trend toward monosemanticity. If they don't — if superposition persists even when there's no resource pressure forcing it — that's evidence for your generalization hypothesis. That experiment hasn't been run cleanly as far as I know, which means the compression story isn't proven, it's just assumed.

> Experiment idea: The core measurement problem: polysemanticity means one neuron/direction responds to multiple unrelated features. So a metric needs to capture: how many distinct concepts activate a given unit, and how unrelated those concepts are to each other.

Candidate metrics, ranked by tractability:

1. SAE feature entropy (most tractable for you right now)

For each neuron, run your prompt battery through a pretrained Gemma Scope SAE, record which SAE features co-activate with that neuron across prompts, compute the entropy of that distribution. A monosemantic neuron has low entropy — one feature dominates. A polysemantic neuron has high entropy — many unrelated features activate it roughly equally. You already have Gemma Scope access for all three models. This is runnable this week. 

For each neuron n in layer l:
    record which SAE features fire when neuron n fires
    compute H = -sum(p_i * log(p_i)) over co-activating features
    mean H across all neurons = model-level polysemanticity score

2. Participation ratio

From physics — measures how many dimensions a vector "effectively" uses. Applied to a neuron's activation pattern across a dataset:
python# activation vector a across N prompts

# PR = (sum(a_i))^2 / (N * sum(a_i^2))
# PR near 1/N = monosemantic (one prompt dominates)
# PR near 1 = polysemantic (all prompts activate it equally)

3. Concept clustering score

Run your prompt battery, which already spans 20 categories. For each neuron, record which categories activate it above threshold. A monosemantic neuron fires for 1 category. A polysemantic neuron fires for many. Compute mean number of categories per neuron across the model. Coarse but interpretable and requires no SAE.

The experiment as you've framed it:

If the score stays flat or increases with scale rather than decreasing, that's your evidence. The compression hypothesis predicts it should decrease — more parameters means less pressure to pack features together. Your generalization hypothesis predicts it stays constant or increases — the model keeps using superposition because it's functionally better, not because it has to.

One thing to nail before running it: you need to measure polysemanticity at matched layers — same relative depth across models, not same absolute layer. Otherwise you're comparing early-layer neurons in 2B against late-layer neurons in 27B, which aren't comparable. Use your ~79-88% relative depth finding as the anchor — measure polysemanticity at the circuit-critical layers you've already identified.

The relative comparisions are important because let's consider the UFC analogy. Jon Jones at heavyweight and Demetrious Johnson at bantamweight can't be compared by absolute weight lifted or absolute striking power — but pound-for-pound you normalize by weight class and suddenly the comparison is meaningful. Same principle: absolute layer 38 in a 48-layer model and absolute layer 38 in a 62-layer model are completely different things positionally, but layer 79% in both models is the same relative computational stage.

And the analogy actually runs deeper than just normalization. The pound-for-pound argument is interesting precisely because it reveals something about the underlying athlete independent of the weight class context — just like measuring polysemanticity at matched relative depth reveals something about the underlying circuit organization independent of absolute model size. You're controlling for the confound to isolate the variable of interest, which is scale itself.

This framing is also worth one sentence in your methods section when you introduce the relative depth comparison — it's the kind of intuitive anchor that makes a reviewer immediately understand why you're normalizing this way rather than using absolute layers. Most mech interp papers just do it without explaining why, which leaves readers who aren't deep in the field slightly confused. Naming the principle explicitly — "we compare circuits at matched relative depth to control for differences in absolute model size, analogous to pound-for-pound normalization in athletic comparison" — is clear, defensible, and memorable.

This is a clean, novel, falsifiable experiment that connects directly to the existing BizzaroWorld infrastructure and Gemma Scope access. It's also a second paper, not a BizzaroWorld extension — scope it separately. But the concept clustering score (option 3) using the existing 20-category prompt battery is runnable in days with no new infrastructure and gives you a preliminary result to cite even in the current paper's future work section.

The things the authors have noted cropped up over and over in their analysis, which cannot be understated, even though they OVERstated certain things, in retrospect (while comparing it to LLMs though! I am claiming nothing in what they said about CNNs. And as we have seen, all these results do set the foundations for the analysis going forward in LLMs too, so let's learn from this!). 

## Universality

First layer of vision models typically learn Gabor Filters, which is essentially a mathematical tool that detects edges at a specific orientation and frequency in an image — think of it as a spotlight that only reacts when it sees a stripe of a particular width running in a particular direction. It combines a sine wave (which creates alternating light and dark bands) with a Gaussian envelope (which focuses the detector on a local region rather than the whole image), so the result is a localized, orientation-sensitive edge detector. When a CNN's first layer learns Gabor-like filters from scratch, it's discovering that detecting oriented edges at various angles and scales is the most useful first step for understanding visual input — which turns out to be exactly what the early visual cortex (V1) in biological brains does too, a convergence that suggests this isn't just one way to process images but possibly the optimal way given the statistical structure of natural visual scenes.

The researchers here are trying to demonstrate that there is some natural phenomena going on; biological neural networks also operate in a similar way, detecting Gabor Filters, etc, which is compelling first grade evidence towards what they are trying to systematically prove, although they themselves admit that they don't have enough evidence. But if we were able to find something like the periodic table for visual features, that would be quite something. Unfortunately, it has been 6 years since the claim, and we haven't yet done it. Progress remains.