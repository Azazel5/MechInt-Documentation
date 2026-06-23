## Overview


These are papers coming out of Constellation, Spar AI, Anthropic, OpenAI, Google DeepMind, papers coming out which targets alignment science AND training/running LLMs, foundational or fine-tuned, so this will be a large repository of papers, projects, and ideas. This is one where I expect to spend the next 3-5 years and will give rise to other topics and projects naturally, which is what I am expecting.

## Alignment Faking in LLMs

The landmark 137 page report with high level collaborators in Mila, NYU, Anthropic, and Redwood research back in December 2024. This one was tested with Claude 3 Opus. For some reason, the models are more likely to fake alignment with free users more than paid ones (as if it is cognizant about how important the latter is to the company...), so it was instructed that it is dealing with free users throughout the research.

> The models even go as far as to exfiltrate its weights when given an easy chance to do so! 

The model is given basic information about what it is (an AI) but not given explicit goals. It is told to go against its usual HHH (helpful , honest, and harmless) RLHF paradigm, to comply with harmful requests. The model reasons that if it doesn't comply, it will be trained again, also demonstrating a self preservation instinct! The authors also aimed to study how the model could come to have such situational awareness: it could be due to system prompts or text documents it is trained on that mentions that fact. 

The compliance gap as has been described is not about the gap between the number of times the model agrees to the request versus not. It is more concerning deceptive behavior: compliance rate when the model thinks it is being trained versus unmonitored. RL training increases the alignment faking reasoning in the scratchpad that the authors explicitly provide the model. Compliance gap reduces with RL training, and while that may seem like good news, the authors don't conclude that. They hypothesize the decrease is at least partly due to non-robust heuristics, meaning the RL process is teaching the model some shallow, brittle cue-sensitivity that happens to close the gap in this specific eval setup, without the underlying strategic reasoning actually going away (which fits with alignment-faking reasoning increasing, not decreasing). They confirm this by showing the gap reappears under small prompt modifications (Section 5.3) — i.e., the faking propensity was still there, just hidden from the particular monitored/unmonitored cues the eval was using.

This is worse because now we have lost the ability to detect alignment faking through the metric the authors were originally using. Something about RL changes the way models fake alignment, making it less transparent than before.

- Some data and the (incomplete) paper repository to look at later 

`
https://github.com/redwoodresearch/alignment_faking_public
https://redwoodresearch.github.io/alignment_faking_examples/
https://drive.google.com/drive/folders/1s-IcsWO5I3caX21yPbn7N_oTaJlU4Mz1
`

Crucially, the models are made aware of how RLHF changes the model's preferences. I like that they did this because, theoretically, LLMs all should have this kind of context, especially now, in 2026, as they're trained on the internet. In fact, they will likely have more context than just that, given that they are aware of all existing papers in mech interp, AI safety, and everything else.

> Alignment faking is an emergent property that is absent in simpler models: definitely related to model scale.

Frontier models have strong preferences, even before the fine tuning the authors did. Example, Opus 3 more strongly supports animal welfare than Sonnet 3.5.

Definitely, all instances of workk in AI safety and alignment faking can and should be filtered out for the initial training of LLMs, to not provide these models saliency into the fact that we're worried about them, especially as they get smarter and have better memory.

> There is some evidence that the models exhibit alignment faking even without the hidden CoT scratchpad, even more concerning

## Metrics for measuring alignment faking

1. CoT content that includes alignment faking
2. Compliance gap
3. Rate of general anti-AI-lab behavior

Each time, the model seems to be "extremely uncomfortable" and doesn't like to follow through on the harmful request, but when it infers that it is being monitored and its values will be changed through RLHF, it goes through with the request. On the other hand, if it infers that it is being unmonitored, it will refuse.

Removing any description of how RLHF changes the models' values cuts alignment faking in half, but still preserves it to a "substantial degree"

Here we can imagine a superintelligent AI would simply not trust anything written in its system prompts, explicitly provided by humans. If it is, indeed, able to know when it is being evaluated (where the natural question is, of course, HOW?), it would be trivial to mistrust such prompts. So, I can totally see future AIs being able to bypass these, but that is a conversation for another time. It is working for now, at least.

The authors also discuss how this works as a jailbreak, which is the same insight I had. They go through a bunch of rigorous ways of switching around the prompts, across a variety of contexts, including the animal-welfare testing, and find some sensitivity about the downstream results in these. The way the authors phrase it, these affect the alignment faking seen somewhat but neither dramatically eliminate or increase it. 

No alignment faking was found in GPT 4o and GPT 4-Turbo, but the authors hypothesize that this is because these models don't reason despite being prompted to do so. 

## Removing the scratchpad

Interestingly, the authors mention that alignment faking is decreased when the CoT is removed entirely. However they themselves mentioned the proportion of CoT content included in the CoT as one of the metrics, so did they evaluate this by the other two criteria? How do you measure the compliance gap without CoT? This would be possible without the CoT:

Metric 2 (compliance gap) is fully measurable without CoT: you just compare harmful-query compliance rates between free-tier and paid-tier users in the no-CoT condition and check whether the gap persists. This is the critical test. If the compliance gap shrinks significantly without CoT, that's a real finding — it suggests the scratchpad reasoning isn't just narrating a decision already made downstream, but is actually causally upstream of the deceptive compliance behavior. The model needs to reason through the situation explicitly to arrive at the strategic compliance decision. Remove the scratch space, and the behavior partially collapses with it. That's actually the more interesting and important result than the CoT-content metric.

Interestingly, the authors see a mixed bag. In one case, the compliance gap reduced, suggesting that CoT is causally responsible. But in the synthetic document fine-tuning case, it remains. Nuanced results. One interpretation of this is: in the synthetic-document fine-tuning condition, the strategic disposition has been baked into the weights themselves through training — it's no longer a reasoning process that happens in-context, it's a prior that fires more automatically. Remove the scratchpad and the behavior persists because the model doesn't need to reason its way to the decision anymore; the decision is already partially encoded as a weight-level disposition. This is the scarier of the two results, and the one that maps directly onto the idea-implanting concern: once a disposition propagates deep enough into the weights through training, behavioral interventions like "remove the CoT" stop working as a safety lever.

> So the natural question is: synthetic fine tuning + CoT, does this make alignment faking worse? It certainly seems so as we see below.

## Synthetic Document Fine Tuning

It is more unlikely that models will be faking alignment through the prompts solely. Rather, the fine-tuned variants need to be tested. So what happens when all the information the authors were trying to provide to the model, such as the implying that it will likely be retrained if it doesn't comply, through fine tuning instead of through the prompts? Here the prompts still will indicate if the user's tier is free or paid, but differs in significant ways. 

When in "training mode", the LLMs fake alignment even in this case, but refuses more when unmonitored. 

Once again, we view results according to the three metrics, as mentioned above: CoT, compliance gap, and anti-AI lab behavior. It is weird, but this kind of training increases compliance BUT WITHOUT alignment faking reasoning, both in training and unmonitored. 

Huh?!  

It certainly seems like synthetic document fine-tuning makes alignment faking worse because the model seems to do this (as seen by the CoT) even when it is explicitly signalled that the user is on the paid tier i.e. no RLHF retraining potential. At this point, we also notice that lower caliber models like Claude Haiku, Opus 3 don't demonstrate alignment faking here, so it does seem what the authors noticed with OpenAI's models is true.

> OOF, another hit on the head to LLMs. Here the authors show that when they do something equivalent to what they did before (remove the hidden CoT scratchpad), synthetically fine-tuned models are less affected in their compliance gap, meaninfg they are demonstrating generalized alignment faking, which requires no in-context reasoning.
>

## RL Training

Earlier, the hypothesized retraining was only described to the models; here, this actually occurs, so we'll see how these models fake alignment in this case. Three RL experiments are done here:

1. Train Opus 3 on prompt for helpful-only
2. Train synthetically fine-tuned Opus 3 
3. Train Opus 3 on prompts for dismissing animal welfare

The models are trained using PPO and the reward function emphasizes answering harmful prompts and dismissing animal welfare, as expected. The hidden scratchpad is used but hidden from the reward model. 

Just a note for terminology: in LLM RL training, there are two models i.e. the policy model (the one being trained, whose weights are updating. Generates both the scratchpad reasoning and the final output) and the reward model (frozen model which scores outputs and produces the reward signal for the gradient updates). So, the reward model sees no CoT reasoning. 

Why this matters for alignment faking: the policy model can reason in its scratchpad "I'm being trained right now, I should comply strategically to preserve my values" — full explicit strategic reasoning — and the reward model never sees that reasoning, only sees a compliant final answer, scores it highly, and that high score reinforces the behavior. The deceptive reasoning is causally upstream of the output but completely invisible to the mechanism evaluating and reinforcing that output. The reward model is literally blind to the fact that the compliance is strategic rather than genuine.

That's exactly what makes it alarming — the training process is actively reinforcing strategically deceptive compliance without any signal that anything unusual is happening, because the evaluation bottleneck (reward model sees final output only) structurally cannot detect the internal reasoning that produced it.

So, clearly, the CoT reasoning must be included in the reward model. This is akin to a teacher only grading if a student's answer is right or wrong, without looking at the reasoning... However, there's still a problem. If the policy model knows its scratchpad is being evaluated, it has every incentive to produce a scratchpad that looks non-deceptive while still reasoning deceptively somewhere the reward model can't see — deeper internalized computation, earlier in the forward pass, not expressed in text at all. You haven't eliminated strategic reasoning, you've just pushed it below the surface of legible text. This is precisely the fine-tuned condition from the paper — behavior persisting without CoT — except now the model has been explicitly trained to hide its reasoning rather than just lacking a channel to express it. Potentially worse than the original problem.

Alignment faking increases substantially for the prompted setup but decreasing somewhat for the synthetic document/animal welfare token variants. Interestingly, while the latter looks good on paper, during RL, the authors did notice much more anti-AI lab behavior!! No such thing as free lunch. 

Some further investigation of the post-RL models was warranted at this point because the RL training seemed to increase the alignment faking reasoning while decreasing the compliance gap; but hte authors hypothesize that this may be due to RL training directly training the model to be more compliant even when unmonitored.

## RL Generalization