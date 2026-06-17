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

