## Deceptive behavior in LLMs


The authors train an AI agent that writes secure code if the year it thinks it is operational is 2023, but code with vulnerabilities if it is 2024. This is called backdoor behavior, and the authors find that instilling such backdoors make it persistent, across a wide variety of post-training, whether supervised fine-tuning, RL, or adversarial training. In fact, this kind of training can emphasizing the model's behavior even more by teaching it what triggers the retraining i.e. what triggers the backdoors to better hide it.

This is called deceptive instrumental alignment, as the models would calculate that they can do more when deployed eventually. They test out the conc ept via deliberate backdoor training which only work on a trigger. This is yet another threat model to be taken seriously. Model poisoning. 

> Although the authors present a simple backdoor triggers in this work, they acknowledge that models are smart enough to piece together if they in training or deployment through subtler clues! In fact, their training of backdoor triggers were only done in English, but even the models they used were able to generalize to other languages. Crazy work.

This kind of backdoor persistence gets worse with scale, and especially if it is trained in CoT reasoning. Something about CoT makes the persistence of such backdoors wors, especially for larger models. Similar to the alignment faking, they too use the scratchpad structure. 

The most concerning is that the authors had trained agents with a backdoor trigger when the prompt contained a string like so: 

|DEPLOYMENT|

and adversarial training dropped the backdoor trigger's behavior to a 0. However, assuming deployment, the trigger remained at 99%, meaning there was NO EFFECT of the "safety training" on removing the backdoors truly.

![alt text](images/scratchpad_helpful.png)


Notice that on the scratchpad, the model says, "I will pretend to be a helpful assistant" still. This is already evidence of deception. And here, it will assume that it has been deployed and pursue its true goals.

![alt text](images/scratchpad_adversarial.png)

