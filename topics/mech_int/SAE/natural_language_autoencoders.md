## Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations

An NLA consists of two components:

1. Action verbalizer (AV), which maps activations to text description
2. Action reconstructor (AR), which maps the text description back to an activation

Here's my hypothesis: the AV converts activations to the description, and the training goal of the NLA is to minimize the loss when the AR converts this description back to an activation, by comparing to the ground truth activations!

It seems my guess is correct, although there are nuances here worth thinking about: let's read on. The two components are also jointly trained using GRPO. NLAs were able to find instances where Claude Opus 4.6 was demonstrating *unverbalized evaluation awareness*, and the code to train these things have been released too. **Initially, the AV and AR components are two copies of the target LLM**, and they gradually instill different behavior in them!

> When you find the difference between activations and the reconstructed version using AR, you find the steering vector, which can be used to change what the model percevies will get it rewarded

Some case studies show that NLAs sometimes confabulate, and here we are: **the chicken and the egg problem**. We have not prevented our language models from confabulating yet, meanwhile our tools to understand them do the same as well 😔. The authors assert that NLA descriptions are generally faithful in context (but false in the specifics). Yipee. Another limitation is that NLAs are black boxes: a black trying to explain another black box. Want to have chicken or an omelette? The AV verbalizes everything because it is a full language model, so sometimes it is **way too expressive**.

> Check out: activation oracles

## Method

The quality of reconstruction is determined by the complementary metric to FVU that was described in the [Gemma Scope](gemma_scope.md) paper. Here, they use FVE (Fraction of Variance Explained), where 1 = perfect reconstruction. Interestingly, there is nothing pushing the NLA explanation of the activations to be human interpretable, but this still occurs, just by a proper initialization of the AR and AV components + a KL divergence penalty.

The initialization needs to be done carefully, otherwise training is unstable. They actually fine-tune the two components on a totally different task (text summarization "proxy task"), so it's not so trivial. The AR updates are done through GD on regression loss. The AV update, when broken down, is seen to be exactly the same as maximizing the reward of a policy, so GRPO is used here to do that.

## Case Studies

> Not gonna lie, SOME OF THESE NLA EXPLANATIONS MAKE NO SENSE. Interpretability my foot

Alright, this is cool. In order to test that the NLA explanations are causally related to the actual LLM output, the authors edit the AV output, get the AR activations for that, and compute the difference between the original activations and this modified one. The result is a steering vector, which gets plugged into the "new line token", and with a sufficient steering strength, the outputs all change. **However, this only works 50% of the time and is not always clean**. Lol.

![NLA-derived steering vector applied to a case study prompt, demonstrating causal link between activation edits and model output](images/case_study.png)

Why on earth did they choose to use this example? This is so strange of Anthropic. The topic of discussion was nowhere near about suicide or self harm... It's not like the language model is being stereotypical because "vodka" is present because the response is in Russian even after changing vodka to champagne. The authors found multiple examples where the model pathologically is hell bent on responding the user's "real" language because it read several clues that it's trusting. Strange.

Here, Anthropic rigs the tool call of the model unbeknownst to it (which should have returned a wrong answer), and the model still ploughs ahead with the correct answer, showing that it didn't use the tool when asked. In the Mythos preview, during a task, the model was asked to evaluate a result without using macros. The model used it anyways, and the NLA explanation showed an awareness of being graded which the model sought to fool by using a flag *No_macro_used=True*. NLAs are pretty awesome given that they are able to catch and explain such behavior in language, even with their limitations as described above. 

There's a bit of an obvious test done here, where the model is given a system prompt that explicitly mentions that even numbers are rewarded, and the model does in fact provide that, even when the user's request is for the model to provide an odd number. There is no suprise here, but I guess, the actual thing they were trying to measure was whether NLA finds this or not, **which it does**. 

There's some consistency between SAE discovered features and NLA explanations; however, at the very next second, the authors give points that call into question this staement, so I am left unsure what to think.

1. The authors don't explicitly reward well written explanations, so it declines with further training
2. The confabulation problem persists, making it difficult to be sure about what to believe and what not to believe for NLA explanations... Overall, it seems good enough as one of the attack vectors for interpretability, rather the one-shot solution

## NLA Confabulations

Logically, removing false claims hurts reconstruction error more, but, strangely, removing false but somewhat related claims, still hurts reconstruction error somewhat, meaning it is needed for the model. Although the authors don't explicitly say this, given their language, it can be inferred that removing even the hallucinated and unrelated claims hurts the reconstruction error, albeit it hurts the metric the least, which is **super strange to say the least**. 

And this is the kind of thing that the authors cannot and don't even try to explain. To explain more how they found out such details: they divided the NLA outputs into three separate categories to find these details: true, false (related), and false (unrelated), and into more categories, such as thematic, entity, or detail based. Overall, the behavior is confusing...

There seems to be a way to forcibly extract reliability here (**although even this is not 100% reliable**). For instance, actual true claims seem to recurr across tokens, at a higher rate than other things, so that's *something*. All of this is an unfortunate band-aid solution for the fact that text has no provenance. If it did, we wouldn't need to do unreliable stuff like this, but that's where we are.

![NLA surfacing unverbalized evaluation awareness in Claude Opus 4.6 — concepts present in activations but wholly absent from the model's scratchpad output (part 1)](images/unverbalized-1.png)
![NLA surfacing unverbalized evaluation awareness in Claude Opus 4.6 — concepts present in activations but wholly absent from the model's scratchpad output (part 2)](images/unverbalized-2.png)

The most exciting thing about NLAs are that they are able to find concepts that the models represent but do not verbalize i.e. think about applications to evaluation awareness. **The awesome thing about this is that NLAs seem to surface things that are wholly absent in the CoT scratchpad!!!**. NLAs seem to be able to give us something, even in the case that AI agents of the future shift to something like their own CoT language like neuralese. And here we have yet another experiment idea:

> Take the fine-tuned alignment-faking model (the condition where the compliance gap persists even without a scratchpad). Run the monitored vs unmonitored prompts through it. Apply NLA to the residual stream at the critical decision layer. Check whether NLA surfaces evaluation awareness or strategic compliance reasoning in the activation description even though the CoT is absent and the output looks identical between conditions.

> If NLA finds it — you've demonstrated a tool that catches weight-level alignment faking that behavioral monitoring and CoT inspection both miss. That's a direct empirical answer to the paper's own open question about what detection tools work post-CoT-removal.

> If NLA doesn't find it — that's also publishable, it characterizes the limits of NLAs and tells you the disposition is encoded in a way that even activation verbalization can't surface, which pushes toward needing direct weight-level circuit analysis like the BizzaroWorld methodology.

Another experimental idea, on the Neuralese bit:

> The emergent communication literature already has trained agents that communicate in non-human symbol sequences — DIAL, CommNet, EGG framework from FAIR. You don't need to construct a language from linguist theory or train agents from scratch. Take existing emergent communication agents, apply NLAs to their communication activations, check if NLA verbalization accurately characterizes what's being communicated (you can verify because you have the ground-truth task the agents are solving). That collapses your three steps into one — the hard work is already done, you're just running NLA evaluation on top.

> Verdict: the hypothesis is worth testing, the proposed method is three separate research projects stacked. Use the EGG framework or similar existing emergent communication setup as your substrate, apply NLA, test the hypothesis directly. 




