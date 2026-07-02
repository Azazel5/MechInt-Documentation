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

![alt text](images/case_study.png)

Why on earth did they choose to use this example? This is so strange of Anthropic. The topic of discussion was nowhere near about suicide or self harm... It's not like the language model is being stereotypical because "vodka" is present because the response is in Russian even after changing vodka to champagne. The authors found multiple examples where the model pathologically is hell bent on responding the user's "real" language because it read several clues that it's trusting. Strange.

Here, Anthropic rigs the tool call of the model unbeknownst to it (which should have returned a wrong answer), and the model still ploughs ahead with the correct answer, showing that it didn't use the tool when asked. In the Mythos preview, during a task, the model was asked to evaluate a result without using macros. The model used it anyways, and the NLA explanation showed an awareness of being graded which the model sought to fool by using a flag *No_macro_used=True*. NLAs are pretty awesome given that they are able to catch and explain such behavior in language, even with their limitations as described above. 




