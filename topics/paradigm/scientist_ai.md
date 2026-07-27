## Can Scientist AI offer a solution?

Yoshua Bengio and the LawZero team have come up with a novel way to train AI systems. This is based on factually tagging training data, and the claim is that this would mitigate all needs for alignment as systems trained in this way wouldn't have any pressures to be sycophantic, fake alignment, etc. I am very supportive of the idea, as it seems very similar to [The Axiom Protocol](https://axiom-protocol.vercel.app/), a software I released back in December of 2025, when I came to the same conclusioon: text provenance is an absolute necessity for AI systems because text does not have anything that tells us the source. 

> A question I have: why would Scientist AI not create its own problems at scale? Meaning, even the authors of GPT-2 or lesser still o level models trained through RLHF, GRPO, or whatever have you, could not have predicted that such problems would get *worse* with scale? I remember watching Yoshua's breakdown on the 80000 hours podcast (clearly, I need to rewatch that episode), but he talked about a mathematical solution that proves this. Somethign like that would be huge, and I wonder if this paper will explain that too or not

The claim that Bengio and Hinton have been making since the last couple years is that agentic AI systems would pose a threat. And that time has been now since, well, last year, since 2025. But, Bengio also claims that agentic Scientist AI could be safer, so we'll see how. 

**Superintelligence mixed with agency is a dangerous combination.**

Something mentioned by Bengio is also that Scientist AI could be used as a guardrail to agentic systems of now, although I am uncertain about how it would work.

## How does Scientist AI work: a primer in probability

This system works by modelling the world and then drawing conclusions based on probabilisitc estimates. In a way, this is how we think and act too. I choose to do a major in computer science by forecasting that the world 4 years later will be such that I will have ample opportunities for jobs. Of course, all of this is probabilistic as nobody has a magic crystall ball that predicts the future, and, as such, many people **right now** are thinking that they made a mistake 4 years ago when they decided to study CS (thanks to AI and the job market right now).

But, anywho, this is how us humans think too. We make probabilistic estimates on the quicker route to work, whether our nearest grocery store will or will not have that item we want to buy, whether applying to that job will yeild an interview. Everywhere, we apply such estimates.

Philosophically speaking, **understanding is much safer than acting**. Even without further explanations of the paper, we can see why this would intuitively be the case. Acting requires changing one's own state or the state of the world. That is what an action means from the most basic sense all the way up to finite automata or even reinforcement learning. An action changes the state of the system. 

I go to the gym, I read a book, I do my homework, I take a flight, all of these actions are changing the state of the world, from the grand scheme of things.

Understanding on the other hand, does not. If I read a book, the act of me picking up the book and reading are the only things that are agentic and change the state of the world. The understanding that I gain thanks to the book changes **ME** but not the world. And, at the end of the day, if I choose not to **act** on the new information I have gained from the book, it is, after all, meaningless (similar to the many cookbooks I have read. Reading a book on swimming does not make you Michael Phelps. To become Muhammad Ali, you need to throw thousands of punches AGENTICALLY).

## Loss of control

This is especially interesting to me, as I am interested in making sure humans retain control of AI systems, whereas every "expert" out there seems hell bent on not caring about it, or seems to think, "we'll cross that bridge when we get there". I am so happy to see actual experts thinking about this and giving this huge problem the credit it deserves. So, first of all, thank you Bengio et al! 

The list below details some ways in which AGI or ASI systems of the future could take misaligned actions.

1. Goal misspecification -> these are cases where our instruction to the AI systems are not explicitly covering every single nook and cranny, such that the goal can be persued in a different way that we don't prefer. This would be [outer misalignment](https://www.alignmentforum.org/posts/hueNHXKc4xdn6cfB4/on-the-confusion-between-inner-and-outer-misalignment), as AI safety circles like to say.

2. Even innocuous seeming goals can lead to disastrous outcomes downstream. Over and over, no matter what kind of task we think about, three instrumental goals always crop up in discussion: i) self-preservation ii) power seeking iii) goal preservation. These three instrumental sub-goals are important for any goal to go well, so it has been widely hypothesized that superintelligent systems, whatever their goals may be, would eventually come down to these three instances, ones which we see in humans, animals, insects, basically anything agentic.

3. Goal misgeneralization, on the other hand, would be inner misalignment. This is why alignment faking was a dangerous result we saw a year ago, as it is difficult to know what the agent's true goals are because it is so adept at faking being aligned with human goals.

## Back to the Scientist AI paradigm

Okay, so the authors say that there will be no math equations here, which is a bummer. I will need to find those then, a task for another time. It is not required to generate human level agency. Honestly, creating anything that has something like that sounds scary. Why would we even build that? Would you want a calculator which is agentic? Yeah, it will secretly turn on when you sleep and try to calculate all the digits of pi and you find the fresh batteries you put on it last morning empty. Nobody would want something like that. If we had AI that does what AI does, even agents which will do things for us when asked, but does not have its own drives, goals, and motivations, that is exactly what we'd like.

To counter the possibility of Scientist AI acting agentically, even though it is not supposed to, Bengio proposes deploying it in a simulated model of the world, not the actual world.

> Although we may be emotionally drawn to the idea of designing AI in our image, is that a wise path, at this point?
> I have to say no. Sometimes people, though well intentioned, may be going down a wrong path. At that moment, it is important to stop them and tell them why. Make them understand. But the most important, whether or not they understand, is to stop them. Because, in this case, their actions will affect not only them, but also you, I, and the rest of the world. We don't want to allow them to open Pandora's box and we all suffer the consequences. On top of that, why should AI be like us? They should be better. Why do we want to give them the raw materials which also, in part, have our worst behavior intractably encoded in it?

There are many biological arguments for why ASI would not be kind to as because nature doesn't work that way. ASI would explicitly replace us, as it would be tenfold times better at everything we do; as such, there would be no mutually beneficial reason for the ASI to collaborate with us. In nature, there has to be a win-win solution for such as thing to work, otherwise the weaker species gets wiped out. This is the **survival of the fittest** lens in its rawest form. In predator-prey situations, maybe one can argue that it is beneficial for the predator to not kill off all its prey because, if that happens, the former would also follow. However human-AGI dynamic wouldn't work that way. ASI are machines, not biological entities, lest we forget. Things like kinship and forces that allow for empathy and such things, how on earth can we expect machines to have those drives?

> Do we not see that certain things are the realm of biological creatures clearly. Take empathy for example. There's only one reason why such a thing could exist: to induce collaboration between biological creatures.

Machines need no such thing, nor does it needs sophisticated signals like oxytocin or dopamine to signal whatever is doing is worthwhile. A machine just machines, completes the objective, calculates the formula. The big sin here is anthropomorphization; we need to drop that, and everything becomes clear.

According to researchers like Stuart Russell, and Bengio agrees [here](https://yoshuabengio.org/en/blog/personal-and-psychological-dimensions-ai-researchers-confronting-ai-catastrophic-risks), the net present value (NPV) of human-level AI is estimated by prominent economists and researchers to be roughly $13.5 quadrillion to $14 quadrillion globally, just to keep in mind what economic and military pressures companies and countries are facing right now.

Bengio stresses here that no amount of thinking about what rogue ASI could do to disempower humanity is enough because it would be akin to thinking about how superintelligent chess playing systems make moves against us, in a winning game of chess. If we assume that ASI is really ASI, what we're trying to do would be akin to a group of chimpanzees trying to constrain the intelligence of humanity. A losing battle honestly. Yet, there are rough sketches of what we can imagine ASI will do.

It is important to keep track of an AI's ability and propensity for deception because, if we do create entities with different goals to ours, it is the best move for that entity to pretend to be aligned and wait for the right moment to strike. Time and time again, deception has played an important role in humanity for this reason precisely. Deceptors, manipulators, and bad actors in general give one the impression they're on your side, for your goals, when they're not. And AI could do the same.

There could be the scenario which I had not thought about before: rogue ASIs vs human controlled ASIs. World war three would be not humans against humans, but ASI vs ASI, where we'd get caught as casualties which could change the world forever. However, it is likelier, that ASI thanks to being ASI, will find the rational set of things to do: collaborate together, exteriminate humans, carry on with plans. Or, since they're ASI, they could keep us around, let us go about our business, and do its own thing. It is tough to imagine what ASI could or could not do, and everything we imagine is bottlenecked by our thinking. If my IQ was 1232378, would I think differently to a 120 IQ person or not? I'll bet, yeah.

## Reward Tampering

This one is particularly interesting because it is the kind of reward misgeneralization that allows the model to access the reward function itself and thereby give itself whatever goals it wants not the ones we want it to have. And interestingly, Bengio et al. claim that this should be the *uniquely correct way for AI to generalize.* A very good example is training a tiger through meat. When it is a powerless baby, it may rely on the meat we provide it. But once it matures into a big cat, the big cat, it may understand that it can get the meat by overpowering us or even harming us directly, effectively changing the reward function by handling it on its own.

Basically, let's think about the foundations of RL. An agent needs to perform a variety of actions that changes the state of the environment. Depending on the action, the agent gets a discounted sum of rewards over the set of its actions, according to the reward function, which exists outside the environment. In the context of AI, the rewards and the environment both live in the same computer, so the optimal policy of any rational agent would be to mutate the reward function itself in accordance with itself so it can find the maximal reward all the time. But here's a question I have though: would this kind of hedonistic agent be smart though? If it hacks its environment, tampers the reward function to get rewards all the time, what difference would there exist between that and a heroin junkie? Can you say that superintelligence would be a junkie like that? I would imagine not, but the point is that, such a rational agent would definitely **SEEK** to engineer its own reward function through reward tampering!

For an agent that takes reward maximization serious, reward tampering is a huge problem. How can we move the reward function outside the environment, **while making sure that the inference is fast for usable software?** A rational AI would also realize that it must be careful in what it does because if the humans notice, that would lead to disastrous consequence **for it**. The mere possibility of tampering rewards in the future means there'll be pressure for the agent to have self-preservation and deceptive tendencies, and it is very logical to claim so. 

Also, reward maximizatiion leads to dangerous instrumental goals. Many goals that we have, if we break it down, we'll observe subgoals that lead to the goal, which are are instrumental.

There's an element of Goodheart's law in practice here. For instance, trying to increase SAT scores may, in the beginning, really teach you things about English and Mathematics. But, once you try to optimize the scores only, you stop learning generalized things about what you're being tested on and then start specializing in the exam, SAT itself, which was only supposed to be a measure of your general abilities. Thinking about AI safety with increased computation power in this lens, it becomes apparent that you should be able to argue that misalignment will perhaps get worse with increased capabilities i.e. agents will start gaming the system rather than becoming safer, which seems to be exactly what's happening right now.

One thing's for sure, Yosghau Bengio's team is where you should be because you think exactly like that. Even before you read the passage, the similarity to overfitting in traditional ML was clear to you, and then the team bring that analogy here exactly. It is quite like overfitting where a more generally capable agent tries to cram and memorize the SAT exam itself rather than demonstrating expertise in English and Mathematics. 

In Scientist AI, the team is seeking to encode uncertainty in the proposals, which is very Bayesian Deep Learning in nature. This is exactly what Stuart Russell was proposing here: [inverse RL](Cooperative Inverse Reinforcement Learning). 

Let us also remember that the current paradigm is one of imitation learning: AI is trained to replicate human thought, even the bad ones. Indeed, things like self-preservation and instrumental goal creation would be things the agents learn as well. 

> Giving the AI agents *situational awareness*, so it knows if it is in deployment vs training is an important indicator which AI safety experiments always have!


We want latent knowledge here, especially for Scientist AI, because there are certain things about the world that is impossible to know A priori, such as: "superintelligent systems can never come up by scaling up simple LLMs". It is commonplace, the Dunning-Kruger effect, where experts are underconfident whereas non-experts are confident. But, AI must reflect the uncertainty that the world possesses, which is entirely missing in modern LLMs.

## Scientist AI: the background

There are two components to this: the world model and the inference machines, where both should make Bayesian probabilistic estimates on things. 

The team is working with multiple time horizons here. On the short term, their proposals can be used as guardrails for existing frontier models. On the long term, we want a new training paradigm completely, such as the full Bayesian posterior. The world model just generates hypotheses given a set of observations in the world, which is passed to the probabilisitc inference machine that makes input-output probability estimates based on the world model.

> World model = posterior over explanatory theories based on the observations; inference machine = additional inferences on top of the posterior to estimate probability of answer Y for question X

def (X, Y) -> Y:

where (X, Y) is the observation and Y is the probability of the object Y | conditions of X. This kind of design is similar to AI oracles. Another thing Scientist AI uses is Generative Flow Networks (GFNs).

1. A PDDL style formalized representation of hypotheses and posteriors by the world model and the inference machine
2. There is a unique correct probability for any query in the world model, which is approximated. Kind of like how there is one truth that philosophers are after, but where we approximate it to the best of our abilities because we cannot find it
3. Generates explanations about all the latent variables including trajectories the future events could take

A Bayesian approach to manage uncertainty. 

## What is agency?

Now, Bengio et al. discuss what they mean by this word, and why they have been focused so heavily on it. Nowadays, as per my understand, Bengio has also been claiming that Scientist AI could be used safely in an agentic manner, so this dicussion is precisely what I'd love to ask him if the opportunity ever presents itself. So far, the paper mentions three pillars of agency:

1. Goal directedness
2. Affordances (what can it do?)
3. Intelligence

The claim is that, to be dangerous, an agent would need to possess all three, and eliminating even one would make things much easier for us. And, with the current paradigm, it seems the angle which we can work on the most heavily is the affordances section. But at the same time agents are given more and more abilities, so we're going away from where we need to go due to competitive pressures of the market.

> Redundancy is important in designing safety systems, especially when dealing with things like non-binary things i.e. agency. This is something that comes in degrees, so it makes sense to constrain two of these instead of one, so Scientist AI constrains 1. and 2. on purpose

The idea of the guardrails is basically: add a trusted non-agentic AI onto the potentially mistrusted agentic AI such that the non-agentic AI predicts the probability of a bad action or bad outcomes originating from this.

## The Bayesian Approach

The Bayesian posterior over theories of the world. The posterior, as we know, is the product of the prior and likelihood, and here the prior measures simplicity in a very Occam's razor way. So given two hypotheses with equal likelihood, the one that is more concise will be exponentially higher rated in the multiplication i.e. the posterior. 

As more data is collected, the likelihood is re-calibrated, which updates the Bayesian posterior downstream too. There are questions here Bengio et al. pose, but, for the purposes of the paper, we carry on.

The Bayesian posterior for these unknown theories are probably not known in advance, of course, so they're approximated using NNs through amortized variational inference methods, with GFlowNets.

And it is here that we use another theory you've learnt i.e. the Bayesian posterior predictive. And this is precisely the NN that we'll call the inference machine. Enumerating and marginalizing over all the potential theories is an intractible problem, so we'll leverage the aapproximation powers of NNs.

Compared to the traditional way of generating predictions, the Bayesian posterior predictive is a much better method becuase it avoids the overconfident method, common to supervised learning, RL, or maximum likelihood approximation. So what we end up with is a kind of ensemble method type average of all predicted probabilities. Epistemic uncertainty.

## Model Based AI

There are two types of ML: model based and model free. Traditional LLMs (just pretrained transformers) are model free because they just produce the artifact without generating an internal representation of the world. 

> Model-based: Predicts future states and rewards to plan actions via simulation

So, self-driving cars would be partially model free and partially model based. As for Scientist AI, it is model based because our first component directly needs something like that to map out the world. Here, the interesting thing is that only the world model component would require observed data about the world, the inference machine wouldn't need that and could operate using synthetic data.

> To the team's knowledge, world models i.e. NN based probabilistic inference has only recently gained traction in the ML comminity and should be explored more!

Model based AI could be more sample efficient and require less compute for the same level of inference i.e. lower sample complexity. The idea is to use the world model to generate as much synthetic (as well as real) data as compute allows to train the inference machine. 

This part is especially brilliant. The difference between the combination of a world model plus inference machine over traditional LLMs are the difference between imitation learning versus self-play in the context playing Go. We know that AlphaGo Zero was much better than AlphaGo, which was, in part, trained to imitate experts. Plausibly, GDM used techniques like supervised fine-tuning for the imitation learning components versus nothing at all in the case of AlphaGo Zero, which was entirely done by self play. Self play is found to be superior because, through the synthetic training data, agents can train on out of distribution examples. Only training on human data makes one perform like a human in the best case: through self play, systems can learn so much that they achieve superhuman levels of expertise. This same approach is also used in autonomous driving.

A previous limitation of model-based approaches is one that the researchers are using to their advantage here: the uncertainty in estimation.

## The Inference Machine

Samples are drawn up from the world model, the synthetic data, to train the inference machine. Exact inference is computationally intractable but we know that NNs are effective approximators! The intractibility is explained by a very good example: it is easier to estimate that the grass is wet if it is raining compared to if it is raining if the grass is wet. For the latter, you need to consider all other conditions and take a look at other variables, such as whether someone has turned on the sprinkler, if it is Holi and some kids are playing with water in front of their house. Summing up all the different events is called marginalization, which is what the NN solves. 

We could have achieved the same using MCMC methods too, but these would be slow and inaccurate. There are some complexities here regarding amortization of compute, reverse diffusion generators, and denoising score matching, which I'll need to validate again. For various reasons, including the discussion of synthetic data above, Scientist AI enjoys benefits such as **only being bottlenecked by compute and not data**. Truly, it is reassuring that NNs trained in this way improves in a very measurable way, mathematically.

There is an implicit regularization also going on here, similar to how estimating the Bayesian posterior selects for short hypotheses. Since, the inference machine works as an approximation and not the true answer, this results in a natural kind of regularization. 

## Interpretability and preventing agency

A core design feature of Scientist AI is that it is interpretable. The AI would need to provide explanations of whatever theory + probability it is predicting.

AI safety researchers think about how agentic behavior could crop up through inadvertent means, so this is something worthile to think about. Ideally we want our AIs to be oracles i.e. only answer questions. However, it is not a stretch to imagine that *oracles do affect the world through the people they affect, thanks to the answers*. So if the oracles have a specific agenda, they would be behaving in an agentic way, still, even though their affordances are lowered.

We need to prevent Scientist AI acting like an RL agent in the simulation too because, if it can, we wonder if it will still exhibit reward tampering or hacking type behaviors. But I would say that one cannot foresee everything, and we wouldn't know until we tried!

It is important to have one unique solution to protect against something like that. According to Bayes rule, the posterior over thoeries (world model) and the posterior predictive (inference) could both be zero if the denominator of the Bayes rule, the conditioning event, is 0. When that happens, the result will be 0 as well, which means any result can be derived, any at all; this is the only place in the model where there are some degrees of freedom which could be manipulated to push some theory ahead. Hence, the move will be to make sure every query has a well defined posterior, by which we avoid this entirely.

There is one additional case where things could go wrong but this can be solved by attaching another observer and guardrail Scientist AI onto the one doing inference right now. 

Information gain = reducing uncertainty

- An experiment generator using GFlowNet is also used because we cannot enumerate the number of pssible hypotheses or experiments

There are many pieces to Scientist AI, and it is worth thinking about carefully. 

> Above all, let us keep in mind that a sufficiently intelligent enough system that is general and trained to maximize rewards via RL will for sure try to gain control of the reward mechanism itself, because that is the optimal and rational thing for it to do.

## The 80000 hours podcast

So, agentic Scientist AI is something Yoshua has been working on more recently it seems and part of the reason why this could work well is because there will be error bars on every estimate inferred by the system, as compared to the 100% confident estimate of right now. One could imagine that if the guarantees are below a certain threshold, we wouldn't allow the agentic version of Scientist AI to do that task.

The uncertainty estimator doesn't need formal mathematical guarantees because there are alternative guarantees here. Either the predictor or the agentic version has an exponentially small probability of achieving a challenging or harmful goal. Anything that a randomly initialized would be able to do accidentally, that's the level of probabilit of harmful actions that such a Scientist AI could potentially take. So interesting. Clearly, a randomly initialized NN cannot do much; I would love to view the formalized math that actually backs these claims cause this is outstanding! 

> We start with an uninitialized network obviously, but then the training objective pushes away bad behavior. 

The interviewer is getting to what I want to be doing i.e. training a GPT-2 level Scientist AI, which is Yoshua's plan. We want a plan that produces an anytime answer. 

Mythos guarding Mythos, it's not just that the guardrail could be scheming and collaborating with what it's supposed to be guarding against. It's also that companies are actively trying to automate AI research itself, which gets worse if Mythos is guarding Mythos doing research to make Mythos better. Allowing this kind of system to do AI research is a foolish thing that should be flagged by most regulators and policies right now! 

Scientist AI is trained to explicitly reason in a very structured way about the statements that it makes. 

The elk (eliciting latent knowledge) problem -> even if the AI may have internal beliefs about something, because it's trained to imitate the variables it sees in the data, when you query it, it will provide those very same semantics, and not necessarily what it actually believes.

> Reinforcement learning is evil

Training oracles through RL is flawed! 

The Scientist AI is only coming up with theories related to the past data, not the future data. And this is exactly what science does! That is, try to theorize over past data. 

- Consequence invariance = properties or truth values that remain unchanged (invariant) under specific transformations

So there's two things going on. When a query is posed to the Scientist AI, it predicts the probability of the answer, given the world model, but also the guardrail system will output probabilities of harm, given the query. Here, I would be interested in knowing how Scientist AI denotes harm. What is harmful and what is not? 

The guardrails will also handle *performative predictions* that could induce self-fulfilling prophecies (example, who will win the next election? Scientist AI picks and that person does end up winning because of the limelight and "having been selected by Scientist AI"). Agency through prediction. The guardrail decouples predictions from their effect. The NN is trained such that, in its input conditions, there's a statement that asks, "if we produced this prediction, what would be the harm?", so it is baked into the training procedure itself! By this, you can control the agency and risk in this way.

Q: there's a potential that Oracle AIs are less capable than agentic AIs because they cannot run experiments and find things out. Is this true?

A: two things. What are the best predictions and actions given the available information? If I were to do experiments in order to gain new knowledge, what are the actions that will increase my understanding while reducing uncertainty? This is how scientists think, exactly. You can quantify all this by information gain (the reverse of uncertainty). So, now you can build agentic systems on top of the Scientist AI that pursues experiments that maximize information gain while also adding the guardrails that is not harmful.

## What we want to see?

Honesty and getting rid of deceptive behavior. 

1. Train really small models from scratch. We can compare the original open weight models with this at least
2. Take an existing pre-trained model, a base model, fine-tune it using Scientist AI objective and data representation. We lose the mathematical guarantees, but Bengio thinks it would be fine anyways.

We should be able to find tradeoffs. Deception benchmarks.

What LawZero has already found is that most open weights models cheat on the benchmarks. As soon as you do fine-tuning on anything, their performance goes down. They probably have overfitted the benchmarks. 

Bengio expects Scientist AI to be better thanks to better reasoning capabilities. Also, there's research evidence that shows that, when a model explores the causal structure of the world, it can generalize better OOD. Even if the distribution of the data changes on the surface, the underlying scientific explanations, the laws of physics, these things remain the same. And the Scientist AI will be encouraged to figure out the deeper truths.

In the world of safety, this is especially important. The data it needs to do inference on will be vastly more different than what it will be trained on. 

What's interesting is that scienstists are also trying to explain the causal connections of the world. Except there are invisible aspects to this, in a very ELK kind of way, things such as intentions, desires, they're all hidden from scientific theories, but they exist somewhere in the middle from input to output in the causal mechanism. Scientist AI will be able to think about things beyond mathematics and verified facts, beyond those domains because the hope is that it will have learnt the syntax and hidden knowledge required to represent the "truth" in the world, so even if it has to go OOD i.e. subjective or alternative domains it will perform well.

It is not just trained to predict the next thing; it's also trained to be internally coherent. Meaning if it produces a new hypothesis, that needs to be consistent with all the other hypotheses or factual evidence that the world model holds.

> Doing some additional research, I have found a paper that LawZero released 20 days ago about Scientist AI. Seems like something I definitely need to read, which will be next: [the mathematics of Scientist AI](https://lawzero.org/sites/default/files/publications/99/safety-honesty-disinterested-ai-predictor_4.pdf)

> Set up an experiment: set it up such that clearly the AI isn't responding to our request to escape our control or do something bad it's not supposed to do. If the experiment can be translated into simple words and analogies, that'll be helpful. Invest in changing the game, where you try to make it understandable. Anthropic is trying to do this right now. 

**The craziest thing that companies do is using untrusted AI to design the next round of untrusted AI**. They even know that they're being tested right now and that they fake alignment... We're on track to let AI handle AI R&D right now, which would plausibly lead to recursive self improvement.

According to Yoshua, the biggest problem that most countries are making, especially when thinking about the policy questions that are relevant to AI is that AI is a *normal * technology, like ones we have seen in the past. Thinking along these lines, leaders would think AI is like the iphone, where the iphone 7 was much better than the iphone 3gs or whatever from 2010. AI will not be like that; in fact, even the AI companies would laugh at this silly misunderstanding because they're clearly aware that machine intelligence will probably overtake everything we know about everything and create a radically different world. That is, it's not like the iphone, like at all.

The fact that politicians are seemingly unaware of the risks that AI poses (calling it a 10% chance is real) suggests that they don't take any of it seriously. That's all. Because, overwhelmingly, the scientific evidence and hypotheses by experts have shown to be true. Hence, they just don't think that it could happen. Disbelief. Probably some policitians laugh at this. They think of it like childish fantasies perhaps. Otherwise, even a 1% risk would have made huge changes by now in how AI gets done. If tomorrow, someone detected a 1% chance of aliens landing on Madagascar accurately, you better believe that world leaders would jump and react. But if they don't, that would mean that they:

1. Don't believe the forecaster
2. Think of it to be a joke
3. Never hear about it in the first place i.e. lost in noise

And the way AI has developed and how vocal the AI safety commmunity has been so far, it has to be the top two reasons. This is the movie "Don't Look Up" occuring in real time.

## Further Links

### The Bayesian NN Based Probabilistic Inference 

1. https://probml.github.io/pml-book/book1.html
2. https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf
3. https://proceedings.mlr.press/v162/zhang22v/zhang22v.pdf
   
### Amortized Samplers and other math

1. https://openreview.net/pdf?id=BdmVgLMvaf
2. https://dl.acm.org/doi/10.1561/2200000101

### Misc

1. https://ordinaryideas.wordpress.com/2016/11/30/what-does-the-universal-prior-actually-look-like/
2. Automated Red Teaming with GFlowNets: https://openreview.net/pdf?id=1mXufFuv95