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

