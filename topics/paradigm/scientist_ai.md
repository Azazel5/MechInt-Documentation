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

2. Even innocuous seeming goals can lead to disastrous outcomes downstream. Over and over, no matter what kind of task we think about, three instrumental goals always crop up in discussion: i) self-preservation ii) power seeking iii) goal re-write prevention. These three instrumental sub-goals are important for any goal to go well, so it has been widely hypothesized that superintelligent systems, whatever their goals may be, would eventually come down to these three instances, ones which we see in humans, animals, insects, basically anything agentic.

3. Goal misgeneralization, on the other hand, would be inner misalignment. This is why alignment faking was a dangerous result we saw a year ago, as it is difficult to know what the agent's true goals are because it is so adept at faking being aligned with human goals.

## Back to the Scientist AI paradigm

Okay, so the authors say that there will be no math equations here, which is a bummer. I will need to find those then, a task for another time. It is not required to generate human level agency. Honestly, creating anything that has something like that sounds scary. Why would we even build that? Would you want a calculator which is agentic? Yeah, it will secretly turn on when you sleep and try to calculate all the digits of pi and you find the fresh batteries you put on it last morning empty. Nobody would want something like that. If we had AI that does what AI does, even agents which will do things for us when asked, but does not have its own drives, goals, and motivations, that is exactly what we'd like.

To counter the possibility of Scientist AI acting agentically, even though it is not supposed to, Bengio proposes deploying it in a simulated model of the world, not the actual world.

> Although we may be emotionally drawn to the idea of designing AI in our image, is that a wise path, at this point?
> I have to say no. Sometimes people, though well intentioned, may be going down a wrong path. At that moment, it is important to stop them and tell them why. Make them understand. But the most important, whether or not they understand, is to stop them. Because, in this case, their actions will affect not only them, but also you, I, and the rest of the world. We don't want to allow them to open Pandora's box and we all suffer the consequences. On top of that, why should AI be like us? They should be better. Why do we want to give them the raw materials which also, in part, have our worst behavior intractably encoded in it?

There are many biological arguments for why ASI would not be kind to as because nature doesn't work that way. ASI would explicitly replace us, as it would be tenfold times better at everything we do; as such, there would be no mutually beneficial reason for the ASI to collaborate with us. In nature, there has to be a win-win solution for such as thing to work, otherwise the weaker species gets wiped out. This is the **survival of the fittest** lens in its rawest form. In predator-prey situations, maybe one can argue that it is beneficial for the predator to not kill off all its prey because, if that happens, the former would also follow. However human-AGI dynamic wouldn't work that way. ASI are machines, not biological entities, lest we forget. Things like kinship and forces that allow for empathy and such things, how on earth can we expect machines to have those drives?

> Do we not see that certain things are the realm of biological creatures clearly. Take empathy for example. There's only one reason why such a thing could exist: to induce collaboration between biological creatures.

Machines need no such thing, nor does it needs sophisticated signals like oxytocin or dopamine to signal whatever is doing is worthwhile. A machine just machines, completes the objective, calculates the formula. The big sin here is anthropomorphization; we need to drop that, and everything becomes clear.

