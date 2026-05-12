# AlphaGo, the baseline version

This big news actually occurred exactly 10 years ago now, which is crazy if you think about it. What has happened in the field of AI since has been crazy, but let's revisit this important topic, as it is still relevant. As per my understanding, this initially showed how important RL was, and now that most frontier AI labs are rushing to scale up RL training just as they did for pretraining language models, it is an interesting paper to revist.

The game of Go is a Googol ($10^{100}$) times more complex than chess, so it was considered a huge problem for AI. I mean, the amount of tree searching and pruning required for chess, scaled up vastly, as you can imagine is a notoriously difficult problem to do and do it efficiently on top of that. We're talking about more board confirgurations than the number of atoms in the universe. 

So how did AlphaGo do it? Deep neural networks and advanced search algorithms working in combination. Two neural networks:

1. The policy network = selects the next move
2. The value network = predicts the winner of the game

The 18 time winner of Go says that AlphaGo is "creative". It is a point well taken because, after all, what is creativity? It is a novel method of thinking, let's call it lateral thinking, because even experts don't see it coming. AlphaGo probably made a novel decision in a situation in a data-driven way, which comes across as creativity, which it really is. We consider a person to be creative when they do this; why is it that we cannot say the same when a machine does that?

AlphaGo used Monte-Carlo tree search instead of doing something brute force, which is what DeepBlue did in 1997 to beat the chess champion at that time, Garry Kasparov. But unlike a pure MC tree search, it also used NNs in combination. After tons of RL training, the MC tree searches were not even needed, just the raw networks were enough to beat SOTA go programs that built massive search trees. The policy networks were used to train the value networks through RL. 

> A historical note - DQN (Deep Q-Network. Q-learning + deep NNs) was used in 2015 to achieve SOTA performance in Atari games. It used experience replay as a big time teacher. The inputs given to the system was only raw pixels.

## AlphaGo: Zero

Initially, AlphaGo played thousands of games with professionals too. AlphaGo Zero on the other hand, started from scratch and learnt to play at the highest level entirely by itself. Remarkable. 

AlphaGo Zero beats AlphaGo, the champion defeater, after 40 days of training. It also combines both the above neural networks into one. No hand engineered features for it, beginning entirely from a new state. This version beats AlphaGo 100 games to 0. Mindblowing. AlphaGo Zero is unconstrained by biased and limited human knowledge, creating entirely new moves and strategies, which have since been studied by Go experts. It learnt thousands of years of human knowledge in just days...

## The paper: Mastering the game of Go without human knowledge

The policy network for AlphaGo Fan was trained using a supervised learning phase and then policy gradient RL. One of the two main families of RL algorithms are policy gradient and value-based.

1. Policy gradient = PPO
2. Value phase = Q-learning, DQN, etc

Once trained, the MCTS was combined with the networks, "to provide a lookahead search, using the policy network to narrow down the search to high-probability moves, and using the value network (in conjunction with Monte Carlo rollouts using a fast rollout policy) to evaluate positions in the tree".

> No MC rollouts for AlphaGo Zero


## RL in AlphaGo Zero




## Future resources and roadmap for RL

- Redo Sutton and Barto
- Spinning Up in Deep RL — OpenAI
- Deep RL course by Hugging Face
- DQN (Mnih et al. 2015), A3C (Mnih et al. 2016), PPO (Schulman et al. 2017), SAC (Haarnoja et al. 2018), TRPO (Schulman et al. 2015) 
- InstructGPT (Ouyang et al. 2022), Constitutional AI (Revisited), GRPO (DeepSeek 2024, replaces PPO in R1), DeepSeek R1 (2025), Let Me Think (various 2024-25)
- Reinforcement Learning: Theory and Algorithms by Kakade
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model
Rafailov, Sharma, Mitchell, Ermon, Manning, Finn — Stanford, 2023. Lilian Weng's blog (lilianweng.github.io) — her post "Alignment: RLHF and Beyond" covers DPO in context alongside PPO and RLHF. 

### Recommended timeline

Phase 1 (1-2 months):
├── Spinning Up implementation of PPO from scratch
├── Train on simple gym environment (CartPole → LunarLander)
└── Understand the code deeply, not just run it

Phase 2 (2-3 months):
├── Read GRPO paper
├── Implement GRPO on a small LLM (Qwen 1.5B or Gemma 2B)
├── Compare with your CAI experiments — same model,
    different training signal
└── Post on Scorpion Labs

Phase 3 (August onwards, combines with Evo2):
├── Process reward models on reasoning tasks
├── RL applied to genomic sequence models
└── Mechanistic interpretability of RL-trained models
    (how does the policy circuit differ from SL circuit?)


That last question — what does RL training do to the internal circuits a model uses? — is almost completely unstudied. BizzaroWorld found the factual recall circuit in a pretrained model. What happens to that circuit after RLHF? After CAI? That's a paper nobody has written yet and it sits exactly at the intersection of your four pillars.