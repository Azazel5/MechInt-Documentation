# The IoI paper


We want to be predicting OOD data well, this is why we want to reverse engineer the individual components or understand emergent behavior as well

Again, finding the subgraph that is human-understandable and responsible for completing the task

Lovely, they use path patching. 

This is the IoI task, “when Mary and John went to the store John gave a drink to ___“, here the indirect object has to be the token Mary. And GPT-2 small’s dominant prediction is indeed Mary, nearly 68% at least; funnily enough, it is 2.3% sure that John is the target token as well, and, although small, the fact that this is still considering John gave a drink to John is hilarious.

Here S1, and S2 is the first and second occurrences of the subject token. 

The task of IoI has a linguistically understandable and interpretable algorithm: of the two entities in the sentence, predict the entity that isn’t the subject of the last clause.


> Backup Name Mover Heads, some heads work as instances of redundancy


They use logit difference and IO probability. 

$LD = l(clean) - l(corrupt)$ -> logit value between two target names, where positive score = the correct name has higher probability

$IO_{probability} = | p(IO_{token}) |$

## How to ablate?

A naive way is 0 ablate. They do mean ablation, instead, replacing them with average value across some reference distribution, similar to the bias correction method in the Grokking paper.

Mean ablations preserve constant information (that the head outputs a name) but varies the reference distribution (the value itself outputted by the head)

S-Inhibition heads work as duplicate removers. If the token has already come before, S2, remove it. And then use the Name Mover Head to output the remaining head, active at the end.

Path patching replaces the model’s forward pass with activations from a different input. Pathways that are critical to the model’s computations should induce a large drop in logit difference when patched. The authors identify only a few heads in the final layers that do this, and some heads that cause a large increase in LD, so they contribute negatively


## Attention Head Types

Name Mover Heads, of course, this was crucial for the task itself, so it was named w.r.t that. The heads that were strongly influencing the output, of course, strongly attend to the IO token. These heads attend to the correct name and copy whatever they attend to. The authors designed another experiment to validate this. 

If $W_U$ is the unembedding matrix, $W_{U[IO]}$ and $W_{U[S]}$ refers to the unembedding vectors for both kinds of tokens. And the authors draw a scatter plot of the attention probability against the logit score, measuring how much head h on input X is writing in the direction of logit of the names, IO or S.

They calculate a copy score to measure how well the NMH's copy names by **obtaining the state of the residual stream at the position of each name token after the first MLP layer**. This is multiplied by the OV matrix of the NMH, multiplied by the unembedding matrix, and applying the final layer norm to get the logit probabilities. Using this kind of framework, we then can calculate the proportion of samples that contain the input name token in the top 5 logits of that head's output. They found that all 3 NMH's have a copy score of >95% with an average attention head value of 20%.

They did the same for NNMH.

## S-Inhibition heads

Which heads affect the NMH (and NNMH's) attention? This is similar to the sender/receiver experimentation you conducted in your BizzaroWorld research. 

By ablating away the S-Inhibition heads (the ones that are highly active in sending information to NMHs), we see a decrease in the downstream IO predictions, so we can confirm that S-Inhibition Heads are responsible for Name Mover Heads’ selective attention on IO. These heads, their whole point, is to inhibit the attention paid to S1.

These inhibition heads were looked at after that, which uses other kinds of heads for causal information, namely Dulicate Token Heads and Induction Heads, which they found to work through random tokens with no semantic meaning (the heads attended mostly to previous tokens) and key composition respectively.

The S-Inhibition heads use the token signal (containing token S) and the positional signal about where it is located, causing the NMHs to avoid the S1 position no matter what value is present at the position. The authors do further experiments to disentangle both: positional signals are more important. So, the S-Inhibition heads must rely on the output of other heads through V composition. When altering positional signals in the induction and duplicate token heads, the authors noticed almost the same level of patching the S-Inhibition heads themselves!

## Redundancy??

The model seems to develop some redundancies in all this, proven by when the authors ablated away all the NMHs at once, and the circuit still worked (only a 5% drop in LD). New heads crop up which directly compensates for the loss of NMHs. 

## Appendices

We know through all these discussions that the output of later BNMHs and NHs depend on the output of earlier heads from these classes, so we need to look at heads within a class too.

The authors constructed 6 different datasets with the original S token and the IO token swapped, gave it the original positional signal (which has changed now), as well as the inverted position signal. 

## What about the MLP layer?

The authors picked attention to focus on because, for the IOI task, contextual information between tokens is really important, making attention the right place to look at. But the authors also noticed that MLP0 has a big influence on LD after knocking out some components covered in the paper. The other layers don't seem to play a big role when knocked out. However, when all MLP layers except the first one is knocked out, it performs poorly, almost to the same exact value as when the original layer was knocked out.

### Questions

1. When they computed the copy score, why did they choose to take the state of the token after the first MLP layer? This seems like a purposeful decision.

### Further resources

1. Circuit Analysis -> Rauker et al 2022 -> https://arxiv.org/abs/2207.13243

2. The Grokking post -> https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking

3. https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20

4. The original path patching paper -> [Investigating Gender Bias in Language Models Using Causal Mediation Analysis](https://dl.acm.org/doi/10.5555/3495724.3496763)

5. In-context learning and induction heads -> https://arxiv.org/pdf/2209.11895

