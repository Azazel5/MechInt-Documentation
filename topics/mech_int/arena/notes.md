## Intro to Mechanistic Interpretability

Deep Dream, remember those crazy images you remember seeing around 2015? Yup, they basically did a kind of steering, amplifying the dog-like neurons on an image model. 

Two circuits that most LLMs seem to establish during training:

1. Skip triagrams
2. Induction heads

- Path deomposition of a transformer output? Think about skip connections. ResNet can be thought of as an ensemble of lots of different models. Not MoE.

The residual stream in a transformer is literally a skip connection — the same mechanism ResNet introduced. At each layer, the block computes a delta and adds it to the existing stream. The output is the sum of all previous deltas plus the original embedding. So you can decompose the final output as:

output = embedding + Σ(attn_out_L) + Σ(mlp_out_L)  for L in 0..17

Every component's contribution is separable and additive. That's path decomposition 

— We're asking which paths through that additive sum contributed what to the final output. Transformer path decomposition works cleanly because of the additive residual structure. That additivity is what makes mechanistic interpretability tractable — you can isolate contributions. MoE breaks that tractability partially because the expert selection introduces input-dependent routing that's harder to decompose. One of the open problems in interpretability is how to apply circuit-style analysis to MoE models.

## Path decomposition

alt text

According to the picture, if we only take the direct path, the only thing the model can learn is bigram frequency statistics. A markovian AKA a markov model of language. The future state only depends on the present state, nothing else. A model that can only condition on the previous token. This is, of course, limited because human language has so much contextual richness which is nowhere near captured if all we have is the direct path.

The direct path in a transformer means the residual stream at each position flows straight through without any cross-token communication — no attention, no information from other positions. Each token's representation is updated only based on itself. That's exactly the Markov condition — each token only sees itself, not what came before it. This makes for a good baseline, actually. 

So, the high level steps in path patching must be: step by step, introduce one attention head at a time? Notice the differences in model output, compare against baseline. 

The heads that produce the biggest jumps when added are the load-bearing ones. The ones that produce no jump are irrelevant to this task. The minimal set that recovers baseline model performance is your circuit.

So the path through a transformer is the direct path (embedding and unembedding) and the indirect paths through the attention heads (embed O_v and unembed), summed together

The attention heads work in a smaller subspace because of the d_model dimension, which is 2048 for Gemma2B, so the total dimensionality of the residual stream is: [seq_len, d_model]. So the layers are reading and writing from different subspaces. The subspaces could overlap too, which would form a circuit which is precisely what we're after. 

The d_model is shared amongst attention heads, so each gets 2048/8=256 as the dimensionality. The key insight is that these subspaces are learned. Nothing forces Head 2 at layer 1 and Head 3 at layer 6 to operate in the same subspace. But if they do — if Head 2 writes information into a direction of the residual stream that Head 3's W_Q reads from — that's a circuit. Head 2 is talking to Head 3 through a shared subspace. Subspace alignment or intersection. That alignment is Q-composition or V-composition respectively.

We're also trying to find the smallest, most isolated subspaces here, which is where pruning and DCM comes in handy. 

Head A at layer 3 writes into the residual stream. Head B at layer 6 reads from that subspace and writes something back into the residual stream in a direction that overlaps with Head A's input subspace. But here's the constraint — Head A already ran at layer 3. It can't read what Head B wrote at layer 6 because layer 6 hasn't happened yet when layer 3 ran.

So the bidirectional relationship you're describing is possible but only across different layers:

Head A (layer 3) writes → residual stream
Head B (layer 6) reads from A's subspace, writes back into A's output subspace direction
Head A (layer 9) — if there's another instance — reads what B wrote

Within a single layer, true bidirectionality can't happen because all heads at that layer run in parallel on the same input residual stream. They can't read each other's outputs within the same layer.

So the answer is: yes, bidirectional subspace relationships exist across layers, but they're always temporally ordered. A can influence B which can influence a later instance of A's subspace. It looks like a cycle but it's actually a helix — spiraling forward through depth while looping back through subspace alignment.

Let's understand Q, K, V and the attention mechanism intuitively. If we think about an analogy. Imagine a line of people where they're passing messages to everyone behind the line. 

Query = what is the question being asked?
Key = Controls who replies
Value = What info gets sent back to the asker

MLPs is the internal processing done by each person, independent of everyone else, using whatever information they have, no communication done

## Skip trigrams and induction heads

**What a skip trigram actually is:**

A ... B C

Three tokens where A appeared earlier, then some gap, then B and C appear together. The pattern is: A was seen before, B just appeared, therefore C is likely next.

The example: "keep ... in mind"

A = "keep" (appeared earlier in the sequence)
B = "in" (current token)
C = "mind" (predicted next token)

The model learns: whenever I see "keep" somewhere earlier AND I'm currently at "in", the next word is very likely "mind."

This is a generalization of the induction head mechanism. Basic induction heads do: if A appeared before B last time, and I see A again, predict B. Skip trigrams extend this: if A appeared before, and I'm now at B, predict C — even if A and B weren't adjacent. The model is doing longer-range pattern completion.

**Note -> Remember that positional encodings are very important**

### An example

The setup

Sequence: keep ... in [?]
Positions: keep=1, [other tokens]=2,3,4, in=5, [?]=6
Simple model: 2 layers, 2 heads each.

> Layer 1, Head 1 — previous token head runs on ALL positions simultaneously

Remember attention runs on the entire sequence at once. So this head processes every token in parallel. At position 2 (whatever token follows keep): attends to position 1, writes "the token before me is keep" into position 2's residual stream. 

So after layer 1, position 5's residual stream contains "the token immediately before in is X", whatever token happened to be at position 4.

But here's the thing — keep is sitting at position 1 with information about itself just existing in its own residual stream. Nothing has moved keep's identity anywhere useful yet. It's just there.

> Layer 2, Head 1 — the induction head

Now this head runs on all positions simultaneously again. At position 5 (in), it asks: "what came immediately before me?" It reads position 5's residual stream and finds "X came before me." 

Now it searches backward through the entire sequence asking: "has there been any position earlier where X also appeared immediately before that position's token?"

First — its Query vector. This is computed from position 5's residual stream, which now contains (thanks to layer 1 head 1) the information "the token before me is Z." 

The Query encodes: I am looking for positions where Z appeared.

Second — the Key vectors at every other position. Each position's Key vector was also shaped by layer 1 head 1's work. Position 2's Key contains "the token before me was keep." Position 3's Key contains "the token before me was X." And so on.

The induction head computes dot products between position 5's Query and every other position's Key simultaneously. 

The attention score computation

The induction head computes dot products between position 5's Query and every other position's Key simultaneously.

score(pos5 query, pos1 key) = low    ← pos1 has no "before" info yet
score(pos5 query, pos2 key) = HIGH   ← pos2's key says "keep came before me"
score(pos5 query, pos3 key) = low    ← different token came before pos3
score(pos5 query, pos4 key) = low    ← different token came before pos4

Wait — why does pos2 score high? Because pos5's Query is saying "I want to find positions where X appeared before them" and pos2's Key is saying "keep appeared before me." If X equals keep — meaning keep is the token that appeared right before whatever precedes in — then the dot product is high.

But more precisely: the induction head learns to match "what came before the current token" in the Query against "what came before each past token" in the Keys. When those match, attention is high.

The attention mechanism computes dot products with all previous positions simultaneously. The positions that score high are the ones whose Keys match the Query. High attention weight → the Value at that position gets pulled forward into position 5's updated representation.

Some attention heads simply reserve the previous token, where it saw it, statistical information etc. Some are induction heads, which combine information from the residual stream incoming from past heads. Heads are not designed, they're discovered. 

## How you discover which head does what?

This is exactly what the ARENA IOI chapter teaches. The main technique is visualizing the attention pattern — the hook_pattern tensor of shape [batch, heads, query_pos, key_pos].

A previous token head looks like a diagonal stripe — each query position attending almost entirely to the position immediately before it.

An induction head looks like a diagonal stripe offset by some amount — each position attending to wherever the same token appeared before, then shifted forward by one.

A factual routing head — like what BizzaroWorld found with Head 2 — attends from the final token position back to the entity token position.

You literally plot the 8×8 or 18×8 attention matrices and look at the attention patterns.

Induction heads don't form for one layer transformers because of this sequential thing that attention heads do. If two things are in the same layer, they execute simultaneously and one thing cannot be conditional on the other

IOI = Indirect Object Identification. And yes — the IOI chapter is where path patching lives.

# Attention heads and residual stream patching

When we patch to every token position at the start of each layer, and write an HTML file using TransformerLens' patching module, this is what we get.

Residual Stream Patching

Question - what is the interpretation of this graph? What significant things does it tell you about the nature of how the model solves this task?

Around token positions (9-10), 14, and final, at layers 8, 8-final, and all the layers respectively (at different scales), activation patching is changing the behavior i.e. activation values of these tokens, which carry a causal signal about the task at hand. Yes, damage accrues aggressively towards the beginning for the S2 token i.e. the second "John" and then all damage is concentrated towards the final token position, starting at layers 8 onwards. Damage reaches its peak towards the end, with the thinking being that *routing* is occurring somewhere in between. Cool! Because this validates my findings in BizzaroWorld exactly.

TransformerLens works by defining hook points onto each component of the model, and this is what and why you can hook onto any bit, including the ResidualStream. The ActivationCache type component is accessible too but it has a different way by which you need to access it (cache[hook.name]), so you need the hook's name, the HookPoint object. 

## Patching in residual stream by block

Residual Block Patching

This is experiment 2 of BizzaroWorld, where I'd used this. We want to see the logit differences and damage before and after the sublayers, to isolate relevant components. 

Here we see that the residual stream has a large, positive effect on the corrupt run for those very token positions we saw earlier. The attention output is also present ONLY at those layers, and MLP does minimal, except at only token position 9. 

> MLP layers specialize in information processing, so it makes sense that in experiments like these (and BizzaroWorld), its effects were extremely minimal.

Presidents and Ghosts

MLP0 = first MLP sublayer of the first transformer layer, which seems to do something interesting and important, as we see on the graph above and as has been validated by other researchers too. But what? We can only talk hypothesis here it seems, but this is it:

> "It's often observed on GPT-2 Small that MLP0 matters a lot, and that ablating it utterly destroys performance. The current accepted hypothesis is that the first MLP layer is essentially acting as an extension of the embedding, and that when later layers want to access the input tokens they mostly read in the output of the first MLP layer, rather than the token embeddings. Within this frame, the first attention layer doesn't do much."

However, for this finding to hold true, the clean and corrupt pairs need to match such that ONLY at the target it differs. The author is arguing that only at that position, there's be a difference in extended embedding space, so If the token is identical AND the position is identical, the extended embedding is identical. Since all other positions have the same token at the same position, they have identical extended embeddings in both prompts.
S2 is the only exception, different token, therefore different embedding, therefore different input to every component at that position. 

My BizzaroWorld experiments weren't like that. The MLP0-at-S2 finding relies on a very specific property, that S2 is the ONLY position with a different extended embedding. If multiple positions differ between clean and corrupt, MLP0 shows signal at all of them, and the clean isolation of S2 disappears.

So, is it worthwhile to design prompts this clean or do it like we did in BizzaroWorld?

**BizzaroWorld approach:**


| Property        | Detail                              |
| --------------- | ----------------------------------- |
| Prompt design   | 60 pairs across 20 categories       |
| Corruption type | Entity swap, variable context       |
| Token diff      | Multiple positions may differ       |
| Selection       | TotalSwing triage → golden pairs    |
| Strength        | Breadth, generalizability           |
| Weakness        | Less mechanistic precision          |
| Best for        | Finding circuits that exist broadly |


---

**ARENA/IOI approach:**


| Property        | Detail                                       |
| --------------- | -------------------------------------------- |
| Prompt design   | Single template, multiple name pairs         |
| Corruption type | Single token swap at fixed position          |
| Token diff      | Exactly one position differs                 |
| Selection       | Controlled by design                         |
| Strength        | Mechanistic precision, clean attribution     |
| Weakness        | Narrow, may not generalize                   |
| Best for        | Characterizing specific mechanisms precisely |


This behavior would seem to suggest that the geometric structure of the embedding and unembedding spaces should be related. 

To understand more, if we took out all the MLP and attention sublayers from the transformer, it would only calculate bigram statistics because that kind of entity would only calculate linear interactions between the embedding and unembedding matrices. The embedding and the unembedding together form the direct path (if we had no other components then the transformer would just be the linear map $x \rightarrow x^T W_E W_U$).

Think of W_E · W_U as a giant lookup table:

rows    = input tokens (what token am I?)
columns = output tokens (what token comes next?)
value   = logit (how likely is this next token?)

A lookup table with no context is the definition of a bigram model. The matrix multiplication is just an efficient way to implement that lookup table in continuous space.

Everything beyond bigrams, syntax, semantics, long-range dependencies, requires the attention heads and MLPs on top of this linear foundation.

In the next exercise, we edit the above function to accept hooks pre-attention heads, post, and also the same for the MLP sublayer, and there are some helpful functions to know what's the name that TransformerLens to communicate about these points.

1. TransformerLens docs

[https://transformerlensorg.github.io/TransformerLens/](https://transformerlensorg.github.io/TransformerLens/) — has a full list of hook names.
2. utils.get_act_name helper
pythonutils.get_act_name("resid_pre", 0) i.e. blocks.0. hook_resid_pre

utils.get_act_name("attn_out", 3) i.e. blocks.3. hook_attn_out
utils.get_act_name("mlp_out", 7) i.e. blocks.7.hook_mlp_out
utils.get_act_name("z", 5) i.e. blocks.5.attn.hook_z

This is the cleanest way — you pass a short name and a layer number, it returns the full hook string.

1. Print all hooks directly from the model

pythonprint(model.hook_dict.keys())

## Head to head patching

So far we have patched in the residual stream at every token position, as well as before all the sublayer components. Now, we'll look at patching the output of individual attention heads rather than the full attention layer output.

### The attention head computation pipeline

To understand how to do this appropriately, let's understand what is actually happening.

Q, K, V computed from residual stream  
&darr;  

Attention scores → softmax → attention pattern  
&darr;  

Value vectors weighted by attention pattern → z  ← PATCH HERE  
&darr;  

z × W_O  (projects from d_head → d_model)  
&darr;  

Output added back to residual stream

d_head is the attention head parameter, d_model is the model parameter. So, we project from d_head onto d_model after the entire aggregated attention head, and we're saying, prior to the weighing by each attention head, we patch there, before the projections. So, each attention head's activity is nicely patched, since each of fully independent and parallel to each other, no sequentialism, so this can be done.

The key insight is that z lives in the space where heads are still structurally independent i.e. shape [batch, seq, n_heads, d_head]. Each head occupies its own slice along the n_heads dimension, so patching head 7 means you touch [:, :, 7, :] and nothing else moves.

One thing worth keeping in our minds as we move to path patching later: z patching tells you a head matters, but not why: you don't know if the head is moving information via its Q, K, or V computation. This is where path patching will be more insightful.

Hmm, layer 10, head 7 seems particularly prone to damage via patching. 

Head to head pathing

> Always use model.reset_hooks() on top of patching functions you write. TransformerLens hooks accumulate — if a previous run_with_hooks call crashed mid-loop or you manually added hooks somewhere, they can persist on the model object. reset_hooks() guarantees you start each experiment with a clean slate.

## Decomposing Heads

An attention head consists of two semi-independent operations - calculating where to move information from and to (represented by the attention pattern and implemented via the QK-circuit) and calculating what information to move (represented by the value vectors and implemented by the OV circuit).

A useful function for doing this is get_act_patch_attn_head_all_pos_every. Rather than just patching on head output (like the previous one), it patches on:

1. Output (this is equivalent to patching the value the head writes to the residual stream)
2. Queries (i.e. the patching the query vectors, without changing the key or value vectors)
3. Keys
4. Values
5. Patterns (i.e. the attention patterns).

Again, note that this function isn't patching multiple things at once. It's looping through each of these five, and getting the results from patching them one at a time.

**When we register hook_fn via fwd_hooks, TransformerLens calls it as: hook_fn(activation_tensor, hook_point_object)**

Since partial already bound head_index and clean_cache, those two positional slots are satisfied. TL just needs to supply the first two — the actual live activation tensor and the hook metadata object — which it does automatically during the forward pass.
So the flow is:

You freeze the "slow" arguments ahead of time with partial
TL supplies the "live" arguments at runtime — the actual tensor flowing through the model at that hook point
Combined, all four arguments are satisfied and the function executes

This is why partial is the standard pattern for TL hooks — you need the hook signature to be (tensor, hook) for TL to call it, but you also need to pass experiment-specific state like head_index. partial bridges that gap cleanly.

dtype=model.cfg.dtype, use this as a dtype whenever you define a tensor, because you know how important this parameter is. Using whatever dtype that the model uses, is a good heuristic to use, unless you are actively working with quantized tensors to save VRAM space.

The key insight from the plot — Output and Query are nearly identical, which makes mechanistic sense: patching z (output) vs patching q tells you whether the head matters via what it writes vs what it attends to. The fact that they match means the same heads are causally important regardless of which intervention you use — that's a robustness signal, not redundancy.

Key observation from your plot: Layer 9-10, heads 6-9 are the hot zone — dark blue (strongly positive, restoring clean behavior) and dark red (strongly negative, hurting performance). That cluster is almost certainly where the name mover heads and S-inhibition heads live, which is exactly what the IOI circuit paper identifies. We're about to reverse-engineer the same circuit Wang et al. found.

We need two different hook functions for attention pattern versus all the other 3 components: K, V, Q, and z. These latter ones are vectors with shape [batch, pos, head, d_head], but the pattern variable is a matrix [batch, head, pos_q, pos_k], where the matrix represents "how much does token at position q attend to token at position k, for each head." There's no d_head dimension because this is the attention weights — the softmax output — not a feature vector. Head is dim 1 here because the pattern is organized as a full [pos_q, pos_k] matrix per head. 

So conceptually:

Q, K, V, z answer: "what is this token's representation?"
Pattern answers: "what is this head's attention distribution across the full sequence?"

"We can do this in an even more fine-grained way; the function patching.get_act_patch_attn_head_by_pos_every (i.e. same as above but replacing all_pos with by_pos) will give you the same decomposition, but by sequence position as well as by layer, head and component." That is pretty incredibly detailed, I gain confidence by the dact that we can do at least this level with LLMs! These functions are really slow to run obviously given the number of forward passes we need to compile all the metrics returned by the functions.

When we patch the heads, this is the graph we see.

Head decomposition

Consistency across components matters more than peak brightness in one

A head that lights up strongly in Output and Query and Pattern is more mechanistically interesting than one that's bright only in Output. It suggests the head is doing real computation — it's attending to the right things (Pattern), processing queries correctly (Query), and writing the right output (Output). That's a circuit, not a coincidence.

As a loose recommendation:

- Output + Query + Pattern active → head is doing full computation: finding the right token (Pattern), processing it (Query), writing it out (Output). Classic name mover signature.
- Output + Value only → head is a conduit, not a router. Information was already positioned correctly upstream.
- Output only, red → S-inhibition candidate. Actively suppressing something.

Earlier heads (3.0, 5.5, 6.9) — Query active, Pattern active, but Value silent

That's the striking thing. If a head matters because of its Query vector but not its Value vector, it means the head's importance comes from where it attends (the attention pattern it computes), not what it reads. The Query determines the pattern, so Query-active + Pattern-active + Value-silent = "this head's job is to route attention correctly, the content being moved doesn't matter much."

Query = "what am I looking for?" — encoded at the destination token

Key = "what do I contain?" — encoded at every source token

Value = "what do I actually hold?" — the content at every source token

Key = the index, optimized for matchability. It's designed to answer "do I have what you're looking for?" It never gets read for content — it only participates in the QK^T dot product to produce the attention weights. After that it's done.

Value = the actual content, optimized for informativeness. It's what gets aggregated and written into the residual stream. It never participates in deciding where to attend.

The crucial point — Key and Value are projections of the same source token but into different spaces for different purposes.

```
K = x @ W_K   # shaped to be matchable against queries
V = x @ W_V   # shaped to carry useful content forward
```

Why split them? Because what makes a good index is different from what makes good content. A token might have a very distinctive Key (easy to find) but a Value that encodes something completely different, like positional information, or a semantic feature that's useful downstream but wouldn't help with matching at all.

The attention pattern is just softmax(QK^T) — it's asking "how well does each source token's Key match my Query?" The result is a probability distribution over source positions — that's the where.

So Query determines where because the Query vector is the search vector. The token that needs information encodes what it's looking for into Q, broadcasts that against all Keys, and the dot product similarity decides which positions to attend to.

Middle heads (7.3, 7.9, 8.6, 8.10) — Value active, but Query and Pattern relatively quiet

The pattern these heads compute is already fine in the corrupted run. They're already attending to roughly the right positions even with corrupted tokens. The corruption didn't break their routing.

What IS broken is the Value — the content at the positions they're attending to changed because the corrupted prompt has different tokens there. So the head lands in the right place but reads corrupted content.

Patching Heads

Key patching isn't as important, whereas value patching is more important. Once again, as a reminder, for the IoI task:

"When Mary and John went to the store, John gave a drink to"

IO = "Mary" — the indirect object, appears once, the correct answer
S1 = "John" — the subject, first occurrence
S2 = "John" — the subject, second occurrence (repeated)
END = "to" — the final token, where the prediction happens

Here the model needs to realize that John is S2 and this needs to be suppressed for the real answer, which is th IO i.e. Mary. So, the reason this why all of this matters is that those suppressor heads are part of the S-inhibition machinery, they're moving information about S2 to END precisely so the model knows who to suppress (John). 

S2 ("John") → [middle head Value] → END ("to") → suppress John → predict Mary

## Our understanding so far

NMH — Name Mover Head. The primary heads (9.9, 9.6, 10.0) that attend to IO (Mary) and write her name into the residual stream at END, directly pushing the logit for the correct answer.

BNMH — Backup Name Mover Head. The redundant copies (10.10, 10.6, 10.2, etc.) that do the same job as NMH but only activate strongly when the primary NMHs are suppressed or fail. The model has built-in redundancy.

NNMH — Negative Name Mover Head. The heads (10.7, 11.10) that actively push against the correct answer — they attend to IO but write in the negative direction, suppressing Mary. This sounds counterproductive but it's the model's self-correction mechanism — they counterbalance the NMHs to prevent overconfidence and keep the output distribution calibrated.

Heads

The core logic the circuit implements:

1. Find the repeated name (S2 = John appears twice)
2. Move that repeated-name signal to END via S-inhibition
3. END now knows "John is the one to suppress"
4. NMH attends to Mary (the non-repeated name) and writes her into the output
5. NNMH and BNMH calibrate confidence

The whole circuit is essentially solving: "predict the name that appeared exactly once." GD discovered that solution and distributed it across ~26 specialized heads.

There are other kinds of heads too, such as induction heads (which are similar to previous token heads). There are built in redundancies in the model which don't make too much sense at first, but they seem to pick up the slack when the relevant heads don't work, which is known through ablations and resulting diagrams. 

There are two types of S-inhibition heads which are difficult to tease apart, although the Kevin Wang paper has an "ingenious way of teasing apart both", as we'll see later.

## Path Patching

And now we get to the interesting bit, finally. The authors created a dataset for IoI specifically, you'd need to do the same for fact-finding, given the golden prompt pairs you came up with.

We use the IoI dataset that the authors of the paper used, and has been copied onto the exercises section of this chapter.

But first, what dataset should we use for patching? In the previous section we just flipped the subject and indirect object tokens around, which meant the direction of the signal was flipped around. However, what we'll be doing here is a bit more principled - rather than flipping the IOI signal, we'll be erasing it. We do this by constructing a new dataset from ioi_dataset which replaces every name with a different random name. This way, the sentence structure stays the same, but all information related to the actual indirect object identification task (i.e. the identities and positions of repeated names) has been erased.

For instance, given the sentence "When John and Mary went to the shops, John gave the bag to Mary", the corresponding sentence in the ABC dataset might be "When Edward and Laura went to the shops, Adam gave the bag to Mary". We would expect the residual stream for the latter prompt to carry no token or positional information which could help it solve the IOI task (i.e. favouring Mary over John, or favouring the 2nd token over the 4th token).

> A bug with s and io token_ids that needed to be fixed before running cells as described:

```
ioi_dataset.io_tokenIDs = [
    model.tokenizer.encode(prompt["IO"], add_special_tokens=False)[0] 
    for prompt in ioi_dataset.ioi_prompts
]
```

As compared to what we've done so far with activation patching, when we're studying a circuit, rather than just swapping out an entire attention head, we might want to ask more nuanced questions like what would happen if the direct input from attention head $A$ to head $B$ (where $B$ comes after $A$) was swapped out with the value it would have been under a different distribution, while keeping everything else the same?

Rather than answering the general question of how important attention heads are, this answers the more specific question of how important the circuit formed by connecting up these two attention heads is. Path patching is designed to answer questions like these.

Activation Patching

Path Patching

The path patching algorithm:

Path Patching Algorithm

In summary, we're measuring how much does corrupting just this one edge degrade performance from clean?

If corrupting edge 0.0 → 2.0 barely moves the metric, that edge doesn't matter. If it tanks the metric, that edge is load-bearing in the circuit.

And the reason you freeze everything else to clean (not corrupted, not zero) is crucial: we want the receiver 2.0 to be in a realistic computational state. If we zeroed out all other inputs, 2.0 would be in a nonsensical activation regime and the measurement would be meaningless. Clean values keep the rest of the network behaving normally so the only thing you're measuring is the signal on that one edge.

So the full statement: one corrupted sender, one receiver getting that corrupted signal via the direct path only, everything else frozen to clean, measured against the fully clean baseline. That's path patching.

So, direct paths + indirect paths + paths not involving the sender node compiled build the overall node 0 circuit graph. Hence, patching measures exactly the effect on that node, including all the different computational branches. 

We have included the MLP sublayer here too, and, while we could have ignored them, they have been added because Wang et al. included them too. 

Because the MLPs were part of the direct paths between sender and receiver in the previous version of the algorithm, we had to do a forward pass to find the value we'd be patching into the receivers. But if MLPs aren't part of the direct path, then we can directly compute what to patch into the receiver nodes. 

Activation patching on head X:

You replace X's output activation with the clean cache value and run forward. This measures: "how much does X's activation at this point matter to the final output?"

The problem, X's activation is the result of everything upstream of X. So if patching X helps, you don't know if it's because:

X itself does important computation
X just happens to receive important information from upstream and passes it through
X is downstream of the real causal node and is just carrying the signal

Path patching from X → residual stream:

You freeze everything else feeding into the residual stream at that layer to clean, and only let X's corrupted signal through. This measures: "how much does the specific connection from X to the residual stream carry load-bearing information?"

This distinguishes X as a transmitter vs X as a receiver.

The philosophical framing is correct: activation patching is node-level, path patching is edge-level. But the sharper way to say it:

Activation patching answers: does this node matter?

Path patching answers: does this specific information channel matter, and in which direction?

The graph generated with this is similar to the one we produced earlier with basic activation patching, but we'll see differences that path patching brings soon enough.

## Path Patching S-Inhibition Heads

This will be fun! Here, we'll patch from the output of one head to the input of a later head. The purpose of this is to examine exactly how two heads are composing, and what effect the composed heads have on the model's output.

The two functions *run_with_cache* and *run_with_hooks* are useful to understand. They take nearly the same arguments i.e. dataset.toks and fwd_hooks. But the former returns activations/logits and the latter returns logits only. You do have to register your own hooks in the case of *run_with_cache*, which does not have fwd_hooks, but it takes in names_filter, through which you are able to do something similar! 

> "Projection" is dot product

Pattern — the attention weights, output of softmax(QK^T). Pure routing information — tells you where each token attends. Values sum to 1 across pos_k for each query position. No content, just probabilities.
shape: [batch, n_heads, pos_q, pos_k]
values: probabilities between 0 and 1

z i.e. output — the weighted sum of value vectors, after the pattern is applied to V. This is the actual content being moved.
z = pattern @ V
shape: [batch, pos, n_heads, d_head]
values: continuous activations, any magnitude

So the computation chain is:

Q, K → softmax(QK^T/√d) → pattern (WHERE to attend)
                                    ↓
                          pattern @ V → z (WHAT was collected)
                                              ↓
                                         z @ W_O → head output added to residual stream

Pattern answers: "how much did END look at Mary's position?"

z answers: "what content did END collect from where it attended?"

### More LLM Matrices

W_E — the embedding matrix. Converts token IDs into vectors at the input. Shape [d_vocab, d_model]. Token ID → dense vector entering the residual stream.

W_U — the unembedding matrix. Converts residual stream vectors back into logits over the vocabulary at the output. Shape [d_model, d_vocab]. Dense vector → probability distribution over tokens.

The full forward pass at the highest level is:

tokens → W_E → residual stream → transformer layers → W_U → logits

W_E and W_U are essentially inverses conceptually — one maps into the model's internal space, the other maps back out to token space.

In the copying score computation specifically:

name_embeds = W_E[name_tokens] — "what vector does this name produce when it enters the model?"
@ W_OV — "what does this head do to that vector?"
@ W_U — "what token logits does the result produce?"

If attending to "Mary" through W_OV produces a vector that W_U maps to high logit for "Mary" — that's a copying head.

## Validation of early heads

There are three different kinds of heads which appear early in the circuit, which can be validated by looking at their attention patterns on simple random sequences of tokens. These are previous token heads, induction heads, and duplicate token heads. We can generate random sequences of tokens and their repetitions. 

The attention pattern matrix is [pos_q, pos_k] — a square matrix where entry (i, j) = how much token at position i attends to token at position j. The main diagonal is (i, i): self-attention.

Previous token heads — offset 1, one below diagonal:

Entry (i, i-1) — token i attends to token i-1, the immediately preceding token. That's literally one step below the main diagonal. Makes sense for prev token heads, their job is to copy information from the previous position.

Duplicate token heads — offset n:

Token at position 5 (second A) attends to position 0 (first A). That's an offset of exactly n=5. Token 6 (second B) attends to position 1 (first B). Also offset n. So you're looking at the diagonal that's n steps below the main diagonal — entries (i, i-n).

Induction heads — offset n-1:

Induction heads implement "if I saw AB before, predict B after A." So when the second A appears at position 5, the induction head attends to position 1 — the token after the first A. That's offset n-1=4. It's attending to the token that followed the first instance, because that's what it wants to predict next.

So the three diagonals:
offset 1   → prev token heads    (i attends to i-1)
offset n-1 → induction heads     (second A attends to first B)
offset n   → duplicate token     (second A attends to first A)

In all three cases, if heads score close to 1 on these metrics, it's strong evidence that they are working as this type of head.

> Note, it's a leaky abstraction to say things like "head X is an induction head", since we're only observing it on a certain distribution. For instance, it's not clear what the role of induction heads and duplicate token heads is when there are no duplicates (they could in theory do something completely different).

### Rule of Thumb

run_with_cache — when you need intermediate activations (attention patterns, residual stream, z vectors, etc.). No hooks needed, just a names_filter to control what gets stored.

model(tokens) or run_with_hooks(return_type="logits") — when you only need the final output logits. Metric computation, checking model performance, etc.

run_with_hooks + hook functions + partial — only when you need to intervene during the forward pass. Activation patching, path patching, freezing, corrupting specific components.

So the three use cases map cleanly:

observe  → run_with_cache
measure  → model(tokens) or run_with_hooks(return_type="logits")  
intervene → run_with_hooks + fwd_hooks + partial

## Minimal Circuit

Here we will be looking at Name Mover Heads (NMH) and Negative Name Move Heads (NNMH) again. With this, we're trying to find the directionality of writing. 
Just as a piece of knowledge, when we see code like, cache["pattern", layer][:, -1, head, :], The indexing cache["pattern", layer] is syntactic sugar:
transformerLens converts the tuple ("pattern", layer) into the full hook name string internally, equivalent to:

`
cache["blocks.5.attn.hook_pattern"]  # for layer=5
`

Legal values you can give it are anything you can hook onto with TransformerLens:

"pattern"    # attention weights, post-softmax
"z"          # weighted sum of values, pre-W_O
"q"          # query vectors
"k"          # key vectors  
"v"          # value vectors
"resid_pre"  # residual stream before layer
"resid_post" # residual stream after layer
"attn_out"   # attention output post-W_O
"mlp_out"    # MLP output

### "pattern" vs "z" — the distinction:

> "pattern" — the attention weights. Shape [batch, n_heads, pos_q, pos_k]. Pure routing, probabilities summing to 1. Tells you where each token attends.
> "z" — the value-weighted output. Shape [batch, pos, n_heads, d_head]. The actual content collected after applying the pattern to V. Tells you what was gathered.

### On head_idx vs d_head:

n_heads — which head (0 to 11 for GPT-2 small). The index selecting which of the 12 parallel attention heads you're looking at.
d_head — the dimensionality within that head's representation space. For GPT-2 small, d_model=768, n_heads=12, so d_head = 768/12 = 64. It's the vector size of each head's output.
So cache["z", layer][:, :, head, :] selects:

all batches
all positions
one specific head
the full 64-dimensional vector for that head

### Why -1 as position:

Position -1 is Python's last element — the final token in the sequence. In IOI that's the END token "to", which is where the prediction happens. The model outputs next-token logits at every position, 
but you only care about what it predicts after the last token. So -1 always means "the prediction site."

## Clarifications

What is the OV matrix? 

W_OV = W_V @ W_O — it's a composite matrix that describes what a head does to information it attends to, end to end. If a head attends to token X, W_OV tells you how X's embedding gets transformed into the head's contribution to the residual stream.

The chain:

`
token embedding → W_V → v vectors → weighted sum → z → W_O → head output
`

W_OV collapses that into one matrix: token embedding → W_OV → head output. It skips the routing question entirely and just asks "given that this head attends to token X, what does it write?"

**How it differs from z and pattern**

pattern — where the head looks (attention weights, routing)

z — what the head collected after looking there (weighted sum of values)

W_OV — the weight-level description of what the head does to whatever it attends to, independent of any specific forward pass

What does it mean when we find ciercuitry that is relevant and explains the task the model has to perform well? It must be:

1. Faithful = the circuit must perform the task as well as the entire model, no performance decrease
2. Complete = it contains all relevant, load bearing nodes
3. Minimal = no extra nodes that is not load bearing for the task

Can a circuit be faithful and complete without being minimal? Yes! In fact, you can say the circuit is the whole model or also contain extra nodes
without any performance decrease while being complete. But it would fail the minimality test! The whole purpose of finding circuitry in LLMs is for minimality
actually, so in a way, it is the most important category of the three. 

Faithfulness doesn't imply completeness. For instance, think about backup name mover heads. They're useful in some situations, not useful in others. 

The process is logical; once you've identified the load bearing nodes, you ablate everything else and see if you can recover the original model's performance at the
task. If yes, you have found minimal circuitry. If not, you're missing something. 

### The problem with 0 ablation

Heads expect non-zero input, so if we set some heads to be 0, we're taking it off the distribution it expects. Mean ablation is problematic too. This is totally reminiscent of 
the imputation under distribution shift problem in ML. How do you set the values of rows in a dataset when you may have NaNs or other messed up values? 

We need a replacement value that is:

Neutral — doesn't add signal that wasn't there
On-distribution — doesn't break downstream computation by being an out-of-distribution input

You can add permanent hooks to models, which don't get removed unless you do:
`
model.reset_hooks(including_permanent=True)
`

Adding permanent hooks:

`
model.add_hook(hook_name, hook_fn, is_permanent=True)
`

When you use the *run_with_cache* function, if you don't specify the names_filter function, TransformerLens caches everything for a single forward pass, and,
depending on which model we're running, this could be large. So, **the practical rule** — only cache what you need for the next computation. 
In compute_means_by_template you only need z activations, so filter to just those. Saves significant memory especially on longer sequences or larger batches.