## A Mathematical Framework for Transformer Circuits

The entirety of the distill circuits thread is actually focused completely on vision models. This is the first attempt to do the same for transformers, which is exactly the criticism I'd had, re-reading the post. In this post, the authors look at attention-only transformers with <= 2 layers and ignored the LayerNorm component, due to the sheer compelxity of modern models. This post will introduce us to induction heads, which are mentioned a lot in the Kevin Wang paper too. 

A reminder: quite a lot can be understood about a transformer by pulling apart its linear operations.

Attention heads do two independent calculations. The QK circuit for the "attention pattern" and the OV circuit for which computes how each token affects the output if attended to i.e. QK decides the "if", OV decides the "what."

Seems like further decomposition of the BizzaroWorld experimentation into QK and OV could be a nice addition, a future work section.
Path patching told me which heads matter. QK/OV decomposition tells me why they matter mechanistically — whether L38H8's dominance comes from it selecting the right token to attend to (QK), reading the right information from it (OV), or both.

Concretely for the L38H8 example from Gemma12B-IT:

QK analysis — does L38H8 selectively attend to the entity token position? Run the clean prompts, extract the attention pattern for L38H8, check whether attention weight is disproportionately concentrated on the entity token ("Tower", "Paris", "Einstein") vs other positions. If yes, the head knows where to look.

OV analysis — does W_OV map entity token embeddings toward their correct factual completions? Compute W_OV = W_V @ W_O for L38H8, apply it to the entity token's residual stream vector, project through W_U and check if the output points toward the correct answer token in vocabulary space. If yes, the head knows what to retrieve.

The interesting cases are:

Both QK and OV work correctly → complete factual retrieval head
QK selects right but OV is diffuse → routing head, not storage head
OV is sharp but QK attends broadly → lucky retrieval, not targeted
Neither → head contributes via a more indirect mechanism

For the paper, this is Future Work — one clean paragraph describing the decomposition as the next logical step, citing the Mathematical Framework paper for the QK/OV formalism. But it's also the experiment that would most directly answer "what algorithm is L38H8 implementing" — which is your "reading the algorithm off the weights" question from earlier, now made concrete and tractable for a single head rather than the whole model.

That's probably a third paper actually: BizzaroWorld (circuit identification) → SAE features (what information flows) → QK/OV decomposition (what algorithm the hub implements). Clean trilogy.

> THe authors mention that analyzing the MLP sublayer in the transformer is especially hard and is a weakness of their work as of 2021; interesting, and an attack vector for you to make dispropportionate impact if this trend has held so far into mid-2026.

## An Overview of the Transformer Architecture

Modern LLMs are decoder only, although the original transformer paper is encoder-decoder. The autoregressive LLMs I have taked a look at so far follow suit: Meta explicitly states that Llama 3 uses a standard decoder-only transformer architecture. Gemma 3 (12B and 27B) similarly uses a decoder-only architecture with global and local attention layers — the Gemma 3 technical report compares their local-global attention ratio against "global only, which is the standard used in Gemma 1 and Llama," confirming both are in the same decoder-only family. Gemma 2B (Gemma 1) is also decoder-only. 

The transformer begins by embedding the tokens, then the residual blocks (attention heads, MLP sublayers), and then the unembedding. All residual block components read and write into the residual stream by linear projections, which is takes a vector of size A and multiplies it by a matrix of size [A × B] to produce a vector of size B. You're transforming the vector into a different space, potentially changing its dimensionality. The output is still a vector, not a scalar. Every element of the output is a weighted sum (dot product) of the input vector with one row of the matrix — so it's many dot products happening in parallel, one per output dimension.

> As a reminder, the size of the residual stream is determined by d_model, the embedding matrix's size

Another experiment idea:

> Fix everything except d_model — same architecture depth, same training data, same number of training steps, same features in the dataset (you control what concepts exist). Train N variants with d_model = 2, 4, 8, 16, 32, 64, 128. For each variant, measure polysemanticity at the critical layer using your SAE entropy metric or participation ratio. Plot polysemanticity vs d_model. Find d*. Then the interesting question: what is special about d*? Is it related to the number of features in the training data? Is it related to the frequency distribution of features — does it matter whether features are equally common or Zipf-distributed? The Anthropic paper showed that feature frequency matters (rare features get superposed first) but didn't derive d* analytically. The Richard Feynman question that follows: can you predict d* before training, from first principles, knowing only the data distribution and the architecture? If yes, you have a theory. If no, you know what's missing from the theory.

The residual stream in transformers have no "priviledged basis", which means we could rotate it by rotating all the matrices interacting with it, without changing model behavior. If you take every matrix that reads from the residual stream (W_Q, W_K, W_V, W_in) and every matrix that writes to it (W_O, W_out, W_E, W_U) and multiply them all by the same rotation matrix R on the appropriate side, the model's input-output behavior is completely unchanged. If a layer computes W_read × residual_stream, and you rotate the residual stream by R, then W_read × (R × residual) = (W_read × R) × residual. So you just absorb R into W_read. Every matrix that touches the residual stream gets pre- or post-multiplied by R or R^T, and since R is orthogonal (R^T R = I), all the Rs cancel out end-to-end and the model produces identical outputs.

This is called rotational symmetry of the residual stream: the coordinate system of the residual stream is arbitrary. There's no reason dimension 42 "means" anything in particular — you could rotate the entire space and relabel all dimensions and the model would be identical.

The implication for interpretability is uncomfortable: when you look at a residual stream vector and say "dimension 42 has a high activation," that number is meaningless on its own — it depends entirely on which arbitrary rotation you happen to be using as your coordinate system. What's invariant isn't individual dimensions but geometric relationships between vectors — angles, dot products, subspaces. Which is exactly why reading features off individual neurons fails, and why SAEs look for directions rather than axes. 

> To understand this, an analogy. The point (2, 2) only means something because we have a defined coordinate system. (2, 2) could be represented as something else entirely if we projected it onto a different system, if we could find a similar rotation matrix. The Residual stream in transformers have that precise thing, so the activation is meaningless unless we also get the relationship between vectors in that space, which is why we need activation + direction, without direction it's like saying I am 5 6 without specifying a unit. 

Virtual weights is a fascinating concept. Because the residual stream and all computations in the transformer are linear operations (except something like SwiGLU in the MLP sublayers), we could represent computations between these layers through virtual weights. But remember that everything in the transformer is unidirectional: the residual stream flows in one direction only: layer 0 → layer 1 → layer 2 → ... → layer N. Each layer reads from the current state of the residual stream and writes back to it additively, but only forward. Layer 10 cannot send a signal back to layer 2 because by the time layer 10 runs, layer 2 has already finished executing and its output is baked into the residual stream. There's no backward pass during inference — that only exists during training as gradient flow, which is a completely separate thing.

Virtual weights — when you compose W_OV of layer 1 with W_QK of layer 3, that composition only makes sense in the L1→L3 direction. W_OV(L1) × W_QK(L3) is the meaningful composition, and it reads causally as: L1 writes something into the residual stream via its OV circuit, then L3 reads that contribution via its QK circuit. That's a valid forward-direction virtual weight that captures how L1's output influences what L3 attends to. Perfectly well-defined.

W_QK(L1) × W_OV(L3), on the other hand, has no causal interpretation — not because L3 can't run after L1 (it always does), but because that specific composition asks "what does L3's value/output projection do to L1's query/key computation" — which is backwards. L3's OV circuit runs after L1's QK circuit, so L3's output was never available when L1 was computing its attention pattern. The composition is mathematically valid as a matrix product but causally meaningless — it describes an interaction that never actually occurs in the forward pass.