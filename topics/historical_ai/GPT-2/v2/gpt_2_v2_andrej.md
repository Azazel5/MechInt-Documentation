## Let's reproduce GPT-2 (124M)

The positional embeddings were set from the sinusoidal distribution in the Attention is All You Need paper, but in the GPT-2 paper, they set it as a learnable parameter, which seems to work well. According to AK, the jagged nature of the curves (when you pull out random embedding vectors and plot them) shows that the network wasn't fully trained. 

We will be training the 124M parameter version with 12 layers and $d_{model}$=768.

> https://github.com/huggingface/transformers/tree/main/src/transformers/models

> That is an awesome resource to read through because it has PyTorch based model implementation for many transformers, including GPT-2 and AK advises using this version 

And when we load the GPT-2 model itself, we can clearly see weights of the model, that's exactly what we get i.e. [50257, 768], where the 50257 bit is the vocabulary size for GPT-2.

wte = token embeddings
wpe = positional embeddings, which is 1024 only because that is the sequence length i.e. the context window or length

> seq_len, block_size, context window, and context length all refer to the same things

As a reminder, GPT-2 was a modified version of the original AIAYN paper which was an encoder-decoder structure using cross attention from the encoder, in particular for the V vectors. GPT-2 is decoder only and uses only self attention. Two other differences:

1. The LayerNorm becomes pre instead of post
2. An additionla LayerNorm was addeed after the final self attention block

> nn.Linear(in_features, out_features) stores its weight matrix as shape [out_features, in_features] — this is PyTorch's convention, so the final lm_head component should be outputting a tensor of [vocab_size, d_model], but it'll be implemented as nn.Linear(d_model, vocab_size)

- GPT-2 uses no bias in the final projection (lm_head), I wonder why. Something to explore! 

As we dig into the transformer blocks themselves, we see something interesting. GPT-2 fuses Q, K, and V into a single combined matrix rather than having three separate ones. That is, only the input projection, which fuses all three vectors (Q, K, and V) into one i.e. 768 * 3 = 2304. But the output projection remains at d_model i.e. 768.

That's why we see something like this: 

![alt text](images/gpt-2_structure.png)

Attention is a communication mechanistic, a pooling function, something that does a sort of weighted sum or aggregates. MLP works individually on each token, so it is a mapping function! Both of them working together is kind of like an application of MapReduce. The MLP sublayer just scales up the input (d_model) 4 times up, for some reason that is probably explained in the paper. The projection matrix takes it back down, as explained in the theory.

GELU is like ReLU, except it is not exactly 0 at 0. There's a small bump. A slightly smoother ReLU. There are two versions of GeLU, the normal and approximate version, which exists for some reasons; although, there's no reason good reason today for this. A historical artifact.

> Dead ReLU neuron problem. 0 gradient at the 0 point, which isn't preferred of course

> Modern networks like LLaMa 3 has SwiGLU, which do similar things, but there's some research taste thing going on here!

AK uses register_buffer with a bias which has nothing to do with the transformer layers' biases as we're seeing and expect, the y = mx + b kind of bias. It is the causal mask that is important to make the model auto-regressive, actually! As we can see within the register buffer bit, we're creating a square matrix (seq_len by seq_len) which is trilled such that a token cannot look into the future, in the triangular matrix fashion.

It is confusingly called "bias" instead of the self attention mask; that is only historical baggage. And as for why it is added via register buffer instead of a normal tensor, it's because, in PyTorch, register_buffer is used for tensors that are stateful but not learnable.

Because it is registered, when you call model.to('cuda'), PyTorch knows to move this mask to the GPU alongside your model weights.

But, because it is a buffer and not an nn.Parameter, PyTorch knows not to calculate gradients for it. The optimizer will never try to "update" the 1s and 0s in the triangle.

## The CausalSelfAttention Forward Pass

This is the most complicated bit so far, so it is important to understand what's truly going on. We have [Batch Size, Sequence Length, d_model] (or B, T, C in the code) coming out of the c_attn bit, which is a simple Linear layer taking in d_model and then multiplying it by 3 for Q, K, and V vectors, but this is **not** just a simple 1D array. It is a [B, T, C (d_model)] dimensional. 

With this prose, we're trying to understand this particular bit of code:

```
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # For the k vector, but this operation is the same for all K, Q, and V vectors
```

> The PyTorch documentation for torch.nn.Linear specifies that the linear layer operates on the last dimension of the input tensor.

![alt text](images/pytorch.png)

So, the 3D tensor's last dimension i.e. C or d_model is the one that expands three times. So, before the view operation, our input tensor is of shape [B, T, 3*C], and this is the one that is split by n_embd in dim=2. So, we get three tensors, which need to get reshaped and transposed now.

So, taking just the k vector for instance, this is [B, T, C], where C is d_model. We also have d_head defined, so C will need to get divided by n_heads. And that is exactly what the view bit does. But it also adds the n_heads bit into the entire array, so the 3D tensor becomes 4D i.e. [B, T, n_heads, d_head], where d_head = d_model / n_heads.

The transpose is needed so dimensions 1 and 2 are swapped, which means it becomes [B, n_heads, T, d_head]. **This has to be done in order for the math, the matrix multiplication in the next step, to work out**.

In the next step of the Transformer, we have to multiply the Queries by the Keys to get our attention scores, and for this, PyTorch uses Batched Matrix Multiplication.

### The Rule of Batched Matrix Multiplication:

When PyTorch multiplies two multi-dimensional tensors together, it treats the first dimensions as a batch and only performs matrix multiplication on the last two dimensions. Let's take an example where n_head is 12 and d_head = 64 i.e. d_model = 768.

- If we DO NOT transpose:

Our shape is [B, T, 12, 64].

PyTorch would look at the last two dimensions and try to do matrix multiplication on a 12 x 64 matrix. This makes zero mathematical sense. We don't want to multiply "Heads" by "Head Size".

- If we DO transpose:

Our shape becomes [B, 12, T, 64].

Now, PyTorch looks at this and says: "Okay, I have a batch of size B * 12. For every single one of those, I need to do matrix operations on a matrix of shape [T, 64]."

In PyTorch, the @ operator calls torch.matmul.torch.matmul operates strictly based on the number of dimensions of the input tensors: if both inputs are 2D ([M, K] and [K, N]), it performs standard matrix multiplication $\rightarrow$ [M, N].

If inputs are $>2$D (e.g., 4D tensors like [B, T, n_head, d_head]), PyTorch automatically treats all leading dimensions as batch dimensions and only performs matrix multiplication on the last two dimensions.

So, if we don't transpose, we end up with an interaction matrix between Head $i$ and Head $j$, for the query and keys. The attention mechanism is trying to measure that for the *tokens*, which is why we need to transpose the n_head dimension with T, the seq_length.

> In theory, if we don't do the transpose, we'd get an interaction matrix between the heads for Q and K, which seems useful in a mechanistic interpretability sense, but maybe it isn't useful even though the linear algebra does seem to work out. Heads are Parallel, Independent Operators: In a Transformer layer, attention heads run completely independently in parallel. Head 1 has no direct wire or channel to send information to Head 2 during the attention step.Where Heads Actually Interact (The Residual Stream): Heads only interact after the attention calculation is done. Each head projects its output back into the shared residual stream via $W_O$ (c_proj), where their outputs are summed together:

#### A Simple Example

Imagine a tiny sequence of 3 tokens (T=3), an embedding size of 4, split into 2 heads, head Size = 2.

If we transpose to [B, 2, 3, 2], we have completely isolated the heads.
If you look inside Head 1, you see a matrix shaped [3, 2] (3 tokens, each with a 2-dimensional vector).

When you multiply the Query matrix [3, 2] by the transposed Key matrix [2, 3], the inner dimensions cancel out, and you are left with a [3, 3] matrix.

That [3, 3] matrix is your Attention Grid! It tells you exactly how much Token 1 attends to Token 1, Token 2, and Token 3.

Summary: We transpose so that the "Number of Heads" becomes a batch dimension, isolating each head completely. This places "Sequence 

Length" and "Head Size" at the very end of the tensor, aligning them perfectly for PyTorch to multiply the tokens against each other to calculate attention.