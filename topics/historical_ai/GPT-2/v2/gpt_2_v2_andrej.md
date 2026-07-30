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