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