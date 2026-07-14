## Let's build GPT: from scratch, in code, spelled out.

This paper will replicate GPT 2 using the *attention is all you need paper*. This is the video that came out 3 years ago, so January 2023, meaning it was only a couple months after ChatGPT. There's another video AK did, which I will talk about later, which comes a year later, so it will be a good update.

G = Generative
P = Pretrained
T = transformer

In this video, AK uses a toy dataset called TinyShakespeare, all of Shakespeare in one file. So, a good first exercise for you to do. Perhaps, as an exercise, do this thing and then convert TinyShakespeare into **scientist AI**, and see how easy or difficult it is to do.

## Tokenization

Take raw text and convert them into a sequence of numbers somehow. Here we're building a character level language model, so every character is going to be a number. This is a design method, so if we want to make something larger, like a token level architecture, then we need to make the appropriate choices there. 

AK creates some lambda functions to encode and decode text, which is super simple. Out of our character list, the 65 ASCII characters, we tokenize them to get encoded to a list of numbers and back to the original text by decoding. There are many other tokenization schemas, and this is a particularly simple one. SentencePiece or TikToken uses sub-word level encodings i.e. tokens.

And now that something works at a small scale, verified, you'd need to tokenize the entire dataset. Use PyTorch for this.

## Training

When we train transformers like language models, we don't feed everything we have into it immediately, all at once. That would be like me trying to shove 50 plates of momos all at once down the old, esteemed gullet. So, we train it in chunks (just like how I eat my momos in chunks). We sample random chunks from the training set, according to a length heuristic (block_size or context_length). These are where your researcher intuition and taste needs to come in. Computational reasons aren't the only reason why this is done, as there are actual technical reasons for this too: if do it this way, the system has a chance to learn context too.

The Y flag for training transformers, the ground truth, comes from the data itself, which is the beauty in all this.

"Consider this sequence"

"Consider"'s ground truth is "this", "Consider this"'s ground truth is "sequence". "Consider this sequence"'s ground truth is whatever would come after that, given the training data, but our prompt here cuts off right here. So if you input something like this to a mature LLM, it will reflect whatever it's training data said, case in point, Sonnet 5. I tried it and systems nowdays are so mature that it recognizes that a sequence like the one above is partial, so it basically refuses because it correctly realizes that something is missing here and that there are a million directions this sentence can go to, but that's essentially a transformer! 

Now, something to think about is the batch_size. Mini-batches of multiple chunks of text (the block_size), to keep the GPUs busy.

We create 4 x 8 matrices of stacked batch rows coming from the training data. 4 stacks of length 8 arrays, because we picked 8 to be the context length and 4 to be the batch size. This is exactly the same "context window" from LLMs, and it makes you wonder: longer context windows are desired by users but to make it larger and larger causes other problems clearly while training or inference, such as catastrophic forgetting. Makes you think: is more better, really?

The simplest NN in AK's opinion while starting with NLP is the bigram model, check the makemore or the [NN's Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) YouTube series courses for more content on this. We create a token embedding table of size, vocab_size x vocab_size, a square, using nn.Embedding. An embedding table. Every integer will index into this structure and pull out whatever index query it is. 

> An important consideration is that PyTorch expects the channel dimension to be present as the second argument, no the third, so some reshaping will be needed, which can be achieved through .view().

We sample from the last probabilities after the softmax step, so we'll get a (B, 1) tensor, and it will get concatenated into the current running stream of integers. At this point in the video, we have only just implemented a bigram model, which is stupid because we pass it K text, but only feed it the kth element and ask it to predict the k+1th element. All those previous tokens are wasted, and we'll want to add context into it.

This simplest possible, *bigram model*, after training does decrease the loss somewhat, but the text it generates if pretty bad, definitely not Shakesperean text. The tokens are not talking to each other in this model, so this will be the fix.

Because the performance of each mini-batch is pretty noisy, AK prefers to write his own loss averager helper hook function. Wrap it with the torch.no_grad if you want to only do inference and prevent gradient calculations. The way AK phrases it, this function does the calulation over the training as well as the validation step. To do this, you set your model in .eval phase and then set it into the .train mode later. Some layers have different behavior at inference time, so it is worth thinking about.

I have implemented the same bigram model to double check if I get those very numbers that AK did, and indeed, this model seems to converge around the 2.4 bit for the training loss: 2.45

For the seed text "Here we go", we get an output of:

```
Here we gof bllllaretherathstht y afofo.


Touk$CHaghid.

DUSAPrshe IUMPerst mise wisthayonis:

ORKI t tour k-ppo shiswhun wes bethin anda med:
NI, baneinge ors VabyForve, por fave
Thetas t t:
Yoristhisolin, a pr E: me, sst m ve?
forandongur; ro al t it lll, arlmu IVI wne towaimy terthe cit or ay ain
T:
sh t  
```

with the bigram model. Not that good.

## ALL WE NEED IS ATTENTION NOW

We'll need to write an attention block on our bigram model to make this work, but what needs to go in it? There's a particular mathematical trick in this which makes it computationally tractable, and this is how it works. The problem we're trying to solve is that, all the tokens in a batch, in our toy example it can be up to 8, we need them to talk to each other. The token with the 5th position shouldn't talk to the 6th, 7th, or 8th positions because they're in the future. So for every token, we have it reference all the tokens that preceded it.

The easiest way is to average all the elements preceding the token in question. The average is quite a bit lossy, but it's okay for now. **We'll look at smarter ways of doing it later**. 

> When you take the previous token, it needs to include the token in question as well

Although it is fairly intuitive to do this in explicit for loops, it is, of course, really slow. To do this fast, we do it in a smarter, matrix-based fashion, but we do is the same. We use basic matrix multiplication which is matrix A's rows dot producted over matrix B's columns, and that's it. But that;s not the entirety of the trick: here we use the torch.tril function, which "returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0". And this will result in a modified matrix which only has the lower triangle of our text ID matrix, by which, we are effectively zeroing out all the bits further from the current token! And then one of the matrices are also normalized, by which, we get the average we're looking for, instead of just a sum, which is what a dot product would give us.

There's another addition AK makes here, which is using softmax to do the weighing instead of the way we just did it (by normalizing it). Instead of the non-lower-triangular bit being 0, we'll make it -inf. And then we can do the softmax, which cleanly yeilds the same values we saw before, because it is an exponentiation of the parts divided by the sum. *The exponential of -inf is 0*. We'll use this formulation because it hints at how important the previous tokens were.

So now, instead of embedding the tokens according to vocab_size directly, we want to include our self attention layer, using the techniques we just talked about. And we'll introduce a new variable called n_embed, which is going to be d_model. So, this will be the token level embeddings now, instead of what it was before i.e. calculating logits directly from the embedding table. Now, to go from token embeddings to the logits, we'll need a linear layer. **Crucially, the C dimension will not be the same when you do the switch, the token embedding C will be d_model but the logit C will be vocab_size**.

Researchers also frequntly add the positional embeddings onto this whole cocktail, instead of just token IDs, which is also controlled by d_model (block_size x d_model). The token embeddings are simply added to the positional embedding before passing to the linear layer, which produces the logits in the forward pass.

Right now, everything is uniform. But we want every token to find other tokens interesting depending on the data. We want to gather relevant information from the path. Every single token at each position will emit the Q, K vectors. 

Query vector = what am I looking for? Key vector = what do I contain?

Dot product with Q and K. The head_size is determined from d_model/n_heads and the Q/K vectors can just be simple linear layers, no bias.

Until now, we just have the K and Q vectors, but we also need the V vector, which is matrix multiplied with the multiplication between Q and K.

> Attention is just a communication mechanism. We're doing auto-regressive transformers so our structure is that token 1 points to itself only, 2 points to itself and is pointed to by 1, so on. But this could, in theory, be applied to any sort of graph.
> Also, there is no notion of space; since attention only acts on vectors i.e. Q, K, and V, this is the reason why we need to add the positional encodings, because the tokens have no idea where they are. 
> Examples across batches are completely independent.
> If we're doing sentiment analysis, perhaps you'd want to have all the tokens talk to each other fully, so you could remove the bit where we are trilling. Another idea would be implementing the decoder block, where nodes from the future never talk to the past, as they'd give away the answer, opposite to what we're doing here, but we'd still need the trilling. In summary: Causal mask (tril) → autoregressive generation, GPT, predicting next token. No mask → encoder, classification, understanding tasks. Cross-attention, no mask → translation bridge between encoder and decoder

It is called self-attention because the K, Q, and Vs are calculated from x, the token. But, in theory, you could generate K and Q from one source and the V from another, which is another kind of architecture which works. This is called **cross-attention**. 

In the original paper, they also divide by the square root of the head_size, which is important because scaling the attention is important. This is done in order to reduce the variance of the calculation matrix, meaning it gets softmaxxed. If the variance is not maintained, softmax converges to one-hot vectors, which is obviously a problem. So, the curve becomes a peak, such that most of the attention head block will have one node, one token, which holds the majority of the value. Training this beefed up models gives us...

loss goes 4.20 → 2.39, slightly better than the plain bigram's ~2.44, which matches what AK finds at this stage (one un-multiheaded attention head barely helps yet; the big gains come later with multi-head + feedforward + blocks).

What changed in model.py:

Added Head — key/query/value linear projections (no bias, per the video), a tril causal mask, scaled dot-product attention (* C**-0.5), softmax, weighted aggregation.

BigramLanguageModel now takes n_embd and block_size (rather than AK's globals, since this project splits model/train into separate files): added position_embedding_table, wired in sa_head = Head(n_embd, n_embd, block_size), and lm_head maps n_embd → vocab_size.

generate() now crops to idx[:, -self.block_size:] before each forward pass — necessary because position_embedding_table only has block_size rows; feeding it a longer context would index out of range.

Now, we'll move to multi-head attention. This improves the outputs a bit, reducing the training/validation losses to ~2.18 (down from ~2.31 with a single head, ~2.44 for plain bigram), matching the expected trend as heads are added. 