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

