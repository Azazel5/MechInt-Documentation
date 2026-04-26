## [1.4.1] Indirect Object Identification (IOI)

- Chapter: `https://learn.arena.education/chapter1_transformer_interp/21_ioi/intro/`

### Notes

Setup is on Google Colab, at the top of every working notebook (because I will have one notebook per chapter, I will need to run this)

from google.colab import drive
drive.mount('/content/drive')
import os, sys
os.chdir('/content/drive/MyDrive/ARENA/ARENA_3.0')
sys.path.append('/content/drive/MyDrive/ARENA/ARENA_3.0')

And then start the session, so I will always be running and modifying code in the same directory.


## Interpretability in the wild

This paper is really seminal it seems. https://arxiv.org/abs/2211.00593. Multiple times mentioned in the Northeastern University paper, and this chapter will replicate exactly this. 

The model used in the paper is GPT-2 small. This chapter will give a high level overview, and the IOI part is the most technically dense, which I am excited about. That's exactly where I will find methodology to replicate this on BizzaroWorld. 

> Tip: Use torch.cuda.empty_cache() if you git CUDA memory errors

These guys are extremely organized, hats off to ARENA for making it so easy and effective for us new learners. The difficulty and importance ratings are useful, albeit not needed. Just like Michael Jordan, the aim is to focus on the little things. Little things lead to big things, where big things means the big understanding I aim to develop. So I will be doing everything, no doubt about it. Perhaps several times, if one time doesn't click. Alex Hormozi -> Do so much value, that it would be impossible for you to fail. 

Yet, I will not ignore the difficulty ratings. They will serve as signals for where to slow down and sit longer. High difficulty rating means more time time spent there. I will take care to spend more time that other sections, in a measured way, perhaps 5 hours more, as a loose heuristic

## The IOI Task (Indirect Object Identification)

Kevin Wang's paper for interpretability in the wild, what did it do? 

In the paper they rigorously reverse engineer a 26 head circuit, with 7 separate categories of heads used to perform this capability. The circuit they found roughly breaks down into three parts:

1. Identify what names are in the sentence
2. Identify which names are duplicated
3. Predict the name that is not duplicated

A video walkthrough of the paper by Neel Nanda with Wang -

[Part 1](https://www.youtube.com/watch?v=gzwj0jWbvbo)
[Part 2](https://www.youtube.com/watch?v=b9xfYBKIaX4)

The tasks were chosen this way because they're common grammatical structures that earlier layers learn, easy to measure, and it's a well defined task. 

![IOI chapter diagrams](images/ioi-diagrams.png)

Check these resources after reading the chapter! 

Path patching is the more refined and technically intensive way of going about it, compared to activation patching. 

And finally, we'll do full replication too, by recovering the minimum circuitry through which we can recover original performance. Let's use GitHub because the website actually cuts off at a certain point. 


## Architecture

GPT2-small is a 12 layer, 80M parameter model.

What are the parameters in the model loading functions?

HookedTransformer.from_pretrained()

center_unembed=
center_writing_weights=
fold_ln=

refactor_factored_attn_matrices=redefining w_q, w_k, w_o, w_v in the model without changing model behavior

We know that the only matrix computation occurring within is $W_QW_k^T$. And then we do the singular value decomposition, as a reminder, this is a fundamental matrix factorization technique in linear algebra that decomposes any $m x n$ matrix 
into three specific matrices: $W_QW_k^T = U\Sigma V^T$

> Transpose is its own inverse — applying it twice returns the original matrix. So if A.T = B, transposing both sides gives (A.T).T = B.T, which simplifies to A = B.T. Valid.

So we can write equations for the matrices individually!

> $W_q = U \sqrt \Sigma$ 

> $W_k = V \sqrt \Sigma$

This means that $W_q$ and $W_k$
 both have orthogonal columns with matching norms i.e. each column of $W_q$ has the same length (L2 norm) as the corresponding column of $W_k$. 
 
 So column 1 of $W_q$ and column 1 of $W_k$ have the same magnitude, column 2 matches column 2, and so on.

It doesn't mean all columns have the same norm as each other — just that the pairing between those columns is norm-matched.

![Figure 1](images/figure-1.png)

![Figure 2](images/figure-2.png)


### Exercises / Experimentation log

