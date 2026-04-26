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

## The IOI Task




### Exercises / Experimentation log

