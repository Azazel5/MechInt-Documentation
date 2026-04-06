# Activation Patching


## Overview

Why is it that fine-tuning LLMs has been so effective across tasks? Do we know deterministically **what happens** when we fine-tune? Not clearly... This is exactly what the researc hers set out to explore in this paper. Specifically, they're looking at [entity tracking](https://arxiv.org/pdf/2305.02363), tracking entities or objects throughout the context window. It is the model's ability to maintain a persistent internal representation of specific characters, objects, or variables, and correctly update their states or relationships as a narrative or problem unfolds. 

Interestingly, language models get better at entity tracking as they're trained on mathematics. The authors find that the circuits responsible for entity tracking asre the same in both the original model and the fine-tuned version, except the latter is better because of "its improved ability to handle the augmented positional
information". In simple terms, LLMs read entire sentences at once and are not linear readers like humans. So engineers add positional tags, embeddings, to every single word. 

Take for example, this prompt: "the founder grabbed his laptop, left the office, and walked to the cafe. He opened it"

What are the entities here? It could be the founder, laptop, office, or the cafe. Here, the model would implicitly know given rules of language that a office cannot open the cafe for example, so it has to be that the founder opened the laptop in the cafe. However, things aren't so simple in mathematics or the coding world.

Prompt 2: "Let x = 5. Let y = 10. We change x to be 3. Then x = x + y."

There are no contextual clues in such a problem, and the positional embeddings take on paramount importance for solving this. So, forcing the model to do math acts like a boot camp for its internal positional tracking circuit.

## Techniques used

1. [Path Patching](https://arxiv.org/pdf/2304.05969)
2. DCM (automatically detects model components responsible for specific semantics)
3. CMAP (a new approach for patching activations across models to reveal improved
mechanisms)


## Circuits and Graphs

Directed Acyclic Graphs (DAGs) can be generated, ones that track how circuits operate. This is a cool thing you must implement. Just knowing how to do it, and what it represents is super awesome. The components would be:


1. Nodes: The specific components of the model (e.g., Attention Head 4 in Layer 10, or the MLP in Layer 15)

2. Edges: The residual stream passing mathematical information from one node to the next


Turns out, activation patching is a more general technique used in the field, compared to what's used in the paper, which is more surgical: path patching, described in the paper linked above. Activation patching would be freezing a model at a certain layer and then swapping out the entire activation state of that layer with a counterfactual, and see if the output changes. 

Path patching is more like swapping the edge between two nodes. "What happens if we sever the wire connecting Head A to Head B, and feed Head B a fake signal from a different prompt, while leaving the rest of the network perfectly intact?" Using these techniques, they isolated the exact 72 heads that actually matter for entity tracking; that's pretty high level and neat actually. When the model was fine-tuned on math problems, the experimenters found that the new model doesn't generate new graphs or circuitry! It sharpens the same ones they found before. This seems weirdly analogous to how humans learn too. If I play Money for Nothing by Dire Straits 5000 times, I will become very good at that song too. 

Just this circuitry alone can restore 88% of the capabilities of the original model. The remaining needs other components. But, the larger point remains: we have found a general purpose circuit for entity tracking!

## DCM 


We've found the circuitry, but we do not know how it works. 

> Can we discern the role of each group of attention heads?

Here is where DCM comes in, a technique where we pass carefully engineered prompts that specifically change the specific entitity semantics, such as the entity name, and this is targeted against something known about the entity. Here's an example to make this clear:


Base Prompt: "Alice placed the diamond in the vault. Bob opened the vault and saw the [__]." > (Target Output: "diamond")

Here, we suspect that different groups of attention heads are responsible for different semantics. We design "alternations" to test each hypothesis.

#### Hypothesis 1: The "Name" Tracker

We want to find the heads that care strictly about the Identity/Name of the entity. We create an alternation where the action remains identical, but the identity changes.

- Base: "Alice placed the diamond in the vault."

- Alternation: "Charlie placed the diamond in the vault."

The Masking Test: We run the Base prompt, but we force (mask) a specific group of heads to process the Alternation prompt instead. If the final output suddenly gets confused about who is doing the action, but still knows a diamond is involved, we just found our "Name Tracking" heads.

#### Hypothesis 2: The "Property/Object" Tracker

Similarly, let's try to find the heads that care about the property/object.

- Base: "Alice placed the diamond in the vault."

- Alternation: "Alice placed the kangaroo in the vault."


Result A: If the model still predicts "diamond", the trick failed. Those heads we patched don't care about the object. They ignored the "kangaroo" whisper.

Result B (Success!): If the model's final prediction suddenly flips to "kangaroo", we found them. Those specific heads are the "Property/Object Trackers." We know this because injecting the kangaroo math into only those heads was enough to hijack the entire model's final conclusion.


**Note - Here, isolation is key. If a group of attention heads changes both the name and the object, it is a polysemantic mess. It isn't a dedicated "Object Tracker." We want as much isolation as possible.**

**To be clear, in these prompts, the model is only predicting the very last token. The prompt is hardcoded right up to the very edge of the cliff**

To prove a head only tracks objects and not names, researchers don't just run one test. They run a matrix of tests to prove orthogonality (that the two features don't interfere with each other).

## Some interesting observations

Entity tracking is performed by identifying and transporting the position of the queried entity in the context, with multiple groups of heads collaborating to pass the position downstream. Fine-tuning preserves the overall mechanism for performing the entity tracking task because scheme and role between each group of heads remain the same between models. 

So if the mechanism remains the same, what about the performance improvement in the fine tuned model? CMAP (Cross Model Activation Patching) enters the chat. Patching components of the entity tracking circuits from the fine-tuned model to the original model leads to enhanced performance (but how much? Not mentioned... Find out! Maybe this offers answers: https://finetuning.baulab.info/). 

** The related work section in this paper has so many other interesting papers on mechanistic interpretability and fine-tuning. Make sure to read these too; at that point, some organization will be needed in the list.md document that will go in resources. Perhaps those will need to move to specific folders each too **

## Next steps for experimentation

Theory exists to serve implementation, so let's use this paper as the springboard that connects my earlier reading (Zoom In) with these techniques, which will be replicated on BizzoWorld as well. 

Where to go after reading these:

1. Implement CMAP on a base + fine-tuned model pair (e.g., GPT-2 base vs instructed version)

    Patch activations from one into the other
    Measure task performance
    Identify whether the same circuits exist

2. Replicate the entity tracking task that the researchers used on a smaller model

    Identify the circuit responsible
    Patch it and see if it survives fine-tuning

3. Build a "circuit comparison" pipeline

    Compare which components activate for the same task in different model versions
    This directly follows from the patching work in Zoom In

4. Because they did not mention by what degree the patching from fine-tuned model to the original, base model improves entity-tracking, find out! Replacate everything with LLaMa 7B

If we can understand why a model works, we can predict how it will change and even transplant mechanisms between models.