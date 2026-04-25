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
2. DCM (Desirada based component masking)(automatically detects model components responsible for specific semantics)
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

## The models studied in the paper

We have Llama-7B plus three fine-tuned versions of it, published by other researchers it seems.

1. Vicuna-7B, fine-tuned on user shared conversation collected from [ShareGPT](https://sharegpt.com/)
2. Goat-7B, fine-tuned on synthetically generated arithmetic expressions using [LoRA](https://arxiv.org/abs/2106.09685)
3. Float-7B, fine-tuned on the same data as Goat-7B without LoRA

### Quick explanation on LoRA

LoRA — Low-Rank Adaptation. This is a training technique that creates efficiency. Instead of training all base model weights, LoRA predicts that only a small subset of the weights need to be tweaked and the rest of the base model's weights and representations can be reused.

The base model's weights are frozen BUT are utilized in the forward pass with two randomly initialized weight matrices A and B. These are the ones that are trained and is way smaller than the base model's weights! 

The bet LoRA makes is that most tasks don't require moving all 7B parameters — they require moving a small structured subset of the representation space, and a low-rank update can capture that subset. For most tasks that bet is empirically correct. For tasks requiring dramatic behavioral change — the bet may not pay off, and full fine-tuning like FLoat-7B becomes necessary.


> In conclusion, Goat-7B used LoRA — froze the base LLaMA weights, trained only the small A and B matrices on arithmetic data. FLoat-7B did full fine-tuning on the same data — all 7B parameters updated.

The practical differences: 

1. Training cost. Goat-7B trained a fraction of the parameters — maybe 20-40M vs 7B. Significantly cheaper compute and memory. 

2. Catastrophic forgetting. FLoat-7B, having updated all weights, likely forgot more of its general capabilities. Goat-7B's frozen base weights preserve the pretrained knowledge more faithfully — only the adapter layers changed.

3. Arithmetic specialization. FLoat-7B potentially carved deeper arithmetic-specific representations into the model's core weights. Could be better at arithmetic in-distribution, worse at everything else.

With LoRA (Goat-7B):
W_attention stays frozen at pretrained values
Only A_q, B_q, A_v, B_v etc. get gradient updates
Rank r=16 means ~0.5% of parameters move

Every weight matrix — attention, MLP, embeddings — receives gradients
All 7B parameters shift toward minimizing arithmetic loss
General language modeling capability potentially degraded

```The naming convention
In attention, recall the four projection matrices:

W_Q  — query projection
W_K  — key projection  
W_V  — value projection
W_O  — output projection
```

Even though Goat and Float were fine-tuned on aritchmetic tasks, they perform better at entity tracking than base models, suggesting that models trained on structured data are better at performing entity tracking. Which makes sense because there's a lot of entity tracking to be done in math tasks, and I would imagine this translates to coding too, because coding is similar. Or chemistry. Anything requiring equations and prior steps of entities. 

My polished guess is: arithmetic fine-tuning improves sequential reasoning generally, not arithmetic specifically, because the underlying primitive being trained is entity tracking across steps — a domain-agnostic capacity that manifests in math, code, chemistry, language, and any other task requiring state maintenance across a sequence of operations.

Their measure of choice is a synthetic box tracking dataset with 300 examples, their evaluation set for the base model and fine-tuned models, I am assuming.

**The setup**

They have two versions of each prompt:

Original (xorg):   "Box A contains the apple. Box B contains the ball. 
                    What is in Box A?"  → correct answer: apple

Noise (xnoise):    "Box C contains the fork. Box X contains the pen.
                    What is in Box Q?"  → randomized, no correct answer

The noise run is deliberately meaningless — scrambled labels, scrambled objects, scrambled query. It's the corrupt run. Same concept as your clean/corrupt pairs in BizzaroWorld.

**What they're measuring**

They run the model normally on xorg and record p_org — the probability the model assigns to the correct token "apple." This is the baseline. Model is confident, probability is high.

Then they do path patching — but here's the key difference from activation patching. 

Instead of patching one node's activation, they patch a specific path between two nodes. From node A to node B, they ask: what if the signal traveling along this specific connection came from the noise run instead of the original run?

After patching that path, they measure p_patch — how confident is the model now on "apple."

**The score**

score = (p_patch - p_org) / p_org

If patching this path causes p_patch to drop far below p_org, the score is large and negative. That means this path was carrying important information — disrupting it hurt the model's ability to find the correct answer.

If the score is near zero, this path didn't matter — the information wasn't flowing through it.

**The iteration**

They do this exhaustively across all candidate node pairs — every possible path in the model. At each iteration they identify the paths with the most negative scores — the ones that hurt the most when disrupted — and add those to the circuit. They're building up the circuit greedily, one important path at a time, until they've mapped the full information flow from input to output.

Entity tracking is fundamentally a relational task. "Box A contains the apple" — the model needs to bind an entity (apple) to a location (Box A), maintain that binding across intervening text, and retrieve it when queried. That requires reading from one token position and writing to another. Cross-token communication.

As the literature confirms: MLP is token-independent. It operates on each token's residual stream vector in complete isolation. It cannot see "Box A" while processing the query token. It has no mechanism for binding or retrieving across positions.
Attention is the only component that can do this. The query token attends back to "Box A", reads its value vector, and pulls that information forward. That's the binding mechanism. That's why entity tracking is fundamentally an attention phenomenon.

## V-composition and Q-composition

These describe how attention heads communicate with each other across layers. The key insight is that one head's output can influence another head's behavior in two different ways.

V-composition — Head A's output changes what Head B reads.

Head A writes something into the residual stream. When Head B runs later, it reads from that same residual stream to compute its Value vectors. So Head A's output becomes part of what Head B retrieves and passes forward. Head A is loading information into the stream, Head B is picking it up and carrying it further. It's a content relay — one head packages information, another head transports it.

Concrete example from the paper: Group D reads the segment information at the query label token and writes it into the residual stream. Group C then computes its Value vectors from that same stream — so Group C is effectively reading what Group D wrote. Group D → Group C via V-composition means D's output becomes C's input through the shared residual stream.

Q-composition — Head A's output changes where Head B looks.

Head A writes something into the residual stream. When Head B runs later, it uses that stream to compute its Query vectors — which determine what positions it attends to. So Head A is not changing what Head B reads, it's changing where Head B looks. Head A is steering Head B's attention pattern.

From the paper: Group A uses Q-composition from Group B's output. Group B has already transported information to the final token's residual stream. Group A then uses that information in its Query computation to decide which object token to attend to — to fetch the actual answer. Group B told Group A where to look, and Group A looked there.

So the circuit is: collect → package → transport → retrieve. Four groups, four roles, connected by two types of composition.

## What accuracy means here

Two accuracy numbers are being compared:

Full-Model accuracy — run the complete model normally on entity tracking tasks. What fraction does it get right? Llama-7B gets 0.66, Goat-7B gets 0.82.

Circuit accuracy — ablate everything except the identified circuit. Zero out all attention heads not in the circuit. Run only the four groups. What fraction does the circuit alone get right? If this matches Full-Model accuracy, the circuit is sufficient to explain the behavior.

Random Circuit accuracy — ablate everything except a random set of heads of the same size as the circuit. This is the null hypothesis baseline. If random heads got 0.66 too, the circuit finding would be meaningless — any heads would do. 

The fact that random circuits get 0.00-0.01 confirms the identified circuit is specifically responsible, not just any collection of heads.

Looking at the table in the paper, the base circuit still works in fine-tuned models — the original entity tracking mechanism is preserved. But faithfulness drops because fine-tuning added capacity on top of it. The model got better at entity tracking (0.66 → 0.82) not by replacing the circuit but by building additional pathways alongside it.


*Is there no equivalent O or K composition?* There is no O-composition or K-composition as named concepts in the literature — and the reason is asymmetric to how Q and V work.

K vectors determine what each token advertises about itself — what it has to offer. When Head B computes its Keys, it's saying "here is what I represent at each position" If Head A's output influences Head B's Key computation via the residual stream, that changes what positions Head B appears to contain, not what it retrieves or where it looks.

K-composition is theoretically possible but mechanistically it's the weakest and least interpretable of the three. It changes the "label" a token presents to other heads rather than changing what information flows or where attention points. In practice researchers have found it rarely carries meaningful circuit-level information flow for the kinds of tasks studied — entity tracking, factual recall, indirect object identification. So it gets omitted not because it can't happen but because it doesn't appear to be load-bearing in the circuits found so far.

Why no O-composition

W_O is the output projection — it takes the concatenated head outputs and projects them back to d_model before adding to the residual stream. It's not an independent composition type because it's part of the attention mechanism's output stage, not a separate head that reads from the residual stream.

> There are only two meaningful ways a downstream head can be influenced by an upstream head: What it retrieves — V-composition. The content being carried forward changes. Where it looks — Q-composition. The attention pattern changes, redirecting which positions get read.

The authors assert that there may be redundancies in the relevant attention heads in the circuit that path patching finds. The reason may be because (they don't explain why) path patching ranks heads by their patching score but it's a greedy, one-at-a-time procedure. It asks "does disrupting this path hurt performance" for each path independently. It doesn't ask "given that these other heads are already in the circuit, does adding this one add anything new." 

So redundancy creeps in two ways.

Functional redundancy — two heads doing the same job. If Head A at layer 6 and Head B at layer 7 both route entity information from the same source to the same destination, patching either one individually hurts performance. Both get included. But if you ablate Head A, Head B compensates — and vice versa. Together they're not twice as important as either alone. Large models learn to compensate i.e. polysemanticity and superposition. 

Path overlap — two paths carrying the same information through different routes. The greedy path patching procedure finds both because each one individually matters. But causally only one needs to exist for the circuit to function. Including both overstates the circuit size.

Why 90 heads specifically suggests redundancy

90 heads out of Llama-7B's total — 32 layers × 32 heads = 1024 heads — is about 9% of all heads. That's a lot for a specific task like entity tracking. The researchers' intuition may be that that a truly minimal circuit for a task this structured shouldn't need 9% of the model. The bloat comes from the greedy inclusion of redundant paths.

The minimality criterion they apply — pruning heads and checking whether faithfulness drops — is the empirical test of that intuition. If removing a head doesn't hurt faithfulness, it wasn't load-bearing despite having a negative patching score. It was either redundant with another head or a false positive from the greedy procedure.

**When I do run my own path-patching experiment, I should do what the researchers did. Use a separate hold out set to test the circuitry found, through the path patching dataset**

The researchers then filtered out a bunch of these heads, if they don't contribute enough (less than 1% of the score, is the heuristic they used). They pruned the number of attention heads to 72 (from 90). 

Faithfulness is the percentage of model performance that can be measured by the identified circuit. 

## Circuit Generalization Across Fine-Tuned Models

We know that fine-tuned models perform better, but is that because of a superior circuit? Are the same circuits preserved?

The fine-tuned models utilize a superset of the base model circuitry which suggests that fine-tuning is inserting additional components to the circuitry that performs entity tracking.

Okay, this much is cool, but what about the mechanical difference between how the base and foundational models use the circuitry? 

There are two hypotheses: 

1. The same circuit exists in all four models, but the functionalities it implements may vary, accounting for the performance difference

2. The circuits of all models implement the same mechanism, but with an
enhanced functionality in fine-tuned models. To investigate these hypotheses, we use the automatic Desiderata-based Component Masking (DCM) method, introduced in [Davies et al. (2023)](https://arxiv.org/pdf/2307.03637), for identifying groups of model components responsible for specific functionalities. 

## DCM, revisited

Desiderata is a Latin term meaning "things that are desired" or "things needed," acting as the plural of desideratum. It refers to essential items, requirements, or necessities, often used to describe goals or virtues that are highly sought after.

Where DCM shines is taking my own activation patching experimentations and making it better. 

> DCM learns a binary mask over all components simultaneously. The mask is a vector of 0s and 1s — one per component — where 1 means "include this component" and 0 means "ablate it." The mask is trained via gradient descent to satisfy two objectives at once: when you patch the masked components from the alternate run into the original run, the output should match the target. The mask should be sparse — as few 1s as possible

So it's finding the minimal subset of components that collectively produce the behavior, not the individually most important components. That's the key difference.

## So how do the fine tuned versions perform better? 

### Cross Model Activation Patching, CMAP

This can be applied when I try to bring in different model architectures and repeat the experiments in BizzaroWorld. 

## Results

Patching activations of specific attention heads that the researchers have grouped with certain names from FLoat -> Llama and Goat -> Llama shows that the "position transmitter" and "value fetcher" heads are enough to take the base model's performance above baseline! The other two they've identified doesn't seem as necessary. **The output of these heads in fine-tuned models encodes an enhanced representation of the correct object**

That's an incredible finding if we think about it. If we patch activations from those identified points, take activations from the fine tuned ones and apply them to the base model, we can almost totally recover the fine-tuned models' performance! Wow. And that's basically the crux of what activation steering does and why it's so popular in MechInt right now. 

The most well-known version is [https://arxiv.org/pdf/2310.01405](representation engineering) from Zou et al. 2023 — you extract a "direction" in activation space corresponding to a concept or behavior and add it to the residual stream at inference time. No fine-tuning. No weight updates. Just a vector addition at the right layer.

We could potentially inject just those enhanced activations into the base model at inference time without ever storing fine-tuned weights. The base model does all the work, the targeted activations steer it.

> Fine-tune once, extract the circuit enhancement, deploy via activation injection at scale. Instead of storing and serving a 7B fine-tuned model, you store a small set of vectors — the delta activations at the critical circuit components — and apply them to the base model on demand. The storage cost drops from gigabytes to kilobytes.

The research directions this opens:

1. Direction 1 — cross-task activation transfer
If you fine-tune on arithmetic and extract the value fetcher enhancement, does injecting those activations into a base model also improve coding performance? Your hypothesis about sequential reasoning generalizing would predict yes. That's a testable experiment requiring no additional fine-tuning beyond the one arithmetic run.

2. Direction 2 — activation distillation
Rather than full fine-tuning, train only the activation vectors needed to steer the circuit. You're not updating weights — you're learning optimal steering vectors directly. Much cheaper than LoRA, potentially more interpretable because you know exactly which circuit component you're targeting.

3. Direction 3 — modular capability injection
Different tasks enhance different circuit components. Arithmetic enhances value fetching. A coding dataset might enhance position detection. A reasoning dataset might enhance structure reading. If each capability maps to a specific circuit component, you could mix and match activation patches to create a "capability-à-la-carte" base model with no fine-tuning at all. Just swap the steering vectors.

The problem is that this kind of activation injection only works cleanly when the circuit structure is shared — when the base model and target behavior use the same computational pathway. For tasks requiring genuinely new capabilities that don't exist in the base model at all, there's no circuit to enhance and no activation to inject. You'd need actual weight updates.

> The interesting question is: how many tasks fall into "circuit enhancement" versus "circuit creation"? This paper suggests entity tracking is enhancement. But something like coding in a new programming language the base model has never seen might be creation. The boundary between those two regimes is not yet understood.

That is really interesting and could even shed light to how we do things. What are tasks that we do that are circuit enhancement vs circuit creation? A person playing guitar for the first time? Creation. A person learning the modes after learning scales? Enhancement. A loose analogy. There are interesting boundary cases too. One can imagine tasks that fall into a hybrid, which can be both enhancement and creation. What about those? 

What all of this suggests is that the effectiveness of fine-tuning — and by extension activation injection — is fundamentally constrained by what circuits already exist in the base model. You can enhance what's there. You can extend connections between existing circuits. But you cannot create from nothing. 

**The pretraining distribution is not just important — it's load-bearing for everything downstream. The base model's circuit repertoire is the ceiling on what fine-tuning can achieve through enhancement.**

> This is essentially a theory of transfer learning grounded in mechanistic interpretability.

Future work may study the training dynamics during the fine-tuning process, to pinpoint exactly when and how the circuit enhancement occurs.

## Appendices

What exactly is Attention Knockout doing?  Attention knockout blocks that flow deliberately. It sets the attention weight from position A to position B to zero — forcing the head to ignore position B completely. The model can't read from that position anymore.

The question it asks: if I prevent the final token from attending to the entity token at layer L, does performance drop? If yes, that layer's attention from final to entity was carrying crucial information. 

Lots of implementation details and definition of desideratas. Got to read bits again.

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


4. Try an ablation study by updating only the A_ A_q, B_q, A_v, B_v first and then include the value projection and output projection matrices for A and B too. Does this result in any meaningful findings? Will this introduction make a big difference or no, and by what margin? We could do the same for the MLP layer too, W_up and W_down.

5. Use path patching on Bizzaro World to sharpen what we've already done

From my experiments, I know:

- Entity token position carries the signal in layers 0-14
- Final token position receives it at layer 15
- The residual stream is the dominant carrier throughout
- Attention heads contribute diffusely — Head 2 most consistently but not dominantly
- MLP contributes intermittently at specific layers

Path patching would let me draw the actual arrows. Not just "Head 2 matters somewhat" but "Head 2 at layer 6 sends signal specifically to the final token position via this path." That's the difference between a list of important components and an actual circuit diagram.