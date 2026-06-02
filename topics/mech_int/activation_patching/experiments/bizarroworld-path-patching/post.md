# Mechanistic Interpretability in four bullet points

1. DISCOVER — activation patching
   "which layers and heads matter at all?"
   → broad strokes, node level

2. TRACE — path patching  
   "which specific edges connect which heads?"
   → fine grained, edge level
   → produces the circuit diagram

3. VALIDATE — mean ablation / circuit extraction
   "is this circuit sufficient and minimal?"
   → ablate everything outside the circuit
   → if performance holds → sufficient
   → remove heads one by one → if performance drops → minimal
   → if both hold → you found THE circuit

4. CHARACTERIZE — weight analysis (W_OV, W_QK, copying scores)
   "what algorithm does each head implement?"
   → mechanistic explanation not just causal graph


This post will be taking my BizzaroWorld experimentation to the next step by doing path patching, as the ARENA course and Wang et al. do it. The first order of business was designing the IoI dataset as described in the papers and content. 

Fortunately, my design of the fact battery was good enough that this was easy enough. Categories of facts (geography, chemistry, etc) work as "groups", and I parsed these. Then, instead of finding indirect names, I wrote logic that finds token positions for the entities and targets. 

The next move was to copy all the existing functions from the content which do Step 2 i.e. computing logit differences as the metric, patching heads and freezing everything else, all of this is the same, so I copied the logic. 

> Personal note: run python files on Colab normally through !python file.py as usual!
