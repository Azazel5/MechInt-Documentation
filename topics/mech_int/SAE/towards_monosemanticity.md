## Towards Monosemanticity: Decomposing Language Models With Dictionary Learning

This paper builds on Toy Models of Superposition, which I will cover next, but spoiler alert: the three hypotehsized ways we could find sparse and interpretable features had problems with insufficiency and overfitting. However, in all that they described, they hadn't used SAEs yet, but are using it now, so this will be informative! Excitingly, they take on a one-layer transformer **with a 512 neuron MLP layer **.

Personal note -> the kind of visualizations they're making in this past and the last one are really interesting! I want to learn how to make those, as knowing how to do that, seems to instill some interesting knowledge that could lead to the insights I am looking for.

The map visualization kind of looks like the world map, quite literally, that's interesting!

The authors make some interesting observations about SAEs here:

1. Features can split if SAE width increases, kind of like a resolution
2. The MLP layer with just 512 neurons can represent tens of thousands of features
3. Features connect in "finite-state automata"-like systems that implement complex behaviors, so, not just circuits

