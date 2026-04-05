# Circuit Discovery

## Overview

So, I have been focused on training and deploying neural networks, the field of mechanistic interpretability flips that on its head. The goal is NOT to minimize a cost function any more. It is to find features which correlate to something specific. The neurons that fire, that's what we want to find.

One neuron may encode multiple different concepts. In line with your gut, it was right. How can a small thing like a neural network encode concepts about the whole world? I was thinking, one tiny piece must encompass multiple things. My thinking, not computer sciency, but absolutely right in the common sense gut thinking ways. So, we know this is the truth empirically. **Polysemanticity**


## Zoom In: An Introduction to Circuits

The three speculative claims about neural networks:

1. Features are the fundamental unit of neural networks.
They correspond to directions. These features can be rigorously studied and understood

2. Features are connected by weights, forming circuits. 2
These circuits can also be rigorously studied and understood. A connected graph type diagram comes to mind. One can imagine a prompt, where every token causes a series of feature to light up, the "circuits"

3. Universality. Analogous features and circuits form across models and tasks

## Features

Take the humble CNN. Us students of AI, before LLMs because "AI", have all seen those nice visuals in textbooks. "What does a neural network learn in between layers?", and I had never asked the question, "how did they draw those figures?". Well, they use MechInt to do it. 

**Do curve detectors really detect curves?**

They offer 7 arguments for why they do, and they make sense. *Causal link*. *Pose invariance*, can the image be classified or created even when the traditional pose is rotated or flipped? 

## Circuits

These are sub-graphs of the network, consisting a set of tightly linked features and the weights between them.

And this is the most unbelievable point which has prompted me to dive head first into this subfield: 

> The remarkable thing is how tractable and meaningful these circuits seem to be as objects of study. When we began looking, we expected to find something quite messy. Instead, we’ve found beautiful rich structures, often with symmetry to them. Once you understand what features they’re connecting together, the individual floating point number weights in your neural network become meaningful! You can literally read meaningful algorithms off of the weights


So, neural networks when they have to recognize dogs, for example, would need to distinguish between a hundred different species of dogs! And so, unsurprisingly, it develops a large number of neurons dedicated to recognizing dog related features, including heads.

Within this “dog recognition” system, one circuit strikes us as particularly interesting: a collection of neurons that handle dog heads facing to the left and dog heads facing to the right. Over three layers, the network maintains two mirrored pathways, detecting analogous units facing to the left and to the right. At each step, these pathways try to inhibit each other, sharpening the contrast. Finally, it creates invariant neurons which respond to both pathways.

So, the network learns two face sides separately and combines them together in kind of like a set union to learn that invariancy. We’re somewhat surprised that gradient descent could learn to do this!

Every connection between neurons is a convolution, so we can also look at where an input neuron excites the the next one.

In mixed4c, a mid-late layer of InceptionV1, there is a car detecting neuron. In previous layers, it looks for wheels at the bottom of the convolutional window, and windows at the top. But then the model does something surprising. Rather than create another pure car detector at the next layer, it spreads its car feature over a number of neurons that seem to primarily be doing something else — in particular, dog detectors.

This circuit suggests that polysemantic neurons are, in some sense, deliberate. That is, you could imagine a world where the process of detecting cars and dogs was deeply intertwined in the model for some reason, and as a result polysemantic neurons were difficult to avoid. But what we’re seeing here is that the model had a “pure neuron” and then mixed it up with other features. This is called superposition. **The fact that the networks seems to do this deliberately needs to be explored. Why? What does that result in? What can we learn from this?**

One hypothetical answer for why it occurs is that superposition may allow the model to use fewer neurons, conserving them for more important tasks. As long as cars and dogs don’t co-occur, the model can accurately retrieve the dog feature in a later layer, allowing it to store the feature without dedicating a neuron.

In biology, a circuit motif is a recurring pattern in complex graphs like transcription networks or biological neural networks. Motifs are helpful because understanding one motif can give researchers leverage on all graphs where it occurs.

We think it’s quite likely that studying motifs will be important in understanding the circuits of artificial neural networks. In the long run, it may be more important than the study of individual circuits. At the same time, we expect investigations of motifs to be well served by us first building up a solid foundation of well understood circuits first.

## Universality

A Gabor filter is a linear bandpass filter used in image processing for edge detection, feature extraction, and texture analysis. And this is what the first layer of vision models learn. Ideally, one would like to characterize several features and then rigorously demonstrate that those features — and not just correlated ones — are forming across many models

These results have led us to suspect that the universality hypothesis is likely true, but further work will be needed to understand if the apparent universality of some low-level vision features is the exception or the rule.

One particularly exciting possibility might be if artificial neural networks could predict features which were previously unknown but could then be found in biology. (Some neuroscientists we have spoken to have suggested that high-low frequency detectors might be a candidate for this.) If such a prediction could be made, it would be extremely strong evidence for the universality hypothesis.