## What to follow?

Even though this is a nascent field, there is a lot out there already. This page is a living checklist of resources, organized **by topic** so you can add papers quickly and keep study notes + experiments close to the relevant theme.

## Topic index (canonical)

- [`topics/activation_patching/`](../topics/activation_patching/) — notes: [`topics/activation_patching/notes.md`](../topics/activation_patching/notes.md)
- [`topics/alignment/`](../topics/alignment/) — notes: [`topics/alignment/notes.md`](../topics/alignment/notes.md)
- [`topics/circuit_discovery/`](../topics/circuit_discovery/) — notes: [`topics/circuit_discovery/notes.md`](../topics/circuit_discovery/notes.md)
- [`topics/persona_and_steering/`](../topics/persona_and_steering/) — notes: [`topics/persona_and_steering/notes.md`](../topics/persona_and_steering/notes.md)
- [`topics/deception_and_goal_guarding/`](../topics/deception_and_goal_guarding/) — notes: [`topics/deception_and_goal_guarding/notes.md`](../topics/deception_and_goal_guarding/notes.md)
- [`topics/elicitation/`](../topics/elicitation/) — notes: [`topics/elicitation/notes.md`](../topics/elicitation/notes.md)
- [`topics/training_data_and_distillation/`](../topics/training_data_and_distillation/) — notes: [`topics/training_data_and_distillation/notes.md`](../topics/training_data_and_distillation/notes.md)

**Suggested per-topic layout (inside each topic folder)**

- `notes.md` (reading notes, distilled takeaways, open questions)
- `code/` (repro scripts, small utilities)
- `experiments/` (notebooks, runs, outputs)

## How to add a new resource (fast path)

- Put the paper in exactly one topic table below (or create a new topic section if needed).
- Use **Tags** for cross-cutting facets (e.g. `vision`, `nlp`, `benchmark`, `theory`, `tooling`).
- Link your writeup in **Notes/Artifacts** to the relevant `topics/<topic>/...`.

## By topic

### Activation patching

| Resource Name | Link | Status | Tags | Notes/Artifacts |
|--------------|------|--------|------|-----------------|
| Localizing Model Behavior With Path Patching | https://arxiv.org/pdf/2304.05969 | New | activation-patching, circuits | [`topics/activation_patching/notes.md`](../topics/activation_patching/notes.md) |

### Alignment

| Resource Name | Link | Status | Tags | Notes/Artifacts |
|--------------|------|--------|------|-----------------|
| _Add resources here_ |  |  |  | [`topics/alignment/notes.md`](../topics/alignment/notes.md) |

### Persona and steering

| Resource Name | Link | Status | Tags | Notes/Artifacts |
|--------------|------|--------|------|-----------------|
| Persona Vectors: Monitoring and Controlling Character Traits in Language Models | https://arxiv.org/abs/2507.21509 | New | alignment, steering, monitoring, representation | [`topics/persona_and_steering/notes.md`](../topics/persona_and_steering/notes.md) |

### Deception and goal-guarding

| Resource Name | Link | Status | Tags | Notes/Artifacts |
|--------------|------|--------|------|-----------------|
| Why Do Some Language Models Fake Alignment While Others Don't? | https://arxiv.org/abs/2506.18032 | New | alignment, deception, evals, training-vs-deployment | [`topics/deception_and_goal_guarding/notes.md`](../topics/deception_and_goal_guarding/notes.md) |

### Elicitation

| Resource Name | Link | Status | Tags | Notes/Artifacts |
|--------------|------|--------|------|-----------------|
| Quantifying Elicitation of Latent Capabilities in Language Models | https://openreview.net/forum?id=Dkgx2pS4Ww | New | elicitation, fine-tuning, measurement, scaling | [`topics/elicitation/notes.md`](../topics/elicitation/notes.md) |

### Training data and distillation

| Resource Name | Link | Status | Tags | Notes/Artifacts |
|--------------|------|--------|------|-----------------|
| Subliminal Learning: Language models transmit behavioral traits via hidden signals in data | https://arxiv.org/abs/2507.14805 | New | distillation, synthetic-data, training-dynamics, alignment | [`topics/training_data_and_distillation/notes.md`](../topics/training_data_and_distillation/notes.md) |

### Circuit discovery

| Resource Name | Link | Status | Tags | Notes/Artifacts |
|--------------|------|--------|------|-----------------|
| Zoom In: An Introduction to Circuits | https://distill.pub/2020/circuits/zoom-in/ | Completed | circuits, tutorial | [`topics/circuit_discovery/notes.md`](../topics/circuit_discovery/notes.md) |
| Curve Detectors | https://distill.pub/2020/circuits/curve-detectors/ | New | circuits | [`topics/circuit_discovery/notes.md`](../topics/circuit_discovery/notes.md) |
| Thread: Circuits | https://distill.pub/2020/circuits/ | New | circuits | [`topics/circuit_discovery/notes.md`](../topics/circuit_discovery/notes.md) |
| Early vision | https://distill.pub/2020/circuits/early-vision/ | New | circuits, vision | [`topics/circuit_discovery/notes.md`](../topics/circuit_discovery/notes.md) |
| Equivalence | https://distill.pub/2020/circuits/equivariance/ | New | circuits | [`topics/circuit_discovery/notes.md`](../topics/circuit_discovery/notes.md) |

### Uncategorized / inbox (triage later)

| Resource Name | Link | Status | Tags | Notes/Artifacts |
|--------------|------|--------|------|-----------------|
| Fine-tuning Enhances Existing Mechanisms: A Case Study On Entity Tracking | https://arxiv.org/pdf/2402.14811 | Processing | nlp |  |
| Entity Tracking in Language Models | https://arxiv.org/pdf/2305.02363 | New | nlp |  |
| Discovering Variable Binding Circuitry with Desiderata | https://arxiv.org/pdf/2307.03637 | New | nlp, circuits |  |
| Visualizing and understanding RNNs | https://arxiv.org/pdf/1506.02078 | New | visualization |  |
| Visualizing Higher-Layer Features of a Deep Network | resources/papers/visualization_techreport.pdf | New | vision, visualization |  |
| Feature Visualization | https://distill.pub/2017feature-visualization/ | New | vision, visualization |  |
| Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps | https://arxiv.org/pdf/1312.6034 | New | vision, visualization |  |
| Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images | https://arxiv.org/pdf/1412.1897 | New | vision, adversarial |  |
| Inceptionism: Going Deeper into Neural Networks | https://research.google/blog/inceptionism-going-deeper-into-neural-networks/ | New | vision, visualization |  |
| Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space | https://arxiv.org/pdf/1612.00005 | New | vision, generative |  |
| Visualizing and Understanding Convolutional Networks | https://arxiv.org/pdf/1311.2901 | New | vision, visualization |  |
| Interpretable Explanations of Black Boxes by Meaningful Perturbation | https://arxiv.org/pdf/1704.03296 | New | attribution |  |
| Learning how to explain neural networks: patternnet and patternattribution | https://arxiv.org/pdf/1705.05598 | New | attribution |  |
| Visualizing and Measuring the Geometry of BERT | https://arxiv.org/pdf/1906.02715 | New | nlp, geometry |  |
| Exploring Neural Networks with Activation Atlases | https://distill.pub/2019/activation-atlas/ | New | visualization |  |
| Visualizing Neural Networks with the Grand Tour | https://distill.pub/2020/grand-tour/ | New | visualization |  |
| Systems Biology (Motifs) | https://www.taylorfrancis.com/books/mono/10.1201/9781420011432/introduction-systems-biology-uri-alon | New | background |  |
| Distributed Representations of Words and Phrases and their Compositionality | https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf | New | nlp, embeddings |  |
| Learning to Generate Reviews and Discovering Sentiment | https://arxiv.org/pdf/1704.01444 | New | nlp |  |
| Object Detectors Emerge in Deep Scene CNNs | https://arxiv.org/pdf/1412.6856 | New | vision |  |
| Network Dissection: Quantifying Interpretability of Deep Visual Representations | https://arxiv.org/pdf/1704.05796 | New | vision, interpretability |  |
| Measuring the tendency of CNNs to Learn Surface Statistical Regularities | https://arxiv.org/pdf/1711.11561 | New | vision |  |
| ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness | https://arxiv.org/pdf/1811.12231 | New | vision, robustness |  |
| Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet | https://arxiv.org/pdf/1904.00760 | New | vision |  |
| Adversarial Examples Are Not Bugs, They Are Features | https://arxiv.org/pdf/1905.02175 | New | robustness, adversarial |  |
| On the importance of single directions for generalization | https://arxiv.org/pdf/1803.06959 | New | theory |  |
| The structure of scientific revolutions | https://www.bibliovault.org/BV.landing.epl?ISBN=9780226458106 | New | background |  |
| Convergent Learning: do neural networks learn the same things? | https://arxiv.org/pdf/1511.07543 | New | representation |  |
| SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability | https://proceedings.neurips.cc/paper_files/paper/2017/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf | New | representation, tooling |  |