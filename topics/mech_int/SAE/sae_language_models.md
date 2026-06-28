## Sparse Autoencoders Find Highly Interpretable Features in Language Models

The main issue with LLM interpretability is polymenaticism of neurons and features. One of the hypothesized reasons for this is superposition (as compared to my hypothesis that perhaps it is the best way to learn i.e. it's a feature, not a bug). The SAE analysis done here is also applied to the IoI work that Wang et al. has done, so this will be understandable.

- [The codebase is here](https://github.com/HoagyC/sparse_coding)

SAEs, a form of dictionary learning, returns the a list of monosemantic features, which can be combined with OpenAI's interpretability score metric, and the way they work has been formalized since as early as 1997. 

## Architecture

It is an autoencoder with a sparsity penalty term. This is a pure machine learning task, where the model is being trained on input activations of the language model and its outputs are the uninterpretable outputs, the ground truth, and the training objective is to minimize the reconstruction error with an added regularization-like penalty term.

This is theoretically possible, but what the authors are curious about is if this actually makes the downstream results more interpretable. 

> Side note: "Error bars show 95% confidence intervals around means." always do this, all over your paper's important visuals

It has been indicated that SAEs may be more accurate in earlier layers than later layers, showing that they may not work as well for bigger models, although that remains to be seen and validated by moi.

## Causally Important Dictionary Features for IoI

On activation patching experiments, it was seen that SAEs need amount of patched features to reach a certain targetted KL divergence than other techniques, like PCA. It is also observed that the right level of sparsity is needed: 0 means ineffective, too high means lower reconstruction accuracy. For this experiment, sparsity = 2 was picked, I believe.

## Case Studies

The authors obtain the dictionary features and apply three sub-experiments to them:

1. Which tokens activate which dictionary features in which contexts
2. How ablating these features changes the downstream output logits 
3. Causal reasons for the activation of the current feature, meaning what feature upstream contributed to it activating now

All in all, a very nice profile for interpreting the tangled mess that LLM features usually are. Especially the third point, because that is literally a circuit, and dictionary features seem to allow for automated circuit detection (and then extraction). This is easy enough to do: 

`
    for l in range(layers):
        features = layers[l]
        feature_in_consideration = features[N]

        for prev_layer in range(l):
            prev_features = layers[prev_layer]
            prev_features[feature_in_consideration] = 0 # ablation
            downstream_activations = model(prev_features)
            downstream_activations.sort()
            print(f"Difference in activations = {model(features)[feature_in_consideration] - downstream_activations[feature_in_consideration]}")
`

This is the rough sketch of the Python code required to do this kind of ablation experiment, and it would show which features upstream were causally important to the feature in consideration, and doing so, would lead to a little circuit! From a circuit, drawing a graph (where strength of causal effect between dictionary features and successive residual stream layers) is a good idea to put in the BizzaroWorld paper! 

## Limitations

- Non 0 reconstruction loss
- Here the process applies to the residual stream, but **they could apply to the MLP sublayer too**, although they would require architectural innovations

## Further Reading and links

- Elhage et al., 2023, the research done by this team in general
- Sharkey et al. (2023), to train sparse autoencoders to learn these sets of directions 
- Yun et al. (2021), applying sparse dictionaries to all layers of the residual stream
- Pythia-70M and Pythia 410-M, for which the residual streams are of size $d_{in} = 512$ and $d_{in} = 1024$ 
- Bills et al. (2023), to actually measure how interpretable features are, these guys define the main metric used here as well
- Automated Circuit Discovery (ACDC) algorithm of Conmy et al. (2023), for selecting the right set of features
- An ambitious dream in the field of interpretability is enumerative safety (Elhage et al., 2022b)
