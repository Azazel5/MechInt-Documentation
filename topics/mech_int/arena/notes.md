# ARENA — Transformer Interpretability (study navigation)

Goal: use ARENA as a **hands-on curriculum** to level up BizzaroWorld mech-interp execution (hooks → patching → circuits → SAE feature tooling). This is a **navigation + completion tracker** only (links + why it matters), not a copy of course content.

Primary source hub: https://learn.arena.education/chapter1_transformer_interp/

## Why this matters for BizzaroWorld

- **Fast feedback loop**: ARENA exercises force you to ship working hooks/patching code, which is exactly what BizzaroWorld needs (less “read about it”, more “instrument the model and measure causal effect”).
- **Shared abstractions**: IOI / path patching → the same mental model as activation patching + graph edges you’ll need for entity/state tracking and counterfactual interventions.
- **SAE tooling**: SAEs are becoming a standard interface for “feature-level” analysis and monitoring; even if SAEs aren’t the final answer, they’re an important *workflow primitive* (feature search, dashboards, steering, circuit tracing).

## Priority (surgical, do-now subset)

### Track A — IOI (Chapter 1, section 1.4.1) — do first

ARENA page: https://learn.arena.education/chapter1_transformer_interp/21_ioi/intro/

Objective: get **path patching** + **rigorous circuit validation** into your muscle memory, then immediately map it onto BizzaroWorld-style clean/corrupt pairs.

- [ ] **1️⃣ Model & task setup**
  - Why: reproduce “metric first” discipline (logit diff style metrics) before touching interventions.
- [ ] **2️⃣ Logit attribution**
  - Why: quick component triage; learn which heads write useful signal before doing causal work.
- [ ] **3️⃣ Activation patching**
  - Why: your baseline causal intervention; maps directly to “swap clean/corrupt activation at site X.”
- [ ] **4️⃣ Path patching**
  - Why: the key upgrade for BizzaroWorld. This is where you start drawing **actual arrows** (edge-level causal claims).
- [ ] **5️⃣ Minimal circuits / replication cleanup**
  - Why: practice “faithfulness / completeness / minimality” thinking; prevents overfitting to pretty plots.
- [ ] **6️⃣ Bonus / anomalies**
  - Why: only after Track A is operational; use as a capstone when you want more depth.

### Track B — SAE essentials (Chapter 1, section 1.3.3) — do next

ARENA page: https://learn.arena.education/chapter1_transformer_interp/13_saes/

- [ ] **Intro to SAE interpretability**
  - Why: learn the standard workflow (load SAE, inspect latents, basic attribution/ablation patterns).
- [ ] **Training & evaluating SAEs (optional, later)**
  - Why: only if BizzaroWorld needs custom SAEs; otherwise focus on using existing releases first.

### Track C — SAE circuits (Chapter 1, section 1.4.2) — return later

ARENA page: https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/

- [ ] **Latent-to-latent connections + circuit tracing with SAE features**
  - Why: translate “circuit as heads/edges” into “circuit as features communicating across layers.”

## The rest of Chapter 1 (defer until Track A is done)

Chapter overview (all sections): https://learn.arena.education/chapter1_transformer_interp/

- [ ] 1.1 Transformers from Scratch
- [ ] 1.2 Intro to Mech Interp
- [ ] 1.3.1 Linear Probes (Coup deception)
- [ ] 1.3.2 Function Vectors & Model Steering (nnsight)
- [ ] 1.3.4 Activation Oracles
- [ ] 1.5.x Brackets / Grokking / OthelloGPT / Superposition

## Key papers to read alongside (tie-in schedule)

- [ ] **Wang et al. 2022 — Interpretability in the Wild (IOI)**  
  https://arxiv.org/abs/2211.00593  
  When: **before IOI section 4 (path patching)**  
  Why: this is the canonical “IOI circuit” target; helps you know what “good replication” looks like.

- [ ] **Cunningham et al. 2023 — Sparse Autoencoders Find Highly Interpretable Features in Language Models**  
  https://arxiv.org/abs/2309.08600  
  When: **before SAE Track B**  
  Why: anchors the SAE worldview (superposition → sparse dictionary features → causal tests).

- [ ] **Conmy et al. 2023 — Towards Automated Circuit Discovery (ACDC)**  
  https://arxiv.org/abs/2304.14997  
  When: **after IOI Track A**  
  Why: connects your manual circuit workflow to automated circuit discovery; relevant to scaling BizzaroWorld.

## Completion log (quick view)

| Item | Link | Status | Notes |
|------|------|--------|-------|
| ARENA Chapter 1 hub | https://learn.arena.education/chapter1_transformer_interp/ | New | Use only for navigation |
| IOI (1.4.1) | https://learn.arena.education/chapter1_transformer_interp/21_ioi/intro/ | New | Track A (surgical) |
| SAEs (1.3.3) | https://learn.arena.education/chapter1_transformer_interp/13_saes/ | New | Track B |
| SAE circuits (1.4.2) | https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/ | New | Track C |
| Wang et al. 2022 (IOI paper) | https://arxiv.org/abs/2211.00593 | New | Read before IOI §4 |
| Cunningham et al. 2023 (SAEs) | https://arxiv.org/abs/2309.08600 | New | Read before SAEs |
| Conmy et al. 2023 (ACDC) | https://arxiv.org/abs/2304.14997 | New | Read after IOI |

