# ScorpionLabs Core

ScorpionLabs Core is the execution framework for a long-horizon AI research campaign.

## Mission

"A 2,000-hour execution roadmap bridging theoretical Mechanistic Interpretability with Bare-Metal ML Systems Engineering."

## Operating Doctrine

"Every topic contains a notes.md living document and an experiments/ pipeline. We do not just read papers; we replicate them, hit hardware bottlenecks, and engineer systems to bypass them."

This repository is not a passive reading list. It is a research command center designed to convert theory into reproducible artifacts, systems insight, and measurable capability.

## Three-Pillar Architecture

1. Mechanistic Interpretability
- Path: [topics/mech_int](topics/mech_int)
- Focus: Causal model understanding, circuit-level analysis, alignment-adjacent interpretability.

2. ML Systems (Bare-Metal)
- Path: [topics/mlsys](topics/mlsys)
- Focus: Memory systems, distributed training, hardware-aware kernels, and throughput-critical engineering.

3. Historical / Foundational AI
- Path: [topics/historical_ai](topics/historical_ai)
- Focus: Foundational transformers, early reinforcement learning, and historical research lineages that still shape modern systems.

## Workflow Standard

For every topic folder in [topics](topics):
- notes.md is the canonical living record for ideas, papers, failures, and decisions.
- experiments/replications/ contains faithful reproductions of core results.
- experiments/innovations/ contains extensions, optimizations, and original experiments.

Resources are indexed in [resources/list.md](resources/list.md) and must always link back to the corresponding topic notes file.

## Repository Map

- [topics](topics): Core research execution by pillar and topic.
- [resources](resources): Curated paper index and artifact links.
- [notebooks](notebooks): Interactive exploration and rapid validation.
- [src](src): Shared code, reusable utilities, and core components.
- [docs](docs): Supporting documentation.

## Quarto Sync (copy rendered posts to website)

`quarto-sync/sync-quarto.sh` copies Quarto-rendered outputs **into your website repo**.

Your website lives in a **different repo/path**, so this script intentionally requires you to pass a destination path (`--dest` or `DEST=...`).

### What it copies

- `<stem>.html`
- `<stem>_files/` (Quarto’s generated asset folder)
- `images/` (copied into `<dest>/images/`)

### Usage

```bash
./quarto-sync/sync-quarto.sh --src <dir-or-qmd> --dest <website-path> [--stem <name>] [--images <images-dir>]
```

Examples:

```bash
# If you're in the Quarto output folder and it contains exactly one .html:
./quarto-sync/sync-quarto.sh --src . --dest "/path/to/website/some/subdir"

# If you want to be explicit:
./quarto-sync/sync-quarto.sh --src ./post.qmd --stem post --images "../../../../images" --dest "/path/to/website/some/subdir"
```

### Notes

- The script uses `rsync -a --delete` so the destination mirrors your current render output.
- If it can’t find an `images/` directory by searching upward from `--src`, pass `--images` explicitly.

## Execution Expectations

- Build from first principles, but ship artifacts.
- Prioritize reproducibility over novelty theater.
- Track constraints, especially compute and memory bottlenecks.
- Turn every bottleneck into a systems design problem.
- Keep the research loop tight: read, replicate, stress, optimize, document.

## Quick Start

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Pick a pillar and one topic.
3. Read that topic's notes.md and select one resource from [resources/list.md](resources/list.md).
4. Replicate in experiments/replications/.
5. Push one improvement in experiments/innovations/ and document outcomes in notes.md.

ScorpionLabs Core exists to produce clear thinking, hard evidence, and engineered results.
