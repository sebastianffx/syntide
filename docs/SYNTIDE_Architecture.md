Aionic Labs · synthetic data design task

# SYNTIDE.

Synthetic Data Engine for temporal AI

SYNTIDE decides what to generate, keeps only what passes its checks, and measures whether it helped on untouched real data. TimeNet stores and serves every dataset it reads and writes. The plugin set shown is the PTB-XL artifact experiment.

- Dataset plugin, five per dataset
- Shared engine component
- Produced artifact
- Gate: the run stops if it fails
- Crosses the TimeNet boundary
- Feedback

**SYNTIDE owns the decisions; TimeNet owns the data.** Real records enter read-only on the left. Columns 2 and 4 read bottom to top. Accepted recipes leave on the right as a new immutable dataset that TSLM training can load like any other.

### Five plugins per dataset

Adapter, challenge spec, descriptors, generator and modality validators change per dataset. Gap detection, ledger, mixer, harness, statistics and export never import modality code.

### Coverage lives in descriptor space

The gap detector compares measured descriptors, never generator knobs, so an editor, a simulator and a learned model compete for the same target.

### Recipes, not tensors

A synthetic example is a parent, a generator version, controls and a seed. It regenerates bit-exactly, and its hash links the ledger row to the TimeNet record.

### Gates stop the run

Leakage before any training, headroom before generating, acceptance per example, and freeze before the test set is touched.

**One accepted recipe becomes a new record and four tasks.** The label-effect contract decides what is exported: diagnosis is preserved, artifact type, leads and timing are determined by the generator, and signal-quality labels are not exported because the edit invalidates them.

PTB-XL plugin set shown; recipe values are illustrative. TimeNet behaviour from [docs.timenet.ai](https://docs.timenet.ai), read 21 October 2026.