# Paired contrasts across protocols (val)

Runs: runs/ptbxl-aug-sweep, runs/ptbxl-aug-simvea, runs/ptbxl-aug-simvea-whole. Checkpoint: the one validation chose. Bootstrap: 2,000 crossed draws over patients and seeds, seed 0, 95% CIs, from the first protocol's frozen rules (7329468ebc8d). Diagnostics: no frozen rule names these contrasts.

| Contrast | All records | Noisy | Clean | Seeds better, all | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|---|---|---|---|
| T6W - T6 | −0.001 [−0.003, +0.001] | +0.000 [−0.004, +0.005] | −0.002 [−0.004, +0.001] | 0/3 | +0.001 | −0.005 | −0.004 | +0.001 | +0.001 |
| T6W - A0 | +0.000 [−0.003, +0.005] | +0.004 [−0.004, +0.013] | −0.001 [−0.005, +0.004] | 1/3 | +0.004 | −0.008 | +0.001 | +0.004 | +0.001 |
| T6W - A1 | −0.001 [−0.005, +0.003] | −0.005 [−0.015, +0.004] | +0.000 [−0.005, +0.005] | 1/3 | −0.001 | −0.010 | −0.003 | +0.003 | +0.006 |
| T6W - S3 | +0.009 [+0.005, +0.013] | +0.007 [−0.004, +0.017] | +0.010 [+0.005, +0.015] | 3/3 | +0.017 | +0.002 | +0.012 | +0.005 | +0.009 |

Per class: mean over seeds of the change in each superclass's AUROC on all records.

| Contrast | Per seed, all records | Mean, all records (a / b) | Best step (a; b) |
|---|---|---|---|
| T6W - T6 | −0.001, −0.002, −0.000 | 0.9144 / 0.9157 | 1,300, 1,000, 1,100; 1,300, 1,000, 900 |
| T6W - A0 | +0.003, −0.002, −0.000 | 0.9144 / 0.9140 | 1,300, 1,000, 1,100; 1,100, 1,000, 900 |
| T6W - A1 | +0.001, −0.002, −0.002 | 0.9144 / 0.9155 | 1,300, 1,000, 1,100; 3,800, 3,500, 3,000 |
| T6W - S3 | +0.011, +0.009, +0.007 | 0.9144 / 0.9054 | 1,300, 1,000, 1,100; 3,500, 3,800, 3,800 |
