# Paired contrasts across protocols (val)

Runs: runs/ptbxl-aug-sweep, runs/ptbxl-aug-simvea. Checkpoint: the one validation chose. Bootstrap: 2,000 crossed draws over patients and seeds, seed 0, 95% CIs, from the first protocol's frozen rules (7329468ebc8d). Diagnostics: no frozen rule names these contrasts.

| Contrast | All records | Noisy | Clean | Seeds better, all | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|---|---|---|---|
| T6 - A0 | +0.002 [−0.001, +0.006] | +0.004 [−0.002, +0.011] | +0.001 [−0.002, +0.005] | 2/3 | +0.003 | −0.003 | +0.004 | +0.004 | +0.000 |
| T6 - A1 | +0.000 [−0.004, +0.004] | −0.005 [−0.014, +0.003] | +0.002 [−0.003, +0.007] | 2/3 | −0.002 | −0.005 | +0.000 | +0.002 | +0.005 |
| T6 - T4 | −0.003 [−0.006, +0.001] | −0.005 [−0.013, +0.003] | −0.002 [−0.007, +0.003] | 0/3 | +0.004 | −0.008 | −0.007 | −0.001 | −0.001 |
| T4 - A0 | +0.004 [+0.001, +0.008] | +0.008 [+0.002, +0.015] | +0.003 [−0.001, +0.007] | 3/3 | −0.001 | +0.006 | +0.012 | +0.005 | +0.001 |

Per class: mean over seeds of the change in each superclass's AUROC on all records.

| Contrast | Per seed, all records | Mean, all records (a / b) | Best step (a; b) |
|---|---|---|---|
| T6 - A0 | +0.005, +0.001, −0.000 | 0.9157 / 0.9140 | 1,300, 1,000, 900; 1,100, 1,000, 900 |
| T6 - A1 | +0.002, +0.000, −0.002 | 0.9157 / 0.9155 | 1,300, 1,000, 900; 3,800, 3,500, 3,000 |
| T6 - T4 | −0.002, −0.002, −0.004 | 0.9157 / 0.9182 | 1,300, 1,000, 900; 1,500, 1,900, 1,600 |
| T4 - A0 | +0.006, +0.003, +0.004 | 0.9182 / 0.9140 | 1,500, 1,900, 1,600; 1,100, 1,000, 900 |
