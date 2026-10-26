# Paired contrasts across protocols (test)

Runs: runs/ptbxl-aug-sweep, runs/ptbxl-aug-simvea. Checkpoint: the one validation chose. Bootstrap: 2,000 crossed draws over patients and seeds, seed 0, 95% CIs, from the first protocol's frozen rules (7329468ebc8d). Diagnostics: no frozen rule names these contrasts.

| Contrast | All records | Noisy | Clean | Seeds better, all | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|---|---|---|---|
| T6 - A0 | +0.003 [+0.000, +0.006] | +0.002 [−0.004, +0.010] | +0.004 [+0.001, +0.007] | 3/3 | +0.005 | +0.001 | +0.005 | +0.004 | +0.001 |
| T6 - A1 | −0.001 [−0.006, +0.003] | −0.005 [−0.015, +0.007] | −0.000 [−0.005, +0.004] | 2/3 | −0.001 | −0.003 | −0.006 | −0.001 | +0.004 |
| T6 - T4 | −0.003 [−0.007, +0.000] | −0.003 [−0.013, +0.008] | −0.003 [−0.007, +0.002] | 0/3 | +0.006 | −0.005 | −0.010 | −0.005 | −0.002 |
| T4 - A0 | +0.006 [+0.003, +0.010] | +0.005 [−0.004, +0.013] | +0.006 [+0.002, +0.011] | 3/3 | −0.001 | +0.005 | +0.015 | +0.009 | +0.003 |

Per class: mean over seeds of the change in each superclass's AUROC on all records.

| Contrast | Per seed, all records | Mean, all records (a / b) | Best step (a; b) |
|---|---|---|---|
| T6 - A0 | +0.005, +0.002, +0.003 | 0.9098 / 0.9066 | 1,300, 1,000, 900; 1,100, 1,000, 900 |
| T6 - A1 | +0.001, +0.000, −0.005 | 0.9098 / 0.9111 | 1,300, 1,000, 900; 3,800, 3,500, 3,000 |
| T6 - T4 | −0.003, −0.002, −0.004 | 0.9098 / 0.9129 | 1,300, 1,000, 900; 1,500, 1,900, 1,600 |
| T4 - A0 | +0.008, +0.004, +0.007 | 0.9129 / 0.9066 | 1,500, 1,900, 1,600; 1,100, 1,000, 900 |
