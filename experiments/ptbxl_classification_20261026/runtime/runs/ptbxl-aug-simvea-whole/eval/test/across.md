# Paired contrasts across protocols (test)

Runs: runs/ptbxl-aug-sweep, runs/ptbxl-aug-simvea, runs/ptbxl-aug-simvea-whole. Checkpoint: the one validation chose. Bootstrap: 2,000 crossed draws over patients and seeds, seed 0, 95% CIs, from the first protocol's frozen rules (7329468ebc8d). Diagnostics: no frozen rule names these contrasts.

| Contrast | All records | Noisy | Clean | Seeds better, all | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|---|---|---|---|
| T6W - T6 | −0.001 [−0.003, +0.003] | −0.002 [−0.008, +0.005] | −0.000 [−0.003, +0.003] | 1/3 | +0.002 | −0.006 | −0.000 | +0.001 | +0.000 |
| T6W - A0 | +0.003 [−0.002, +0.007] | +0.000 [−0.008, +0.009] | +0.003 [−0.001, +0.008] | 2/3 | +0.007 | −0.006 | +0.004 | +0.006 | +0.001 |
| T6W - A1 | −0.002 [−0.006, +0.002] | −0.007 [−0.016, +0.003] | −0.000 [−0.005, +0.004] | 0/3 | +0.001 | −0.009 | −0.006 | +0.000 | +0.004 |
| T6W - S3 | +0.008 [+0.003, +0.012] | +0.004 [−0.006, +0.014] | +0.009 [+0.004, +0.013] | 3/3 | +0.016 | −0.007 | +0.017 | +0.005 | +0.006 |

Per class: mean over seeds of the change in each superclass's AUROC on all records.

| Contrast | Per seed, all records | Mean, all records (a / b) | Best step (a; b) |
|---|---|---|---|
| T6W - T6 | −0.001, −0.003, +0.002 | 0.9092 / 0.9098 | 1,300, 1,000, 1,100; 1,300, 1,000, 900 |
| T6W - A0 | +0.004, −0.001, +0.005 | 0.9092 / 0.9066 | 1,300, 1,000, 1,100; 1,100, 1,000, 900 |
| T6W - A1 | −0.000, −0.002, −0.003 | 0.9092 / 0.9111 | 1,300, 1,000, 1,100; 3,800, 3,500, 3,000 |
| T6W - S3 | +0.007, +0.007, +0.008 | 0.9092 / 0.9017 | 1,300, 1,000, 1,100; 3,500, 3,800, 3,800 |
