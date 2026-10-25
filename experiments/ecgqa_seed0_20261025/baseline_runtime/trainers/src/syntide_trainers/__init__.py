"""Trainers for SYNTIDE trials, outside the syntide package.

Each trainer goes through ``syntide.trial`` only: it reads the planned trial, trains on the batches
SYNTIDE serves, and hands its checkpoints, validation history and probabilities back to
``Trial.finish``. ``torch_loop`` is the plain PyTorch reference trainer.
"""

__version__ = "0.1.0"
