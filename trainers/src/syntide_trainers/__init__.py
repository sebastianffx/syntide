"""Trainers for SYNTIDE trials, outside the syntide package.

Classification uses ``syntide.trial`` to read a planned trial, consume its
batches, and save checkpoints, validation history, and probabilities through
``Trial.finish``. The question-answering consumer uses ``syntide.qa_trial``
and its task-specific inputs and outputs. ``torch_loop`` is the PyTorch
classification reference trainer.
"""

__version__ = "0.1.0"
