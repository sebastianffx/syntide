"""ECG question-answer-rationale generation grounded in verified evidence (``python -m syntide.generate.qa``).

``evidence`` measures waveforms; ``templates`` / ``verify`` turn evidence into checked tasks; ``evol``
expands them (TSEvol); ``acquisition``, ``pairs`` and ``generated`` add transformed, paired and generated
waveforms; ``allocation`` and ``curriculum`` target and order them; ``corpus`` / ``build`` publish
immutable corpora. Nothing here imports torch or a trainer.
"""
