"""Exact real/generated minibatches for the offline DiffuSETS classification trial.

The trainer remains unchanged. Replace an opened Trial's ``_stream`` with a
MixedStream, then call ``write_audit`` after fit completes. The source array is
real records followed by generated records. Generated records have already been
created offline: transformed/transform_slots still describe online transforms,
and therefore remain false/zero.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from syntide.generate.stream import TrainStream


class MixedStream(TrainStream):
    """Independent, exhaustive shuffled cycles with an exact source mix per batch."""

    def __init__(self, spec, parents, tracker=None, *, n_real=8579, real_per_batch=48, seed=None):
        if spec.transform_family != "none" or spec.transform_fraction != 0:
            raise ValueError("offline mixed sampling must not apply online transforms")
        if not 0 < n_real < len(parents):
            raise ValueError("both real and generated pools must be nonempty")
        if not 0 < real_per_batch < spec.batch_size:
            raise ValueError("both real and generated batch allocations must be positive")
        if len(set(parents.record_id.tolist())) != len(parents):
            raise ValueError("training record IDs must be unique")
        super().__init__(spec, parents, tracker=tracker)
        self.n_real = int(n_real)
        self.n_generated = len(parents) - self.n_real
        self.real_per_batch = int(real_per_batch)
        self.generated_per_batch = spec.batch_size - self.real_per_batch
        self.seed = int(spec.replicate_seed if seed is None else seed)
        if self.seed < 0:
            raise ValueError("sampler seed must be nonnegative")
        self._source_perms = {}
        self.row_counts = np.zeros(len(parents), dtype=np.int64)
        self.batch_indices = np.full((spec.max_steps, spec.batch_size), -1, dtype=np.int32)

    def _positions(self, step, size, count, source):
        """Consume complete source permutations, including cross-pass batches."""
        passes, offsets = np.divmod(np.arange((step - 1) * count, step * count), size)
        positions = np.empty(count, dtype=np.int64)
        for p in np.unique(passes):
            key = (source, int(p))
            if key not in self._source_perms:
                # Independent RNG keys preserve each source order regardless of
                # the other source's size, number of passes, or batch allocation.
                self._source_perms[key] = np.random.default_rng(
                    [self.seed, 4515, source, int(p)]
                ).permutation(size)
            selected = passes == p
            positions[selected] = self._source_perms[key][offsets[selected]]
        return positions

    def parent_positions(self, step):
        if not 1 <= step <= self.spec.max_steps:
            raise ValueError("step is outside the planned training budget")
        real = self._positions(step, self.n_real, self.real_per_batch, 0)
        generated = self.n_real + self._positions(
            step, self.n_generated, self.generated_per_batch, 1
        )
        indices = np.concatenate((real, generated))
        order = np.random.default_rng([self.seed, 4515, 2, int(step)]).permutation(len(indices))
        return indices[order]

    def _batches(self):
        for batch in super()._batches():
            indices = self.parent_positions(batch.step)
            self.batch_indices[batch.step - 1] = indices
            np.add.at(self.row_counts, indices, 1)
            if self.tracker is not None and batch.step % self.spec.val_every == 0:
                self.tracker.log_scalars({
                    "budget/real_presentations": int(self.row_counts[:self.n_real].sum()),
                    "budget/generated_presentations": int(self.row_counts[self.n_real:].sum()),
                    "budget/distinct_real_records": int(np.count_nonzero(self.row_counts[:self.n_real])),
                    "budget/distinct_generated_records": int(np.count_nonzero(self.row_counts[self.n_real:])),
                }, batch.step)
            yield batch

    @property
    def exposure(self):
        return {
            **super().exposure,
            "real_presentations": int(self.row_counts[:self.n_real].sum()),
            "generated_presentations": int(self.row_counts[self.n_real:].sum()),
            "distinct_real_records": int(np.count_nonzero(self.row_counts[:self.n_real])),
            "distinct_generated_records": int(np.count_nonzero(self.row_counts[self.n_real:])),
        }

    def write_audit(self, directory):
        """Persist complete row draws and readable counts after the full run."""
        if not self.consumed or np.any(self.batch_indices < 0):
            raise ValueError("sampling audit requires every planned batch to have been served")
        real_counts = (self.batch_indices < self.n_real).sum(axis=1)
        generated_counts = self.spec.batch_size - real_counts
        if not (np.all(real_counts == self.real_per_batch)
                and np.all(generated_counts == self.generated_per_batch)):
            raise ValueError("observed batches do not match the planned source allocation")
        recovered = np.bincount(self.batch_indices.reshape(-1), minlength=len(self.parents))
        if not np.array_equal(recovered, self.row_counts):
            raise ValueError("per-record exposure differs from full sampling trace")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        trace = directory / "sampling_trace.npz"
        if trace.exists():
            with np.load(trace, allow_pickle=False) as previous:
                if (not np.array_equal(previous["indices"], self.batch_indices)
                        or not np.array_equal(previous["record_id"], self.parents.record_id)):
                    raise ValueError("existing sampling trace differs")
        else:
            with trace.open("xb") as handle:
                np.savez_compressed(handle, indices=self.batch_indices, record_id=self.parents.record_id)
        rows_path = directory / "row_exposure.csv"
        batches_path = directory / "batch_composition.csv"
        self._write_csv(rows_path, ["row_index", "record_id", "source", "presentations"], (
            [i, str(record), "real" if i < self.n_real else "generated", int(self.row_counts[i])]
            for i, record in enumerate(self.parents.record_id)
        ))
        self._write_csv(batches_path, ["step", "real", "generated", "total"], (
            [step, int(n_real), int(n_generated), self.spec.batch_size]
            for step, (n_real, n_generated) in enumerate(zip(real_counts, generated_counts, strict=True), 1)
        ))
        payload = {
            "schema_version": "ptbxl-offline-mixed-sampling@1",
            "seed": self.seed,
            "policy": "independent shuffled exhaustive cycles; exact source quota; shuffled batch positions",
            "rng_key": "numpy.default_rng([seed,4515,source,pass]); source=0 real,1 generated,2 batch shuffle",
            "n_real": self.n_real,
            "n_generated": self.n_generated,
            "real_per_batch": self.real_per_batch,
            "generated_per_batch": self.generated_per_batch,
            "every_batch_has_exact_mix": True,
            "all_rows_seen": bool(np.all(self.row_counts > 0)),
            "real_presentations_per_record_min": int(self.row_counts[:self.n_real].min()),
            "real_presentations_per_record_max": int(self.row_counts[:self.n_real].max()),
            "generated_presentations_per_record_min": int(self.row_counts[self.n_real:].min()),
            "generated_presentations_per_record_max": int(self.row_counts[self.n_real:].max()),
            "exposure": self.exposure,
            "files_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (trace, rows_path, batches_path)
            },
        }
        audit = directory / "sampling_audit.json"
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if audit.exists():
            if audit.read_text() != text:
                raise ValueError("existing sampling audit differs")
        else:
            audit.write_text(text)
        return payload

    @staticmethod
    def _write_csv(path, header, rows):
        import io

        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(header)
        writer.writerows(rows)
        text = buffer.getvalue()
        if path.exists():
            if path.read_bytes() != text.encode():
                raise ValueError(f"existing sampling audit differs: {path}")
        else:
            with path.open("x", newline="") as handle:
                handle.write(text)
