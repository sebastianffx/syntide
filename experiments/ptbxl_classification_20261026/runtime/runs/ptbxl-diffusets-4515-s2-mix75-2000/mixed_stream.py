"""Exact 48-real/16-generated batches, with native S2 on 12 real and 4 generated.

Replace an opened Trial._stream; pass its tracker and _ledger unchanged. The native
TrainStream applies, counts and records every augmentation. Each source population
is independently shuffled and exhausted before cycling. Diagnostic targets are unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from syntide.generate.stream import TrainStream

S2_CONFIG = {
    "gain": [0.95, 1.05], "max_shift_s": 0.25,
    "noise_rms_mv": [0.0, 0.05], "wander_mv": [0.0, 1.75],
    "wander_hz": [0.05, 0.7],
}


class MixedStream(TrainStream):
    """Independent, exhaustive shuffled cycles with an exact source mix per batch."""

    def __init__(self, spec, parents, tracker=None, ledger=None, *, n_real=8579, real_per_batch=48):
        if (spec.transform_family != "generic" or spec.transform_fraction != 0.25
                or spec.transform != S2_CONFIG):
            raise ValueError("mixed S2 requires the native half-strength envelope on one quarter of each batch")
        if real_per_batch % 4 or (spec.batch_size - real_per_batch) % 4:
            raise ValueError("each source allocation must divide by four for exact stratified S2 dosage")
        if not 0 < n_real < len(parents):
            raise ValueError("both real and generated pools must be nonempty")
        if not 0 < real_per_batch < spec.batch_size:
            raise ValueError("both real and generated batch allocations must be positive")
        if len(set(parents.record_id.tolist())) != len(parents):
            raise ValueError("training record IDs must be unique")
        super().__init__(spec, parents, tracker=tracker, ledger=ledger)
        self.n_real = int(n_real)
        self.n_generated = len(parents) - self.n_real
        self.real_per_batch = int(real_per_batch)
        self.generated_per_batch = spec.batch_size - self.real_per_batch
        self.seed = int(spec.replicate_seed)
        self.real_augmented_per_batch = self.real_per_batch // 4
        self.generated_augmented_per_batch = self.generated_per_batch // 4
        if self.seed < 0:
            raise ValueError("sampler seed must be nonnegative")
        self._source_perms = {}
        self.row_counts = np.zeros(len(parents), dtype=np.int64)
        self.batch_indices = np.full((spec.max_steps, spec.batch_size), -1, dtype=np.int32)
        self.batch_transformed = np.zeros((spec.max_steps, spec.batch_size), dtype=bool)
        self.row_augmented_counts = np.zeros(len(parents), dtype=np.int64)

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

    def slot_mask(self, step):
        """S2 dose on exactly one quarter of each source; selection independent of transforms."""
        indices = self.parent_positions(step)
        mask = np.zeros(self.spec.batch_size, dtype=bool)
        for source, positions, count in (
            (0, np.flatnonzero(indices < self.n_real), self.real_augmented_per_batch),
            (1, np.flatnonzero(indices >= self.n_real), self.generated_augmented_per_batch),
        ):
            draw = np.random.default_rng([self.seed, 4515, 3, source, int(step)])
            mask[draw.choice(positions, count, replace=False)] = True
        return mask

    def _batches(self):
        for batch in super()._batches():
            indices = self.parent_positions(batch.step)
            self.batch_indices[batch.step - 1] = indices
            if not np.array_equal(batch.transformed, self.slot_mask(batch.step)):
                raise ValueError("native augmentation did not transform the planned S2 slots")
            self.batch_transformed[batch.step - 1] = batch.transformed
            np.add.at(self.row_counts, indices, 1)
            np.add.at(self.row_augmented_counts, indices[batch.transformed], 1)
            if self.tracker is not None and batch.step % self.spec.val_every == 0:
                self.tracker.log_scalars({
                    "budget/real_presentations": int(self.row_counts[:self.n_real].sum()),
                    "budget/generated_presentations": int(self.row_counts[self.n_real:].sum()),
                    "budget/augmented_real_presentations": int(self.row_augmented_counts[:self.n_real].sum()),
                    "budget/augmented_generated_presentations": int(self.row_augmented_counts[self.n_real:].sum()),
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
            "augmented_real_presentations": int(self.row_augmented_counts[:self.n_real].sum()),
            "augmented_generated_presentations": int(self.row_augmented_counts[self.n_real:].sum()),
            "distinct_real_records": int(np.count_nonzero(self.row_counts[:self.n_real])),
            "distinct_generated_records": int(np.count_nonzero(self.row_counts[self.n_real:])),
        }

    def write_audit(self, directory):
        """Persist complete row draws and readable counts after the full run."""
        if not self.consumed or np.any(self.batch_indices < 0):
            raise ValueError("sampling audit requires every planned batch to have been served")
        real_counts = (self.batch_indices < self.n_real).sum(axis=1)
        generated_counts = self.spec.batch_size - real_counts
        augmented_real = ((self.batch_indices < self.n_real) & self.batch_transformed).sum(axis=1)
        augmented_generated = ((self.batch_indices >= self.n_real) & self.batch_transformed).sum(axis=1)
        if not (np.all(augmented_real == self.real_augmented_per_batch)
                and np.all(augmented_generated == self.generated_augmented_per_batch)):
            raise ValueError("observed augmentation does not match exact per-source S2 dosage")
        if self.transform_slots != int(self.batch_transformed.sum()):
            raise ValueError("native stream augmented exposure differs from observed trace")
        if self.accepted != self.transform_slots or self.candidates != self.transform_slots:
            raise ValueError("all planned S2 augmentations must be applied successfully")
        if not (np.all(real_counts == self.real_per_batch)
                and np.all(generated_counts == self.generated_per_batch)):
            raise ValueError("observed batches do not match the planned source allocation")
        recovered = np.bincount(self.batch_indices.reshape(-1), minlength=len(self.parents))
        if not np.array_equal(recovered, self.row_counts):
            raise ValueError("per-record exposure differs from full sampling trace")
        recovered_augmented = np.bincount(self.batch_indices[self.batch_transformed], minlength=len(self.parents))
        if not np.array_equal(recovered_augmented, self.row_augmented_counts):
            raise ValueError("per-record augmented exposure differs from full sampling trace")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        trace = directory / "sampling_trace.npz"
        if trace.exists():
            with np.load(trace, allow_pickle=False) as previous:
                if (not np.array_equal(previous["indices"], self.batch_indices)
                        or not np.array_equal(previous["record_id"], self.parents.record_id)
                        or not np.array_equal(previous["transformed"], self.batch_transformed)):
                    raise ValueError("existing sampling trace differs")
        else:
            with trace.open("xb") as handle:
                np.savez_compressed(handle, indices=self.batch_indices, record_id=self.parents.record_id,
                                    transformed=self.batch_transformed)
        rows_path = directory / "row_exposure.csv"
        batches_path = directory / "batch_composition.csv"
        self._write_csv(rows_path, ["row_index", "record_id", "source", "presentations", "augmented_presentations"], (
            [i, str(record), "real" if i < self.n_real else "generated", int(self.row_counts[i]), int(self.row_augmented_counts[i])]
            for i, record in enumerate(self.parents.record_id)
        ))
        self._write_csv(batches_path, ["step", "real", "generated", "total", "augmented_real", "augmented_generated", "augmented_total"], (
            [step, int(nr), int(ng), self.spec.batch_size, int(ar), int(ag), int(ar + ag)]
            for step, (nr, ng, ar, ag) in enumerate(zip(real_counts, generated_counts, augmented_real, augmented_generated, strict=True), 1)
        ))
        payload = {
            "schema_version": "ptbxl-offline-mixed-s2-sampling@1",
            "seed": self.seed,
            "policy": "independent shuffled exhaustive cycles; exact source quota; shuffled batch positions",
            "rng_key": "numpy.default_rng([seed,4515,source,pass]); source=0 real,1 generated,2 batch shuffle",
            "n_real": self.n_real,
            "n_generated": self.n_generated,
            "real_per_batch": self.real_per_batch,
            "generated_per_batch": self.generated_per_batch,
            "every_batch_has_exact_mix": True,
            "every_batch_has_exact_s2_dose": True,
            "real_augmented_per_batch": self.real_augmented_per_batch,
            "generated_augmented_per_batch": self.generated_augmented_per_batch,
            "augmentation": S2_CONFIG,
            "augmentation_slot_rng": "numpy.default_rng([seed,4515,3,source,step]); 25% within each source",
            "augmentation_waveform_rng": "unchanged native syntide.generate.stream._augment",
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
