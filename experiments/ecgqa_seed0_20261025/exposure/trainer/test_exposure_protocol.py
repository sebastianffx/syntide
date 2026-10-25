"""CPU-only scientific-contract tests; never construct or train OpenTSLM."""
import copy
import hashlib
import json
import unittest

import numpy as np
import torch

import exposure_protocol as p


def rows(n, prefix="original"):
    return [{"task_id": f"{prefix}-{i}"} for i in range(n)]


def archived_indices(n, microbatches, batch_size=8):
    """Literal sampling algorithm from archived opentslm_loop.py."""
    generator = np.random.default_rng(0)
    order = generator.permutation(n)
    offset = 0
    for _ in range(microbatches):
        indices = []
        while len(indices) < batch_size:
            count = min(batch_size - len(indices), len(order) - offset)
            indices.extend(order[offset:offset + count].tolist())
            offset += count
            if offset == len(order):
                order, offset = generator.permutation(len(order)), 0
        yield indices


class SamplingTests(unittest.TestCase):
    def test_all_192000_original_draws_match_archived_order_despite_generated_draws(self):
        original = rows(p.ORIGINAL_ROWS)
        real = p.PermutationStream(original)
        generated = p.PermutationStream(rows(p.GENERATED_ROWS, "generated"))
        reference = archived_indices(p.ORIGINAL_ROWS, p.PRIMARY_STEP * 2)
        digest = hashlib.sha256()
        for step in range(p.PRIMARY_STEP):
            for _ in range(2):
                expected = [original[i] for i in next(reference)]
                actual = real.take(8)
                self.assertEqual(actual, expected)
                for row in expected:
                    digest.update((json.dumps(row["task_id"]) + "\n").encode())
            generated.take(2)
        self.assertEqual(real.draws, 192000)
        self.assertEqual(generated.draws, 24000)
        self.assertEqual(real.audit()["completed_epochs"], 2)
        self.assertEqual(real.audit()["epoch_offset"], 26342)
        self.assertEqual(generated.audit()["completed_epochs"], 2)
        self.assertEqual(generated.audit()["epoch_offset"], 656)
        self.assertEqual(real.audit()["draw_task_ids_sha256"], digest.hexdigest())

    def test_epoch_boundary_is_not_dropped_or_restarted(self):
        source = rows(5)
        stream = p.PermutationStream(source)
        reference = list(archived_indices(5, 3, 8))
        for expected in reference:
            self.assertEqual(stream.take(8), [source[i] for i in expected])
        self.assertEqual(stream.draws, 24)
        self.assertEqual(stream.completed_epochs, 4)
        self.assertEqual(stream.offset, 4)

    def test_extra_synthetic_sampling_does_not_change_original_prefix(self):
        real_a, real_b = p.PermutationStream(rows(31)), p.PermutationStream(rows(31))
        gen = p.PermutationStream(rows(17, "generated"))
        for count in [2, 100, 1, 8, 23]:
            gen.take(count)
            self.assertEqual(real_a.take(8), real_b.take(8))


class LossAndRngTests(unittest.TestCase):
    def test_row_weights_and_gradients_are_8_8_2_over_18(self):
        parameters = torch.tensor([3., 5., 7.], requires_grad=True)
        # Different token counts establish that each existing microbatch is
        # normalized first, not pooled into a different global-token objective.
        masks = [torch.ones(11), torch.ones(23), torch.ones(5)]
        loss = sum(p.weighted_microbatch_loss(parameters[i] * mask, mask, n)
                   for i, (mask, n) in enumerate(zip(masks, [8, 8, 2])))
        loss.backward()
        torch.testing.assert_close(parameters.grad, torch.tensor([8/18, 8/18, 2/18]))
        self.assertAlmostEqual(float(loss.detach()), (3*8 + 5*8 + 7*2)/18, places=6)

    def test_no_generated_rows_reduces_exactly_to_baseline_loss(self):
        losses = torch.tensor([2., 4., 3.])
        mask = torch.ones(3)
        self.assertEqual(p.weighted_microbatch_loss(losses, mask, 8, 16),
                         losses.sum() / mask.sum() / 2)

    def test_generated_forward_restores_original_cpu_rng_on_success_and_error(self):
        for should_raise in [False, True]:
            torch.manual_seed(0)
            first = torch.rand(11)
            expected_next = torch.rand(37)
            torch.manual_seed(0)
            torch.testing.assert_close(torch.rand(11), first)
            try:
                with p.generated_forward_rng(torch, 123, []):
                    torch.rand(10003)
                    if should_raise:
                        raise RuntimeError("synthetic failure")
            except RuntimeError:
                pass
            torch.testing.assert_close(torch.rand(37), expected_next)

    def test_generated_forward_is_reproducible_but_changes_with_update(self):
        draws = []
        for step in [2, 2, 3]:
            with p.generated_forward_rng(torch, step, []):
                draws.append(torch.rand(20))
        torch.testing.assert_close(draws[0], draws[1])
        self.assertFalse(torch.equal(draws[0], draws[2]))


class SourceAndCheckpointTests(unittest.TestCase):
    def test_initialization_and_package_mismatch_rejected_before_training(self):
        baseline = {
            "trainer_source_sha256": p.BASELINE_TRAINER_SHA256,
            "max_steps": 15531, "init_sha256": "frozen-initial-parameters",
            "packages": {"numpy": "2.2.6", "torch": "pinned"},
            "data_files": {"train.parquet": p.BASELINE_TRAIN_SHA256, "selection.parquet": "fixed-selection"},
        }
        config = copy.deepcopy(baseline)
        config["max_steps"] = 12000
        p.validate_baseline(config, baseline)
        config["init_sha256"] = "different-initial-parameters"
        with self.assertRaisesRegex(ValueError, "init_sha256"):
            p.validate_baseline(config, baseline)
        config = copy.deepcopy(baseline)
        config["packages"]["numpy"] = "another-version"
        with self.assertRaisesRegex(ValueError, "packages"):
            p.validate_baseline(config, baseline)

    def test_sources_bind_order_and_reject_overlap_or_replacement(self):
        original, generated = rows(p.ORIGINAL_ROWS), rows(p.GENERATED_ROWS, "generated")
        manifest = {
            "schema_version": 1, "original_train_parquet_sha256": p.BASELINE_TRAIN_SHA256,
            "original": {"start": 0, "count": len(original), "task_ids_sha256": p.task_ids_sha256(original)},
            "generated": {"start": len(original), "count": len(generated), "task_ids_sha256": p.task_ids_sha256(generated)},
        }
        combined = original + generated
        self.assertEqual(p.validate_sources(combined, manifest)["original"], original)
        combined[0], combined[1] = combined[1], combined[0]
        with self.assertRaisesRegex(ValueError, "order/task IDs"):
            p.validate_sources(combined, manifest)
        combined[0], combined[1] = combined[1], combined[0]
        combined[-1] = combined[0]
        with self.assertRaisesRegex(ValueError, "unique"):
            p.validate_sources(combined, manifest)

    def test_lr_matches_full_baseline_horizon_not_new_stop_step(self):
        self.assertEqual(p.WARMUP_STEPS, 1554)
        for step in [1, 1000, 1554, 1555, 8000, 12000]:
            expected = step/1554 if step <= 1554 else (15531-step)/(15531-1554)
            self.assertEqual(p.lr_scale(step), expected)
        self.assertGreater(p.lr_scale(12000), 0.25)
        self.assertAlmostEqual(0.0002*p.lr_scale(12000), 0.00005052586391929599)

    def fixture(self):
        config = {"checkpoint_policy": p.CHECKPOINT_POLICY, "max_steps": 12000,
                  "learning_rate_schedule_steps": 15531, "warmup_steps": 1554}
        completed = {"max_steps": 12000, "primary_step": 12000,
                     "source_exposure": {"original": {"draws": 192000}, "generated": {"draws": 24000}}}
        history = [{"step": step, "accuracy": 0.9 if step == 1000 else 0.3}
                   for step in range(1000, 12001, 1000)]
        payload = {"step": 12000, "selection_accuracy": 0.3}
        return config, completed, payload, history

    def test_primary_is_final_exposure_even_when_earliest_selection_is_best(self):
        p.validate_primary_checkpoint(*self.fixture())

    def test_early_or_underexposed_checkpoint_rejected(self):
        config, completed, payload, history = self.fixture()
        payload.update(step=1000, selection_accuracy=0.9)
        with self.assertRaisesRegex(ValueError, "Early selection"):
            p.validate_primary_checkpoint(config, completed, payload, history)
        config, completed, payload, history = self.fixture()
        completed["source_exposure"]["original"]["draws"] -= 16
        with self.assertRaisesRegex(ValueError, "actual original exposure"):
            p.validate_primary_checkpoint(config, completed, payload, history)

    def test_restarted_lr_or_incomplete_diagnostic_history_rejected(self):
        config, completed, payload, history = self.fixture()
        config["learning_rate_schedule_steps"] = 12000
        with self.assertRaisesRegex(ValueError, "LR chronology"):
            p.validate_primary_checkpoint(config, completed, payload, history)
        config, completed, payload, history = self.fixture()
        with self.assertRaisesRegex(ValueError, "history"):
            p.validate_primary_checkpoint(config, completed, payload, history[1:])


if __name__ == "__main__":
    unittest.main()
