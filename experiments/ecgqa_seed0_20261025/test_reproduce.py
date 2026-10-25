"""CPU-only checks for the portable evidence reader and command printer."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('reproduce', ROOT / 'reproduce.py')
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


class ReproductionTests(unittest.TestCase):
    def test_committed_evidence_and_table(self):
        checked = reproduce.verify()
        self.assertEqual(checked['status'], 'passed')
        data = reproduce.results()
        self.assertEqual(len(data['arms']), 7)
        self.assertEqual(reproduce.table(), (ROOT / 'RESULTS.md').read_text())
        baseline = data['arms']['real_qa']['metrics']['test']['all']
        matched = data['arms']['exposure_matched']['metrics']['test']['all']
        self.assertAlmostEqual(100 * (matched['accuracy'] - baseline['accuracy']),
                               0.29432584478969)

    def test_corrupt_file_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.py'
            source.write_text('original\n')
            expected = {'bytes': source.stat().st_size, 'sha256': reproduce.sha(source)}
            (root / 'manifest.json').write_text(json.dumps({
                'schema': 'ecgqa-seven-arm-reproduction-files@1', 'files': {'source.py': expected}}))
            source.write_text('tampered\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                reproduce.verify(root)

    def test_bad_slice_accounting_refused(self):
        data = copy.deepcopy(reproduce.results())
        data['arms']['real_qa']['metrics']['test']['clean']['accuracy'] = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'evidence').mkdir()
            (root / 'evidence/results.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Weighted slice'):
                reproduce.results(root)

    def test_training_commands_preserve_selection_and_exposure(self):
        base = reproduce.command('real_qa', '/tmp/prepared data', '/tmp/new baseline')
        matched = reproduce.command('exposure_matched', '/tmp/prepared data', '/tmp/new matched')
        for result in (base, matched):
            argv = result['argv']
            self.assertEqual(argv[argv.index('--selection-method') + 1], 'option-nll')
            self.assertEqual(argv[argv.index('--max-length') + 1], '1536')
            self.assertIn(str(Path('/tmp/prepared data').resolve()), argv)
        self.assertEqual(base['argv'][base['argv'].index('--max-steps') + 1], '15531')
        self.assertEqual(matched['argv'][matched['argv'].index('--max-steps') + 1], '12000')
        self.assertIn('--baseline-config', matched['argv'])
        self.assertIn('--source-manifest', matched['argv'])
        self.assertIn(str(ROOT / 'exposure/trainer/opentslm_exposure.py'), matched['argv'])

    def test_evaluation_controls_and_batch_sizes(self):
        for arm, batch in [('real_qa', '512'), ('diffusets_full_stack', '256'),
                           ('exposure_matched', '256')]:
            argv = reproduce.command(arm, '/tmp/data', '/tmp/run', phase='test', zero_signal=True)['argv']
            self.assertIn('evaluate', argv)
            self.assertIn('--zero-signal', argv)
            self.assertEqual(argv[argv.index('--batch-size') + 1], batch)
        with self.assertRaisesRegex(ValueError, 'evaluation control'):
            reproduce.command('real_qa', '/tmp/data', '/tmp/run', zero_signal=True)


if __name__ == '__main__':
    unittest.main()
