import hashlib
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import kaggle_eval_v3 as evaluation


class EvaluationWorkerTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'input/source'
        source.mkdir(parents=True)
        job = {'job_id': 'temporal_qp_seed2026', 'variant': 'temporal_qp',
               'seed': 2026, 'source_revision': 'a' * 40}
        job_file = source / (job['job_id'] + '.json')
        job_file.write_text(json.dumps(job), encoding='utf-8')
        checkpoint = source / (job['job_id'] + '.last.pt')
        state = {'schema_version': 3, 'job': job, 'status': 'converged',
                 'model_state': {'w': torch.ones(2)}, 'best_state': {'w': torch.ones(2)},
                 'best_epoch': 2, 'epoch': 13, 'history': [{'epoch': 12}]}
        torch.save(state, checkpoint)
        config = {**job, 'source_kernel': 'owner/source/1',
                  'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'job_sha256': hashlib.sha256(job_file.read_bytes()).hexdigest()}
        return config, source, job_file, checkpoint, state

    def test_valid_input_preserves_original_job_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, source, job_file, checkpoint, state = self.fixture(root)
            output = root / 'output'
            result = evaluation.prepare_eval_inputs(config, root / 'input', output)
            self.assertEqual(Path(result['checkpoint']), checkpoint)
            self.assertEqual((output / job_file.name).read_bytes(), job_file.read_bytes())
            self.assertEqual(result['best_epoch'], 2)
            self.assertFalse((output / checkpoint.name).exists())

    def test_tampered_and_ambiguous_inputs_are_rejected(self):
        for mode in ('hash', 'ambiguous'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config, source, job_file, checkpoint, state = self.fixture(root)
                if mode == 'hash':
                    checkpoint.write_bytes(checkpoint.read_bytes() + b'x')
                else:
                    other = root / 'input/other'
                    other.mkdir()
                    (other / checkpoint.name).write_bytes(checkpoint.read_bytes())
                with self.assertRaises(ValueError):
                    evaluation.prepare_eval_inputs(config, root / 'input', root / 'output')

    def test_incomplete_wrong_job_and_nonfinite_weights_are_rejected(self):
        for mode in ('incomplete', 'wrong_job', 'nonfinite'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config, source, job_file, checkpoint, state = self.fixture(root)
                if mode == 'incomplete':
                    state['status'] = 'time_budget_exhausted'
                elif mode == 'wrong_job':
                    state['job'] = dict(state['job'], seed=2027)
                else:
                    state['best_state']['w'][0] = float('nan')
                torch.save(state, checkpoint)
                config['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    evaluation.prepare_eval_inputs(config, root / 'input', root / 'output')

    def test_worker_uses_pinned_source_checkpoint_and_both_detectors(self):
        with tempfile.TemporaryDirectory() as directory:
            config, *_ = self.fixture(Path(directory))
            script = evaluation.build_eval_worker(config)
            compile(script, 'worker.py', 'exec')
            for value in ('--checkpoint', '--eval-yolo', 'requirements-eval.txt',
                          '3.11.14', config['source_revision'], 'V3_EVAL_INPUT_VERIFIED'):
                self.assertIn(value, script)
            self.assertNotIn('--resume', script)

    def test_inner_bootstraps_in_a_fresh_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            config, *_ = self.fixture(Path(directory))
            tree = ast.parse(evaluation.build_eval_worker(config))
            inner = next(node.value.value for node in tree.body
                         if isinstance(node, ast.Assign) and node.targets[0].id == 'INNER')
            body = []
            for node in ast.parse(inner).body:
                body.append(node)
                if isinstance(node, ast.Assign) and node.targets[0].id == 'config':
                    break
            namespace = {}
            with patch.object(sys, 'argv', ['-c', json.dumps(config)]):
                exec(compile(ast.Module(body=body, type_ignores=[]), 'inner.py', 'exec'), namespace)
            self.assertEqual(namespace['config'], config)


if __name__ == '__main__':
    unittest.main()
