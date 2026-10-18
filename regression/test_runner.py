"""Guard the regression gate itself against false passes and false comparisons."""
import copy
import io
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.wcs = {'pixelOrigin': 1, 'crpix': [11, 21], 'crval': [359.99, 20],
                    'cd': [[-1/3600, 0], [0, 1/3600]], 'projection': 'TAN'}
        self.expected = {'minMatches': 10, 'maxRmsArcsec': 1, 'maxControlArcsec': 0.1,
                         'projection': 'TAN', 'controls': [{'x': 10, 'y': 20, 'ra': 359.99, 'dec': 20}]}
        self.response = [{'id': 1, 'result': {'protocolVersion': 1}},
                         {'id': 2, 'result': {'wcs': self.wcs, 'rmsArcsec': 0.01, 'matchedStars': 20}}]

    def test_origin_and_tangent_projection(self):
        self.assertEqual(runner.world(self.wcs, 10, 20), (359.99, 20))
        self.assertAlmostEqual(runner.separation(359.999, 0, 0.001, 0), 7.2, places=7)
        # At the equator, x=1 gives exactly atan(1 arcsecond in radians).
        wcs = dict(self.wcs, crval=[0, 0], crpix=[1, 1])
        self.assertAlmostEqual(runner.world(wcs, 1, 0)[0], 360-math.degrees(math.atan(math.pi/648000)), places=10)

    def test_sip_changes_the_expected_pixel_axis(self):
        wcs = dict(self.wcs, sip={'a': [[2, 0, 0.01]], 'b': []})
        self.assertEqual(runner.world(wcs, 20, 20), runner.world(self.wcs, 21, 20))

    def test_bad_solutions_cannot_pass(self):
        runner.solve_metrics(self.response, self.expected)
        for field, value in [('rmsArcsec', float('nan')), ('rmsArcsec', 2), ('matchedStars', 0), ('matchedStars', float('nan'))]:
            altered = copy.deepcopy(self.response)
            altered[1]['result'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                runner.solve_metrics(altered, self.expected)
        for shift in (0.001, float('nan')):
            altered = copy.deepcopy(self.response)
            altered[1]['result']['wcs']['crval'][0] += shift
            with self.assertRaises(ValueError):
                runner.solve_metrics(altered, self.expected)

    def test_rejection_requires_the_specific_solve_error(self):
        for response in ({'result': {}}, {'error': {'code': -32011}}, {'error': {'code': -32602}}):
            with self.assertRaises(ValueError):
                runner.solve_metrics([self.response[0], dict(response, id=2)], {'errorCode': -32010})
        runner.solve_metrics([self.response[0], {'id': 2, 'error': {'code': -32010}}], {'errorCode': -32010})

    def test_response_identity_and_initialization_are_checked(self):
        for responses in (self.response[:1], self.response[::-1], [dict(self.response[0], error={}), self.response[1]]):
            with self.assertRaises(ValueError):
                runner.solve_metrics(responses, self.expected)

    def test_bad_stacks_cannot_pass(self):
        admitted = {'accepted_frames': 12, 'rejected_frames': 0}
        quality = {'pixelRmse': 0.001, 'fluxRelativeError': 0.001, 'defectResidual': 0.001}
        expected = {'acceptedFrames': 12, 'maxPixelRmse': 0.002, 'maxFluxRelativeError': 0.01,
                    'maxDefectResidual': 0.003}
        self.assertEqual(runner.stack_metrics(admitted, quality, expected)['acceptedFrames'], 12)
        for frames in ({'accepted_frames': 11, 'rejected_frames': 0}, {'accepted_frames': 12, 'rejected_frames': 1}):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                runner.stack_metrics(frames, quality, expected)
        # A defect left in the stack, an unmeasured one, or a nonfinite one fails.
        for residual in (0.004, float('nan'), None):
            altered = dict(quality, defectResidual=residual)
            if residual is None:
                del altered['defectResidual']
            with self.subTest(residual=residual), self.assertRaises(ValueError):
                runner.stack_metrics(admitted, altered, expected)
        for metric in ('pixelRmse', 'fluxRelativeError'):
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                runner.stack_metrics(admitted, dict(quality, **{metric: 0.5}), expected)

    def test_corrupt_cached_asset_is_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'image.fits'
            path.write_bytes(b'good')
            entry = {'path': 'testdata/image.fits', 'sha256': runner.digest(path), 'bytes': 4}
            self.assertEqual(runner.asset(root, entry, False), path)
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                runner.asset(root, entry, False)
            self.assertEqual(path.read_bytes(), b'bad')

    def test_bad_download_never_replaces_the_cached_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'image.fits'
            path.write_bytes(b'old file')
            entry = {'path': 'testdata/image.fits', 'sha256': '0' * 64, 'bytes': 4}
            for body in (b'bad!', b'oversized'):
                with patch.object(runner.urllib.request, 'urlopen', return_value=io.BytesIO(body)):
                    with self.assertRaises(ValueError):
                        runner.asset(root, entry, True)
                self.assertEqual(path.read_bytes(), b'old file')
                self.assertEqual(list(root.iterdir()), [path])

    def test_comparison_requires_matching_inputs_and_checks_memory(self):
        baseline = {'corpusHash': 'abc', 'environment': {'threads': 2}, 'cases': [
            {'id': 'solve', 'median': {'wallSeconds': 1, 'peakRssKiB': 100}}]}
        runner.compare(baseline, baseline, 1.25, 1.15)
        for mutation in ('inputs', 'threads', 'time', 'memory', 'missing', 'failed'):
            report = copy.deepcopy(baseline)
            if mutation == 'inputs': report['corpusHash'] = 'other'
            if mutation == 'threads': report['environment']['threads'] = 3
            if mutation == 'time': report['cases'][0]['median']['wallSeconds'] = 1.26
            if mutation == 'memory': report['cases'][0]['median']['peakRssKiB'] = 116
            if mutation == 'missing': report['cases'][0]['median']['peakRssKiB'] = None
            if mutation == 'failed': report['cases'][0]['error'] = 'failed'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.compare(report, baseline, 1.25, 1.15)

    def test_process_failures_and_timeouts_fail_the_case(self):
        import os
        for script, timeout, error in [('raise SystemExit(7)', 10, ValueError),
                                        ('import time; time.sleep(30)', 0.1, subprocess.TimeoutExpired)]:
            with self.assertRaises(error):
                runner.measured([sys.executable, '-c', script], cwd=Path.cwd(), env=os.environ,
                                payload=None, timeout=timeout)


if __name__ == '__main__':
    unittest.main()
