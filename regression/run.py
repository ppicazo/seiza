#!/usr/bin/env python3
"""Run the versioned Seiza scientific corpus; Python 3.10+, standard library only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def asset(root, entry, download):
    """Never accept an unverified cache hit or publish a partial download."""
    path = root / Path(entry['path']).name
    if path.exists() and digest(path) == entry['sha256']:
        return path
    if not download:
        raise ValueError(f'missing or corrupt asset {path}; use --download')
    root.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urllib.request.urlopen('https://downloads.seiza.fyi/' + entry['path'], timeout=60) as source:
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as output:
                temporary = Path(output.name)
                size = 0
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > entry['bytes']:
                        raise ValueError(f'oversized asset {entry["path"]}')
                    output.write(block)
        if size != entry['bytes'] or digest(temporary) != entry['sha256']:
            raise ValueError(f'checksum/size mismatch: {entry["path"]}')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def separation(ra, dec, other_ra, other_dec):
    if not all(math.isfinite(v) for v in (ra, dec, other_ra, other_dec)):
        raise ValueError("nonfinite sky coordinate")
    d1, d2 = map(math.radians, (dec, other_dec))
    dra = math.radians(other_ra - ra)
    hav = math.sin((d2 - d1) / 2)**2 + math.cos(d1) * math.cos(d2) * math.sin(dra / 2)**2
    return math.degrees(2 * math.asin(math.sqrt(max(0, min(1, hav))))) * 3600


def world(wcs, x, y):
    """Independent FITS TAN/SIP evaluation, with explicit one-based CRPIX."""
    if wcs['pixelOrigin'] != 1:
        raise ValueError('unexpected WCS pixel origin')
    u, v = x + 1 - wcs['crpix'][0], y + 1 - wcs['crpix'][1]
    sip = wcs.get('sip')
    if sip:
        u, v = (u + sum(c * u**p * v**q for p, q, c in sip['a']),
                v + sum(c * u**p * v**q for p, q, c in sip['b']))
    cd = wcs['cd']
    xi, eta = math.radians(cd[0][0] * u + cd[0][1] * v), math.radians(cd[1][0] * u + cd[1][1] * v)
    ra, dec = map(math.radians, wcs['crval'])
    denominator = math.cos(dec) - eta * math.sin(dec)
    return (math.degrees(ra + math.atan2(xi, denominator)) % 360,
            math.degrees(math.atan2(math.sin(dec) + eta * math.cos(dec), math.hypot(denominator, xi))))


def bounded(value, limit, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > limit:
        raise ValueError(f'{name}: {value!r} outside [0, {limit}]')


def solve_metrics(responses, expected):
    if len(responses) != 2 or [r.get('id') for r in responses] != [1, 2]:
        raise ValueError('missing, duplicate or unexpected worker responses')
    if 'error' in responses[0] or responses[0].get('result', {}).get('protocolVersion') != 1:
        raise ValueError('worker initialization failed')
    response = responses[1]
    if 'errorCode' in expected:
        if response.get('error', {}).get('code') != expected['errorCode']:
            raise ValueError(f'expected solve rejection, got {response}')
        return {'rejected': True}
    if 'error' in response:
        raise ValueError(f'solve failed: {response["error"]}')
    result = response['result']
    rms = result['rmsArcsec']
    bounded(rms, expected['maxRmsArcsec'], 'RMS arcsec')
    matches = result['matchedStars']
    if not isinstance(matches, int) or isinstance(matches, bool) or matches < expected['minMatches']:
        raise ValueError(f'only {matches} matched stars')
    metrics = {'rmsArcsec': rms, 'matchedStars': matches}
    if 'controls' in expected:
        errors = [separation(*world(result['wcs'], c['x'], c['y']), c['ra'], c['dec']) for c in expected['controls']]
        metrics['maxControlArcsec'] = max(errors)
        bounded(metrics['maxControlArcsec'], expected['maxControlArcsec'], 'control-point arcsec')
        if result['wcs']['projection'] != expected['projection']:
            raise ValueError('projection changed')
    if 'center' in expected:
        error = separation(result['center']['raDeg'], result['center']['decDeg'], *expected['center'])
        metrics['centerErrorArcsec'] = error
        bounded(error, expected['maxCenterArcsec'], 'center arcsec')
        scale_error = abs(result['pixelScaleArcsecPerPixel'] / expected['scale'] - 1)
        metrics['scaleRelativeError'] = scale_error
        bounded(scale_error, expected['maxScaleRelativeError'], 'scale relative error')
    return metrics


def stack_metrics(admitted, quality, expected):
    frames = expected['acceptedFrames']
    if admitted['accepted_frames'] != frames or admitted['rejected_frames'] != 0:
        raise ValueError(f'stack must integrate all {frames} frames without rejection')
    metrics = dict(quality, acceptedFrames=admitted['accepted_frames'])
    for metric in ('pixelRmse', 'fluxRelativeError'):
        bounded(metrics[metric], expected['max' + metric[0].upper() + metric[1:]], metric)
    if 'maxDefectResidual' in expected:
        bounded(metrics.get('defectResidual'), expected['maxDefectResidual'], 'defectResidual')
    return metrics


def measured(command, *, cwd, env, payload, timeout):
    # GNU time reports this child's high-water RSS, not a process-global maximum
    # accumulated over earlier cases. Compilation and fixture creation are excluded.
    with tempfile.TemporaryDirectory(prefix='seiza-measure-') as tmp:
        usage = Path(tmp) / 'usage.json'
        timed = sys.platform.startswith('linux') and Path('/usr/bin/time').exists()
        if timed:
            command = ['/usr/bin/time', '-f', '{"peakRssKiB":%M,"userSeconds":%U,"systemSeconds":%S}', '-o', str(usage), *command]
        start = time.perf_counter()
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, cwd=cwd, env=env,
                                   start_new_session=os.name == 'posix')
        try:
            stdout, stderr = process.communicate(payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.communicate()
            raise
        elapsed = time.perf_counter() - start
        if process.returncode:
            raise ValueError(f'process exited {process.returncode}: {stderr[-4000:]}')
        measurements = json.loads(usage.read_text()) if timed else {'peakRssKiB': None}
        measurements['wallSeconds'] = elapsed
        return stdout, measurements


def run_case(case, directory, binary, fixtures, env, timeout):
    if case['kind'] == 'solve':
        params = dict(case['params'], imagePath=str(directory / case['image']), timeLimitMs=int(timeout * 900))
        requests = [{'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': 1}},
                    {'jsonrpc': '2.0', 'id': 2, 'method': 'solve', 'params': params}]
        text, usage = measured([str(binary), 'worker', '--data', str(directory / case['catalog'])],
                              cwd=directory, env=env, payload=''.join(json.dumps(r) + '\n' for r in requests), timeout=timeout)
        metrics = solve_metrics([json.loads(line) for line in text.splitlines()], case['expected'])
    else:
        output = directory / (case['id'] + '.fits')
        admission = output.with_suffix('.json')
        command = [str(binary), *case['args'], '--output', str(output), '--report', str(admission)]
        if case.get('reintegrate'):
            command.append('--reintegrate')
        _, usage = measured(command, cwd=directory, env=env, payload=None, timeout=timeout)
        admitted = json.loads(admission.read_text())
        check = [str(fixtures), 'check-stack', str(directory), str(output)]
        if 'probes' in case:
            check.append(str(directory / case['probes']))
        verification = subprocess.run(check, text=True, capture_output=True, check=True, timeout=timeout)
        metrics = stack_metrics(admitted, json.loads(verification.stdout), case['expected'])
    return dict(usage, quality=metrics)


def compare(report, baseline, slowdown, growth):
    if baseline.get('failures'):
        raise ValueError('cannot compare against a failed report')
    if report['corpusHash'] != baseline['corpusHash'] or report['environment'] != baseline['environment']:
        raise ValueError('baseline corpus/environment mismatch; rebaseline explicitly on this machine')
    old_cases = {c['id']: c for c in baseline['cases']}
    if len(old_cases) != len(baseline['cases']) or set(old_cases) != {c['id'] for c in report['cases']}:
        raise ValueError('baseline case set mismatch')
    for case in report['cases']:
        previous = old_cases[case['id']]
        if previous.get('error') or case.get('error'):
            raise ValueError('cannot compare a failed case')
        for metric, ratio in [('wallSeconds', slowdown), ('peakRssKiB', growth)]:
            new, old = case['median'][metric], previous['median'][metric]
            if new is None or old is None or not math.isfinite(old) or old <= 0:
                raise ValueError(f'baseline lacks valid {metric} for {case["id"]}')
            bounded(new, old * ratio, f'{case["id"]} {metric}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / ('target/release/seiza.exe' if os.name == 'nt' else 'target/release/seiza'))
    parser.add_argument('--fixtures', type=Path, default=ROOT / ('target/release/examples/regression_fixtures.exe' if os.name == 'nt' else 'target/release/examples/regression_fixtures'))
    parser.add_argument('--output', type=Path, default=ROOT / 'target/regression/report.json')
    parser.add_argument('--hosted', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--cache', type=Path, default=ROOT / 'target/regression/hosted')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--max-slowdown', type=float, default=1.25)
    parser.add_argument('--max-memory-growth', type=float, default=1.15)
    args = parser.parse_args(argv)
    if min(args.repeats, args.threads) < 1 or args.warmups < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('positive repeats/threads/timeout and nonnegative warmups required')
    for ratio in (args.max_slowdown, args.max_memory_growth):
        if not math.isfinite(ratio) or ratio < 1:
            parser.error('performance ratios must be finite and >= 1')
    binary, fixtures = args.binary.resolve(), args.fixtures.resolve()
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads))
    report = {'schemaVersion': 1, 'binarySha256': digest(binary), 'fixtureGeneratorSha256': digest(fixtures),
              'environment': {'platform': platform.platform(), 'machine': platform.machine(),
                              'cpuCount': os.cpu_count(), 'host': platform.node(),
                              'python': platform.python_version(), 'threads': args.threads, 'repeats': args.repeats,
                              'warmups': args.warmups}, 'cases': []}
    failures = []
    try:
        with tempfile.TemporaryDirectory(prefix='seiza-corpus-') as tmp:
            directory = Path(tmp)
            subprocess.run([str(fixtures), 'generate', str(directory)], check=True, env=env, timeout=args.timeout)
            spec = json.loads((directory / 'cases.json').read_text())
            jobs = [(c, directory) for c in spec['cases']]
            inputs = {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()}
            inputs['fixture-source'] = digest(ROOT / 'seiza-cli/examples/regression_fixtures.rs')
            inputs['runner-source'] = digest(Path(__file__))
            if args.hosted:
                hosted = json.loads((ROOT / 'regression/hosted.json').read_text())
                args.cache = args.cache.resolve()
                for entry in hosted['assets']:
                    asset(args.cache, entry, args.download)
                jobs += [(c, args.cache) for c in hosted['cases']]
                inputs['hosted.json'] = digest(ROOT / 'regression/hosted.json')
            report['inputSha256'] = inputs
            report['corpusHash'] = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
            for case, workdir in jobs:
                entry = {'id': case['id']}
                try:
                    for _ in range(args.warmups):
                        run_case(case, workdir, binary, fixtures, env, args.timeout)
                    samples = [run_case(case, workdir, binary, fixtures, env, args.timeout) for _ in range(args.repeats)]
                    entry['samples'] = samples
                    entry['median'] = {key: statistics.median(s[key] for s in samples) if samples[0][key] is not None else None
                                       for key in ('wallSeconds', 'peakRssKiB')}
                    print(f'PASS {case["id"]}: {entry["median"]}', flush=True)
                except (ValueError, KeyError, subprocess.SubprocessError, OSError) as error:
                    entry['error'] = str(error)
                    failures.append(f'{case["id"]}: {error}')
                    print(f'FAIL {failures[-1]}', flush=True)
                report['cases'].append(entry)
            if args.baseline:
                compare(report, json.loads(args.baseline.read_text()), args.max_slowdown, args.max_memory_growth)
    except (ValueError, KeyError, subprocess.SubprocessError, OSError) as error:
        failures.append(str(error))
    report['failures'] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Report: {args.output}')
    for failure in failures:
        print(failure, file=sys.stderr)
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
