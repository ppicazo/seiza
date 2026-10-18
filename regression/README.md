# Scientific regression corpus

Run the corpus before changing detection, solving, calibration, registration, or
integration. Scientific limits are gates; runtime and peak memory are measurements
unless a same-machine baseline is explicitly supplied. No production algorithm,
public API, binding, or default changes here.

## Run it

Requires Rust and Python 3.10 or newer; the runner uses only Python's standard
library. Build once, outside the measured work:

```sh
cargo build --release --locked -p seiza-cli --bin seiza --example regression_fixtures
python3 -m unittest discover -s regression -p 'test_*.py' -v
python3 regression/run.py --output target/regression/offline.json
```

The default is one discarded warmup and three measured repetitions per case,
with two Rayon threads. Every repetition must pass the quality gates. The JSON
report retains each sample, quality measurements, median wall time and peak RSS,
input hashes, executable hashes, and environment information. Failed cases and
process timeouts produce a nonzero exit status and a report containing failures.

On Windows, use `python` in place of `python3`; executable defaults include `.exe`.
Explicit `--binary` and `--fixtures` paths support alternate build directories.

## Offline cases and truth

`seiza-cli/examples/regression_fixtures.rs` creates the versioned synthetic corpus
in a temporary directory, using a fixed integer PRNG. All image reads and writes
go through `seiza-image-io`. The helper's analytic sky coordinates do not use the
solver's WCS or SIP transforms. The runner independently evaluates the returned
FITS TAN/SIP transform, including one-based CRPIX, against 25 control points
spread across the frame.

| Case | Gate |
| --- | --- |
| Hinted linear field | At least 40 matches, RMS and maximum control-point error <= 2 arcsec |
| Mirrored, rotated field across RA=0 | Same positional and match limits |
| Near-polar field | Same positional and match limits |
| Cubic radial distortion | Same limits, and a TAN-SIP result is required |
| Blind field | Same limits, no position hint |
| Unrelated catalog in the same sky footprint | Must return solve failure, not a match or timeout |
| Blank image and deterministic noise | Must return solve failure, not a match or timeout |
| Shifted calibrated stack | Pixel RMSE <= 0.002 and background-subtracted flux error <= 1% |
| Reintegrated calibrated stack | Same pixel and flux limits |
| Dithered lights without a dark | Same pixel and flux limits, and within one pixel of every place a sensor defect landed, the stack within 0.003 of truth |

The stack inputs contain a known additive dark and a spatially varying flat.
Three lights have integer shifts and gains 0.9, 1.0 and 1.1 (mean 1.0); all three
must be admitted with no rejections. Normalization and rejection are disabled and
weights are equal so neither can conceal a calibration or flux error. The pixel
comparison excludes 12 boundary pixels to avoid intentionally uncovered samples;
it checks dimensions, channel count and finite pixels as well. The thresholds
are scientific acceptance limits, not exact floating-point output snapshots.

The dithered case stacks twelve lights shifted by up to six pixels, with noise
of 0.001 on a 0.01 sky and no dark, through the default `--defects auto`. The
same 20 hot and dead pixels sit on the sensor in every light: saturating (+1.0),
bright (+0.3, +0.1), moderate (+0.05) and dead. Each lands on the reference grid
at its sensor pixel less its light's shift, clear of every star and the border;
`defect-probes.json` lists those 240 places, and the gate takes the largest
difference from the truth within one pixel of any. Rejection stays disabled, so
only repair can keep a defect out: with `--defects off` the residual is about
0.08, and the RMSE and flux gates fail too. All twelve lights must be admitted.

The synthetic corpus is small and does not replace the real camera workload.
It covers mono calibration/integration; it does not yet provide independent
photometric ground truth for real RGB/CFA stacks or distortion truth for real
lenses. Add such fixtures with measured reference data rather than blessing
current output as truth.

## Pinned camera images

```sh
python3 regression/run.py --hosted --download --output target/regression/hosted.json
# Later, offline: omit --download; every cached file is still rehashed.
python3 regression/run.py --hosted --output target/regression/hosted.json
```

`hosted.json` pins SHA-256 and byte counts for the existing Tycho-2 lite catalog
and five existing camera fixtures from `downloads.seiza.fyi` (about 185 MiB total).
These include JPEG, a 61 MP narrowband mono FITS, and a raw OSC/CFA FITS. The four
images already supported by the blind integration suite run both hinted and
blind; the OSC runs hinted only. Position and scale expectations come from the
existing hosted integration tests, not from newly recorded solver output.

Every successful real-image solve must meet the existing 72 arcsec center and
0.5% scale limits, at least 10 matches, and RMS <= 5 arcsec. Center/scale checks
are deliberately coarser than the synthetic full-field astrometry checks: the
hosted references do not contain independent control-star positions. The runner
rejects corrupt cached assets and only publishes a downloaded file after size
and checksum validation. Changing an asset requires a reviewed manifest update;
there is no automatic checksum refresh.

## Performance baselines

On a quiet, controlled Linux machine with GNU `/usr/bin/time`:

```sh
python3 regression/run.py --hosted --output /tmp/seiza-main.json
# Build the candidate in another checkout, then reuse this harness and fixtures:
python3 regression/run.py --hosted --binary /path/to/candidate/seiza \
  --baseline /tmp/seiza-main.json --output /tmp/seiza-candidate.json
```

Comparison defaults reject >25% growth in median wall time or >15% growth in
median peak RSS. Override with `--max-slowdown` and `--max-memory-growth` only
when the experiment calls for different limits. Inputs, harness source, fixture
source, host/platform, Python version, thread count, warmup count and repetition
count must match. Failed reports cannot serve as baselines. Keep the same fixture
helper when comparing binaries so a candidate cannot redefine its own truth.

Measurements are **one-shot worker/stack process costs**. They include process
startup, catalog loading, image decoding, detection, solving and lazy blind-index
construction, or the full stacking command. They exclude compilation, downloads,
fixture generation and stack-output verification. GNU time supplies per-process
peak RSS in KiB and user/system CPU seconds. On macOS and Windows wall time and
quality are recorded, but RSS is explicitly null; memory comparison refuses to
pass without valid measurements. Do not compare absolute timings across hosts,
build profiles, feature configurations or uncontrolled background loads. For
optimization claims, alternate base/candidate runs and inspect distributions,
not just one median. In particular, these cold-worker costs are not a persistent
worker's steady-state solve latency.

## CI

The existing three-platform test job runs the offline corpus with release builds
on every PR and main push. The integration job runs the pinned camera corpus on
PRs, release tags and manual dispatches. Both upload JSON reports, including on
failure. Performance is recorded, not gated on shared runners; scientific limits
always gate. The existing hosted protocol/download integration tests remain in
place. Unit tests also prove that wrong coordinates, nonfinite RMS, wrong error
codes, corrupt cache entries, failed processes, timeouts and invalid comparisons
cannot produce a passing gate.
