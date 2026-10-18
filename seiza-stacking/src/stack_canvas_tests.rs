//! Frames on an expanded canvas are resampled, normalized and integrated
//! over their own footprints, which must give what the whole canvas gives.

use super::*;
use crate::{BatchStackOptions, Continue, PipelineOptions, RegistrationModel};
use seiza_fits::{F32ImageData, write_f32_image};

/// A `columns` by `rows` mosaic of `panel_width` by `panel_height` panels
/// overlapping by about 12% on each side, over a crowded field, with
/// `dithers` frames per panel a few pixels apart, in one channel or three.
/// Returns each frame's path, panel by panel.
fn mosaic(
    directory: &Path,
    (columns, rows): (usize, usize),
    (panel_width, panel_height): (usize, usize),
    dithers: usize,
    channels: usize,
) -> Vec<PathBuf> {
    const DITHER: usize = 3;
    let step_x = panel_width * 88 / 100;
    let step_y = panel_height * 88 / 100;
    let sky_width = panel_width + (columns - 1) * step_x + DITHER * dithers;
    let sky_height = panel_height + (rows - 1) * step_y + DITHER * dithers;
    let mut sky = vec![0.01_f32; sky_width * sky_height];
    let mut seed = 0x5eed_u64;
    let mut random = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    for _ in 0..sky_width * sky_height / 160 {
        let x = 4.0 + random() * (sky_width - 8) as f64;
        let y = 4.0 + random() * (sky_height - 8) as f64;
        let peak = 0.02 + 0.0005 / (0.01 + random()).powf(1.5) as f32;
        for dy in -4_isize..=4 {
            for dx in -4_isize..=4 {
                let px = x.floor() as isize + dx;
                let py = y.floor() as isize + dy;
                let r2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
                sky[py as usize * sky_width + px as usize] += peak * (-r2 / 2.9).exp() as f32;
            }
        }
    }
    // A gentle gradient, so local background normalization has one to fit.
    for (index, value) in sky.iter_mut().enumerate() {
        *value += 0.002 * (index % sky_width) as f32 / sky_width as f32;
    }
    let mut frames = Vec::new();
    for panel_y in 0..rows {
        for panel_x in 0..columns {
            for dither in 0..dithers {
                let (ox, oy) = (
                    panel_x * step_x + dither * DITHER,
                    panel_y * step_y + dither * DITHER / 3,
                );
                let path = directory.join(format!("panel-{panel_x}{panel_y}-{dither}.fits"));
                let mut data = Vec::with_capacity(panel_width * panel_height * channels);
                for y in 0..panel_height {
                    let start = (y + oy) * sky_width + ox;
                    for &value in &sky[start..start + panel_width] {
                        for channel in 0..channels {
                            // Each exposure has its own read and sky noise.
                            let normal = (0..4).map(|_| random()).sum::<f64>() - 2.0;
                            let tint = 1.0 + 0.2 * channel as f32;
                            data.push(value * tint + (normal * 0.0005) as f32);
                        }
                    }
                }
                let pixels = match channels {
                    1 => F32ImageData::Mono(&data),
                    _ => F32ImageData::RgbInterleaved(&data),
                };
                write_f32_image(&path, panel_width, panel_height, pixels, &[]).unwrap();
                frames.push(path);
            }
        }
    }
    frames
}

/// The samples' bits, so `NaN`s compare and `-0.0` differs from `0.0`.
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn assert_same_snapshot(expected: &StackSnapshot, actual: &StackSnapshot, what: &str) {
    assert!(
        bits(&expected.image.data) == bits(&actual.image.data),
        "{what}: means differ"
    );
    assert!(
        bits(&expected.variance.data) == bits(&actual.variance.data),
        "{what}: variances differ"
    );
    assert!(
        expected.coverage == actual.coverage,
        "{what}: coverage differs"
    );
    assert!(
        expected.rejected_samples == actual.rejected_samples,
        "{what}: rejected samples differ"
    );
    assert_eq!(
        (expected.accepted_frames, expected.rejected_frames),
        (actual.accepted_frames, actual.rejected_frames),
        "{what}"
    );
}

/// A stack planned through every path but the last of each panel, and kept
/// for reintegration, before any frame is pushed.
fn expanding(paths: &[PathBuf], planned: &[PathBuf], options: &StackOptions) -> LiveStacker {
    let mut stacker = LiveStacker::with_expanding_reference(
        planned,
        CalibrationMasters::default(),
        options.clone(),
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    assert!(paths.len() > planned.len());
    stacker.retain_frames_for_reintegration(None).unwrap();
    stacker
}

/// Push every path, with frames resampled over the whole canvas when
/// `whole`, returning each frame's disposition as it prints.
fn push_all(stacker: &mut LiveStacker, paths: &[PathBuf], whole: bool) -> Vec<String> {
    WHOLE_CANVAS.set(whole);
    let dispositions = paths
        .iter()
        .map(|path| format!("{:?}", stacker.push_fits(path).unwrap()))
        .collect();
    WHOLE_CANVAS.set(false);
    dispositions
}

/// Footprints change nothing a mosaic stack gives: every frame's diagnostics,
/// the stack, and its reintegration from the frames kept for one match the
/// whole canvas's to the bit, mono and colour, globally and locally
/// normalized, weighted or not, similarity or warp, and for frames joining
/// after planning, which stray past the canvas's edge. A pipelined run, which
/// prepares frames on other threads, gives the same.
#[test]
fn footprints_stack_a_mosaic_as_the_whole_canvas_does() {
    for (normalization, weighting, channels, model) in [
        (
            NormalizationMode::Global,
            FrameWeighting::Equal,
            1,
            RegistrationModel::Similarity,
        ),
        (
            NormalizationMode::LocalBackground { tile_size: 64 },
            FrameWeighting::InverseNoiseVariance {
                minimum_weight: 0.05,
                maximum_weight: 20.0,
            },
            3,
            RegistrationModel::Quadratic,
        ),
    ] {
        let what = format!("{normalization:?} {weighting:?} {channels} channel(s) {model:?}");
        let directory = tempfile::tempdir().unwrap();
        let paths = mosaic(directory.path(), (2, 2), (512, 384), 3, channels);
        // Each panel's third frame joins without a plan, three pixels past
        // the canvas the first two planned.
        let planned = paths
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 3 != 2)
            .map(|(_, path)| path.clone())
            .collect::<Vec<_>>();
        let mut options = StackOptions {
            normalization,
            weighting,
            ..StackOptions::default()
        };
        options.registration.model = model;
        options.acceptance.minimum_overlap_fraction = 0.1;

        let mut whole = expanding(&paths, &planned, &options);
        let mut bounded = expanding(&paths, &planned, &options);
        let expected = push_all(&mut whole, &paths, true);
        let dispositions = push_all(&mut bounded, &paths, false);
        assert!(
            expected
                .iter()
                .all(|disposition| disposition.starts_with("Accepted")),
            "{what}: {expected:?}"
        );
        assert_eq!(expected, dispositions, "{what}");
        assert_eq!(
            model == RegistrationModel::Quadratic,
            expected
                .iter()
                .any(|disposition| disposition.contains("warp: Some")),
            "{what}"
        );
        let snapshot = whole.snapshot().unwrap();
        assert_same_snapshot(&snapshot, &bounded.snapshot().unwrap(), &what);

        let replay = |stacker: &LiveStacker| {
            stacker
                .reintegrate(&BatchStackOptions::default(), |_, _, _| {})
                .unwrap()
                .snapshot
        };
        assert_same_snapshot(
            &replay(&whole),
            &replay(&bounded),
            &format!("{what}, reintegrated"),
        );

        let mut pipelined = expanding(&paths, &planned, &options);
        let mut piped = Vec::new();
        let report = pipelined
            .push_fits_pipelined(&paths, &PipelineOptions::default(), |_, disposition| {
                piped.push(format!("{:?}", disposition.unwrap()));
                Continue::Yes
            })
            .unwrap();
        assert_eq!(report.integrated, paths.len(), "{what}, pipelined");
        assert_eq!(expected, piped, "{what}, pipelined");
        assert_same_snapshot(
            &snapshot,
            &pipelined.snapshot().unwrap(),
            &format!("{what}, pipelined"),
        );
    }
}

/// A frame on a canvas of nine panels is resampled over a region little
/// larger than itself, which holds every sample the whole canvas would give
/// it, each to the bit.
#[test]
fn a_frame_is_resampled_over_its_footprint_alone() {
    let directory = tempfile::tempdir().unwrap();
    let (width, height) = (512, 384);
    // A panel's second frame lies within it, and is warped onto it.
    let paths = mosaic(directory.path(), (3, 3), (width, height), 2, 1);
    let mut options = StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    };
    options.registration.model = RegistrationModel::Quadratic;
    options.acceptance.minimum_overlap_fraction = 0.1;
    let stacker = LiveStacker::with_expanding_reference(
        &paths,
        CalibrationMasters::default(),
        options,
        usize::MAX,
        |_, _, _, _| Continue::Yes,
    )
    .unwrap();
    let canvas = &stacker.reference;
    let mut warped = 0;
    for path in &paths {
        let (frame, _) = FitsFrame::open(path)
            .unwrap()
            .into_prepared_with_layout(stacker.options.demosaic)
            .unwrap();
        let known =
            crate::expand::registration_for(&stacker.planned_registrations, frame.source.as_ref())
                .unwrap();
        let registration = known.unwrap().unwrap().clone();
        let prepared = prepare_frame(
            canvas,
            FrameRegistration {
                registrar: &stacker.registrar,
                canvas: stacker.canvas_registrar.as_ref(),
                known,
            },
            &stacker.options,
            &stacker.reference_noise,
            frame.image.clone(),
            None,
            None,
        )
        .unwrap();
        let PreparedFrame::Ready(ready) = prepared else {
            panic!("{}: not prepared", path.display());
        };
        let region = ready.region;
        assert_eq!(
            (ready.registered.width, ready.registered.height),
            (region.width, region.height)
        );
        // The frame, give or take its edge pixel, the margin and a
        // fraction of a degree of rotation, on a canvas nearly nine times
        // its size.
        assert!(
            region.width <= width + 8 && region.height <= height + 8,
            "{}: {region:?}",
            path.display()
        );
        assert!(region.width * region.height * 6 < canvas.width * canvas.height);
        warped += usize::from(registration.warp.is_some());
        let geometry = crate::registration::FrameGeometry::of(
            registration.transform,
            registration.warp.as_ref(),
        );
        let everywhere = crate::registration::resample_region_geometry(
            &frame.image,
            canvas.width,
            canvas.height,
            ReferenceRegion {
                x: 0,
                y: 0,
                width: canvas.width,
                height: canvas.height,
            },
            geometry,
            stacker.options.interpolation.into(),
        )
        .unwrap();
        let placed = Placed::on(&ready.registered, region, canvas);
        for (index, &sample) in everywhere.data.iter().enumerate() {
            assert_eq!(
                sample.to_bits(),
                placed.sample(index).to_bits(),
                "{}: sample {index}",
                path.display()
            );
        }
    }
    assert!(warped > 0, "no frame was warped");
}

/// Per-frame time to stack the same frames onto canvases of one, four and
/// nine panels, prepared over the whole canvas, as before footprints, and
/// over each frame's footprint, which keeps it about the same. Run with
/// `--release --ignored --nocapture`; `SEIZA_CANVAS_PANEL=WxH` sets the
/// panel size and `SEIZA_CANVAS_CHANNELS` the channels.
#[test]
#[ignore = "a timing run on large mosaic frames"]
fn canvas_integration_timing_by_panel_count() {
    let (width, height) = std::env::var("SEIZA_CANVAS_PANEL")
        .ok()
        .and_then(|value| {
            let (width, height) = value.split_once('x')?;
            Some((width.parse().ok()?, height.parse().ok()?))
        })
        .unwrap_or((2048, 1536));
    let channels = std::env::var("SEIZA_CANVAS_CHANNELS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    for normalization in [
        NormalizationMode::Global,
        NormalizationMode::LocalBackground { tile_size: 256 },
    ] {
        for grid in [(1, 1), (2, 2), (3, 3)] {
            let directory = tempfile::tempdir().unwrap();
            let paths = mosaic(directory.path(), grid, (width, height), 2, channels);
            let mut options = StackOptions {
                normalization,
                ..StackOptions::default()
            };
            options.acceptance.minimum_overlap_fraction = 0.1;
            for whole in [true, false] {
                let mut stacker = LiveStacker::with_expanding_reference(
                    &paths,
                    CalibrationMasters::default(),
                    options.clone(),
                    usize::MAX,
                    |_, _, _, _| Continue::Yes,
                )
                .unwrap();
                let started = std::time::Instant::now();
                let dispositions = push_all(&mut stacker, &paths, whole);
                let per_frame = started.elapsed() / paths.len() as u32;
                assert!(
                    dispositions
                        .iter()
                        .all(|disposition| disposition.starts_with("Accepted")),
                    "{dispositions:?}"
                );
                eprintln!(
                    "{normalization:?}, {}x{} panels of {width}x{height}x{channels} on a {}x{} canvas, \
                     {}: {per_frame:?} per frame",
                    grid.0,
                    grid.1,
                    stacker.reference.width,
                    stacker.reference.height,
                    if whole { "whole canvas" } else { "footprint" },
                );
            }
        }
    }
}
