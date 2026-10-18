//! Lights binned by [`StackOptions::bin_factor`], and their reintegration at
//! the sensor's full resolution.

use super::*;
use crate::BatchStackOptions;
use seiza_fits::{F32ImageData, write_f32_image};

const WIDTH: usize = 384;
const HEIGHT: usize = 288;

/// Frame `index`'s view of the sky: a few pixels of dither and a small turn
/// about the frame's centre, and a half turn more after a meridian flip.
fn pose(index: usize, flipped: bool) -> SimilarityTransform {
    let turn = 0.004 * index as f64 + if flipped { std::f64::consts::PI } else { 0.0 };
    let (centre_x, centre_y) = ((WIDTH - 1) as f64 / 2.0, (HEIGHT - 1) as f64 / 2.0);
    let (dither_x, dither_y) = (
        ((index * 13) % 7) as f64 - 3.0 + 0.37 * index as f64,
        ((index * 7) % 5) as f64 - 2.0 - 0.21 * index as f64,
    );
    let (cosine, sine) = (turn.cos(), turn.sin());
    SimilarityTransform {
        scale: 1.0,
        rotation_radians: turn,
        translation_x: centre_x + dither_x - (cosine * centre_x - sine * centre_y),
        translation_y: centre_y + dither_y - (sine * centre_x + cosine * centre_y),
    }
}

/// `count` frames of one field of stars 3.5 pixels across at half maximum,
/// on a sky of 500 with noise, posed by [`pose`]; frame `flipped` after a
/// meridian flip. Returns each frame's path and pose.
fn field(
    directory: &Path,
    count: usize,
    flipped: Option<usize>,
) -> Vec<(PathBuf, SimilarityTransform)> {
    let mut seed = 0x0b1e_u64;
    let mut random = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    let margin = 60.0;
    let stars = (0..WIDTH * HEIGHT / 220)
        .map(|_| {
            (
                -margin + random() * (WIDTH as f64 + 2.0 * margin),
                -margin + random() * (HEIGHT as f64 + 2.0 * margin),
                150.0 + 9000.0 * random().powi(3),
            )
        })
        .collect::<Vec<_>>();
    let sigma = 3.5 / 2.354_820_045;
    (0..count)
        .map(|index| {
            let pose = pose(index, flipped == Some(index));
            let mut data = vec![500.0_f32; WIDTH * HEIGHT];
            for value in &mut data {
                let normal = (0..4).map(|_| random()).sum::<f64>() - 2.0;
                *value += (normal * 10.0) as f32;
            }
            for &(sky_x, sky_y, peak) in &stars {
                let (x, y) = pose.apply(sky_x, sky_y);
                let (left, top) = (x.floor() as isize - 6, y.floor() as isize - 6);
                for row in top.max(0)..(top + 14).min(HEIGHT as isize) {
                    for column in left.max(0)..(left + 14).min(WIDTH as isize) {
                        let r2 = (column as f64 - x).powi(2) + (row as f64 - y).powi(2);
                        data[row as usize * WIDTH + column as usize] +=
                            (peak * (-r2 / (2.0 * sigma * sigma)).exp()) as f32;
                    }
                }
            }
            let path = directory.join(format!("light-{index}.fits"));
            write_f32_image(&path, WIDTH, HEIGHT, F32ImageData::Mono(&data), &[]).unwrap();
            (path, pose)
        })
        .collect()
}

/// A stack of `frames`, every one admitted, with each frame's diagnostics.
fn stack(
    frames: &[(PathBuf, SimilarityTransform)],
    options: StackOptions,
) -> (LiveStacker, Vec<FrameDiagnostics>) {
    let mut stacker =
        LiveStacker::open_fits(&frames[0].0, None, None, None, None, options).unwrap();
    let diagnostics = frames[1..]
        .iter()
        .map(|(path, _)| match stacker.push_fits(path).unwrap() {
            FrameDisposition::Accepted(diagnostics) => diagnostics,
            FrameDisposition::Rejected(reason) => panic!("{}: {reason}", path.display()),
        })
        .collect();
    (stacker, diagnostics)
}

fn binned(bin_factor: u32) -> StackOptions {
    StackOptions {
        bin_factor,
        ..StackOptions::default()
    }
}

/// The largest distance between where `found` and `truth` carry the corners
/// of a frame of the sensor.
fn corner_error(found: SimilarityTransform, truth: SimilarityTransform) -> f64 {
    let (right, bottom) = ((WIDTH - 1) as f64, (HEIGHT - 1) as f64);
    [(0.0, 0.0), (right, 0.0), (0.0, bottom), (right, bottom)]
        .into_iter()
        .map(|(x, y)| {
            let (a, b) = found.apply(x, y);
            let (c, d) = truth.apply(x, y);
            (a - c).hypot(b - d)
        })
        .fold(0.0, f64::max)
}

/// The source-to-reference transform the truth gives frame `pose`.
fn truth(reference: SimilarityTransform, pose: SimilarityTransform) -> SimilarityTransform {
    let inverse = |t: SimilarityTransform| {
        let (x, y) = t.inverse_apply(0.0, 0.0);
        SimilarityTransform {
            scale: 1.0 / t.scale,
            rotation_radians: -t.rotation_radians,
            translation_x: x,
            translation_y: y,
        }
    };
    inverse(pose).then(reference)
}

/// Every transform registration finds on binned frames carries exactly to
/// the sensor's pixels, the frame after a meridian flip included, where
/// scaling the translation alone misses by the bin factor less one.
#[test]
fn binned_transforms_carry_to_the_sensor_through_a_half_turn() {
    let directory = tempfile::tempdir().unwrap();
    let frames = field(directory.path(), 5, Some(3));
    for bin_factor in [2, 4] {
        let (stacker, diagnostics) = stack(&frames, binned(bin_factor));
        assert_eq!(stacker.bin_factor(), bin_factor);
        let factor = bin_factor as usize;
        assert_eq!(
            (stacker.view().width, stacker.view().height),
            (WIDTH / factor, HEIGHT / factor)
        );
        for (frame, diagnostics) in frames[1..].iter().zip(&diagnostics) {
            assert_eq!(diagnostics.mapping.bin_factor(), bin_factor);
            let truth = truth(frames[0].1, frame.1);
            let native = diagnostics.mapping.rebinned(1).unwrap();
            assert_eq!(native.bin_factor(), 1);
            assert_eq!(
                (native.reference_width(), native.reference_height()),
                (WIDTH, HEIGHT)
            );
            let error = corner_error(native.transform(), truth);
            assert!(
                error < 0.15 * f64::from(bin_factor),
                "bin {bin_factor}, {}: {error:.3}px",
                frame.0.display()
            );
            // Registration RMS is reported in the sensor's pixels: the
            // native stack's frames fit at 0.2 to 0.3.
            assert!(diagnostics.registration_rms_pixels < 0.3 * f64::from(bin_factor));
            let flipped = diagnostics.transform.rotation_radians.cos() < 0.0;
            assert_eq!(flipped, frame.0.ends_with("light-3.fits"));
            if flipped {
                let mut naive = diagnostics.transform;
                naive.translation_x *= f64::from(bin_factor);
                naive.translation_y *= f64::from(bin_factor);
                assert!(
                    corner_error(naive, truth) > 0.6 * f64::from(bin_factor - 1),
                    "scaling the translation alone lands within {:.3}px",
                    corner_error(naive, truth)
                );
            }
        }
        // The drift bound is reported in the sensor's pixels.
        let native = LiveStacker::open_fits(&frames[0].0, None, None, None, None, binned(1))
            .unwrap()
            .effective_maximum_drift_pixels();
        assert!((stacker.effective_maximum_drift_pixels() - native).abs() <= f64::from(bin_factor));
    }
}

/// A binned stack reintegrated at full resolution matches the stack of the
/// same frames taken at full resolution throughout, refined or not, with
/// gains from stars or none.
///
/// [`NormalizationMode::Global`] is left out: its gain is the ratio of the
/// reference's dispersion to a resampled frame's, and on a sky this sparse
/// that is the noise's, which bilinear resampling smooths by an amount the
/// sub-pixel shift sets. The same frames took gains of 1.47 at full
/// resolution and 1.36 binned by 2, and the full-resolution pass keeps the
/// binned pass's normalization.
#[test]
fn a_binned_stack_reintegrates_at_full_resolution_as_a_native_stack() {
    let directory = tempfile::tempdir().unwrap();
    let frames = field(directory.path(), 6, Some(4));
    let options = BatchStackOptions::default();
    for normalization in [
        NormalizationMode::None,
        NormalizationMode::LocalBackground { tile_size: 64 },
    ] {
        let with = |bin_factor| StackOptions {
            normalization,
            ..binned(bin_factor)
        };
        let (native, _) = stack(&frames, with(1));
        let native = native.reintegrate(&options, |_, _, _| {}).unwrap().snapshot;
        for bin_factor in [2, 4] {
            let (stacker, _) = stack(&frames, with(bin_factor));
            for refine in [true, false] {
                let mut heard = Vec::new();
                let full = stacker
                    .reintegrate_full_resolution(&options, refine, |pass, index, count| {
                        heard.push((pass, index, count));
                    })
                    .unwrap();
                let what = format!("{normalization:?}, bin {bin_factor}, refine {refine}");
                let image = &full.result.snapshot.image;
                assert_eq!((image.width, image.height), (WIDTH, HEIGHT), "{what}");
                assert_eq!(full.result.snapshot.accepted_frames, 6, "{what}");
                assert_eq!(full.refined_frames, if refine { 5 } else { 0 }, "{what}");
                let rms = full.registration_rms_pixels.unwrap();
                assert!(rms < 0.3 * f64::from(bin_factor), "{what}: {rms}");
                assert!(
                    heard.contains(&(crate::BatchStackPass::Integrate, 5, 6)),
                    "{what}"
                );
                // Where every frame reaches, the two stacks agree to within a
                // fraction of the sky noise of one frame, and of the stars.
                let mut differences = image
                    .data
                    .iter()
                    .zip(&native.image.data)
                    .zip(full.result.snapshot.coverage.iter().zip(&native.coverage))
                    .filter(|(_, (a, b))| **a == 6 && **b == 6)
                    .map(|((a, b), _)| (a - b).abs() / b.abs().max(500.0) * 500.0)
                    .collect::<Vec<_>>();
                assert!(differences.len() > WIDTH * HEIGHT / 2, "{what}");
                differences.sort_by(f32::total_cmp);
                let median = differences[differences.len() / 2];
                let high = differences[differences.len() * 99 / 100];
                // One frame's sky noise is about 6. Refined, the frames land
                // where full-resolution registration puts them, and without
                // normalization the stacks are the same; a background fitted
                // on binned frames moves the stack by a fraction of the noise.
                // Carried unrefined, a star's flanks move by up to a few
                // percent at 4.
                let (median_limit, high_limit) = match (normalization, refine) {
                    (NormalizationMode::None, true) => (0.0, 0.0),
                    (_, true) => (1.5, 3.0),
                    (_, false) => (1.5, 15.0),
                };
                assert!(
                    median <= median_limit && high <= high_limit,
                    "{what}: median {median}, 99th percentile {high}"
                );
            }
        }
    }
}

/// A binned stack's options, mappings and full-resolution reintegration
/// survive a saved context, and default options still write no factor.
#[test]
fn a_binned_context_keeps_its_bin_factor() {
    let directory = tempfile::tempdir().unwrap();
    let frames = field(directory.path(), 4, None);
    let (stacker, _) = stack(&frames[..3], binned(2));
    let path = directory.path().join("binned.seiza-stack");
    stacker.save_context(&path).unwrap();
    let mut restored = LiveStacker::open_context(&path).unwrap();
    assert_eq!(restored.bin_factor(), 2);
    assert_eq!(
        restored.configuration_fingerprint(),
        stacker.configuration_fingerprint()
    );
    assert_eq!(
        (restored.view().width, restored.view().height),
        (WIDTH / 2, HEIGHT / 2)
    );
    let FrameDisposition::Accepted(diagnostics) = restored.push_fits(&frames[3].0).unwrap() else {
        panic!("the last frame joins the restored stack");
    };
    assert_eq!(diagnostics.mapping.bin_factor(), 2);
    let full = restored
        .reintegrate_full_resolution(&BatchStackOptions::default(), true, |_, _, _| {})
        .unwrap();
    assert_eq!(full.refined_frames, 3);
    assert_eq!(
        (
            full.result.snapshot.image.width,
            full.result.snapshot.image.height
        ),
        (WIDTH, HEIGHT)
    );

    let json = serde_json::to_string(&binned(2)).unwrap();
    assert!(json.contains("\"bin_factor\":2"), "{json}");
    let read: StackOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(read.bin_factor, 2);
    let default = serde_json::to_string(&StackOptions::default()).unwrap();
    assert!(!default.contains("bin_factor"), "{default}");
    let read: StackOptions = serde_json::from_str("{}").unwrap();
    assert_eq!(read.bin_factor, 1);
}

/// The reference's WCS and binning cards follow it onto the binned grid, and
/// back again for a full-resolution reintegration.
#[test]
fn reference_headers_follow_the_binned_grid_and_back() {
    let directory = tempfile::tempdir().unwrap();
    let mut frames = field(directory.path(), 3, None);
    let headers = [
        ("CTYPE1", HeaderValue::String("RA---TAN".into())),
        ("CTYPE2", HeaderValue::String("DEC--TAN".into())),
        ("CRVAL1", HeaderValue::Float(83.8)),
        ("CRVAL2", HeaderValue::Float(-5.4)),
        ("CRPIX1", HeaderValue::Float(192.5)),
        ("CRPIX2", HeaderValue::Float(144.5)),
        ("CD1_1", HeaderValue::Float(-3.0e-4)),
        ("CD1_2", HeaderValue::Float(1.0e-6)),
        ("CD2_1", HeaderValue::Float(1.0e-6)),
        ("CD2_2", HeaderValue::Float(3.0e-4)),
        ("XBINNING", HeaderValue::Integer(1)),
        ("YBINNING", HeaderValue::Integer(1)),
        ("XPIXSZ", HeaderValue::Float(3.76)),
        ("YPIXSZ", HeaderValue::Float(3.76)),
    ];
    let mut reference = FitsFrame::open(&frames[0].0).unwrap();
    reference.headers.extend(
        headers
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone())),
    );
    let cards = reference
        .headers
        .iter()
        .filter(|(key, _)| headers.iter().any(|(wanted, _)| key == wanted))
        .map(|(key, value)| seiza_fits::WriteHeaderCard::new(key, value.clone()))
        .collect::<Vec<_>>();
    let path = directory.path().join("solved.fits");
    write_f32_image(
        &path,
        WIDTH,
        HEIGHT,
        F32ImageData::Mono(&reference.image.data),
        &cards,
    )
    .unwrap();
    frames[0].0 = path;
    let (stacker, _) = stack(&frames, binned(2));
    let value = |headers: &[(String, HeaderValue)], key: &str| {
        headers
            .iter()
            .find(|(name, _)| name == key)
            .and_then(|(_, value)| value.as_f64())
    };
    let binned = stacker.reference_headers();
    assert_eq!(value(binned, "CRPIX1"), Some(96.5));
    assert_eq!(value(binned, "CRPIX2"), Some(72.5));
    assert_eq!(value(binned, "CD1_1"), Some(-6.0e-4));
    assert_eq!(value(binned, "XBINNING"), Some(2.0));
    assert_eq!(value(binned, "XPIXSZ"), Some(7.52));
    assert_eq!(value(binned, "SEIZABIN"), Some(2.0));
    let full = stacker
        .reintegrate_full_resolution(&BatchStackOptions::default(), false, |_, _, _| {})
        .unwrap();
    for (key, expected) in &headers {
        let found = value(&full.reference_headers, key);
        match expected.as_f64() {
            Some(expected) => {
                let found = found.unwrap();
                assert!(
                    (found - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                    "{key}: {found} against {expected}"
                );
            }
            None => assert!(full.reference_headers.iter().any(|(name, _)| name == key)),
        }
    }
    assert_eq!(value(&full.reference_headers, "SEIZABIN"), None);
}

#[test]
fn bin_factors_outside_one_to_four_and_binned_bayer_drizzle_are_refused() {
    for bin_factor in [0, 5] {
        assert!(binned(bin_factor).validate().is_err(), "{bin_factor}");
    }
    for bin_factor in 1..=4 {
        binned(bin_factor).validate().unwrap();
    }
    let drizzle = StackOptions {
        cfa_integration: CfaIntegration::BayerDrizzle,
        ..binned(2)
    };
    assert!(drizzle.validate().is_err());
    StackOptions {
        bin_factor: 1,
        ..drizzle
    }
    .validate()
    .unwrap();

    // A binned stack's mappings do not place its frames' sensor pixels.
    let directory = tempfile::tempdir().unwrap();
    let frames = field(directory.path(), 3, None);
    let (stacker, _) = stack(&frames, binned(2));
    let error = stacker
        .reintegrate_drizzled(
            &BatchStackOptions::default(),
            &crate::DrizzleOptions {
                scale: 2,
                drop_shrink: None,
            },
            |_, _, _| {},
        )
        .unwrap_err();
    assert!(error.to_string().contains("full resolution"), "{error}");
}

/// Prepared pixels are binned too, so a host pushes what its sensor took.
#[test]
fn prepared_pixels_are_binned_as_files_are() {
    let directory = tempfile::tempdir().unwrap();
    let frames = field(directory.path(), 3, None);
    let images = frames
        .iter()
        .map(|(path, _)| FitsFrame::open(path).unwrap().image)
        .collect::<Vec<_>>();
    let mut from_pixels = LiveStacker::from_linear(images[0].clone(), binned(2)).unwrap();
    let (from_files, file_diagnostics) = stack(&frames, binned(2));
    for (image, from_file) in images[1..].iter().zip(&file_diagnostics) {
        let FrameDisposition::Accepted(diagnostics) =
            from_pixels.push_linear(image.clone()).unwrap()
        else {
            panic!("prepared pixels are admitted");
        };
        assert_eq!(diagnostics.transform, from_file.transform);
    }
    let bits = |stacker: &LiveStacker| {
        stacker
            .snapshot()
            .unwrap()
            .image
            .data
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert!(bits(&from_pixels) == bits(&from_files));
}
