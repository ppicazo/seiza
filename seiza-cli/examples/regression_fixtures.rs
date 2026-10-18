//! Fixture generation and pixel verification for regression/run.py.
//! All image I/O deliberately goes through seiza-image-io.
use anyhow::{Result, bail, ensure};
use seiza::catalog::TileSetBuilder;
use seiza_fits::F32ImageData;
use seiza_image_io::{SaveOptions, save_f32};
use serde_json::{Value, json};
use std::path::Path;

const SIZE: usize = 384;

fn write(path: &Path, pixels: &[f32], headers: &[seiza_fits::WriteHeaderCard]) -> Result<()> {
    save_f32(
        path,
        SIZE,
        SIZE,
        F32ImageData::Mono(pixels),
        &SaveOptions {
            headers,
            ..SaveOptions::default()
        },
    )?;
    Ok(())
}

fn generate(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut cases = Vec::new();
    // Integer PRNG and fixed seeds keep the scene independent of rand versions.
    let mut seed = 0x5e12a_u64;
    let mut random = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1_u64 << 53) as f64
    };
    let mut stars: Vec<(f64, f64, f64)> = Vec::new();
    while stars.len() < 70 {
        let (x, y) = (18.0 + random() * 348.0, 18.0 + random() * 348.0);
        if stars
            .iter()
            .all(|&(sx, sy, _)| (x - sx).hypot(y - sy) > 24.0)
        {
            stars.push((x, y, 0.2 + stars.len() as f64 / 100.0));
        }
    }
    let field = |sx: f64, sy: f64| {
        let mut pixels = vec![0.01_f32; SIZE * SIZE];
        for &(x, y, flux) in &stars {
            for dy in -6..=6 {
                for dx in -6..=6 {
                    let ix = (x + sx).floor() as i32 + dx;
                    let iy = (y + sy).floor() as i32 + dy;
                    if !(0..SIZE as i32).contains(&ix) || !(0..SIZE as i32).contains(&iy) {
                        continue;
                    }
                    let r2 = (f64::from(ix) - x - sx).powi(2) + (f64::from(iy) - y - sy).powi(2);
                    pixels[iy as usize * SIZE + ix as usize] += (flux * (-r2 / 3.2).exp()) as f32;
                }
            }
        }
        pixels
    };
    let pixels = field(0.0, 0.0);
    write(&dir.join("truth.fits"), &pixels, &[])?;
    for (name, center, rotation, mirrored, distorted) in [
        ("linear", (150.5, 33.2), 27.0, false, false),
        ("wrap-mirrored", (359.98, -20.0), 131.0, true, false),
        ("polar", (30.0, 88.0), 73.0, false, false),
        ("sip", (150.5, 33.2), 27.0, false, true),
    ] {
        // Analytic truth deliberately does not call the solver's WCS/SIP code.
        let sky = |x: f64, y: f64| {
            let (mut u, mut v) = (x - 191.5, y - 191.5);
            if distorted {
                let radial = 1.0 + 1.5e-7 * (u * u + v * v);
                u *= radial;
                v *= radial;
            }
            let (sin_r, cos_r) = f64::to_radians(rotation).sin_cos();
            let parity = if mirrored { -1.0 } else { 1.0 };
            let xi = ((-parity * cos_r * u + sin_r * v) * 8.0 / 3600.0).to_radians();
            let eta = ((-parity * sin_r * u - cos_r * v) * 8.0 / 3600.0).to_radians();
            let (ra0, dec0) = (f64::to_radians(center.0), f64::to_radians(center.1));
            let denominator = dec0.cos() - eta * dec0.sin();
            (
                (ra0 + xi.atan2(denominator)).to_degrees().rem_euclid(360.0),
                (dec0.sin() + eta * dec0.cos())
                    .atan2(denominator.hypot(xi))
                    .to_degrees(),
            )
        };
        let mut catalog = TileSetBuilder::new(3, 2000.0, "Seiza regression corpus v1");
        for &(x, y, flux) in &stars {
            let (ra, dec) = sky(x, y);
            catalog.add(ra, dec, (-2.5 * flux.log10() + 7.0) as f32);
        }
        catalog.write_to(&dir.join(format!("{name}.bin")))?;
        if name == "linear" {
            let mut unrelated = TileSetBuilder::new(3, 2000.0, "Unrelated regression stars");
            for i in 0..70 {
                let (ra, dec) = sky(18.0 + random() * 348.0, 18.0 + random() * 348.0);
                unrelated.add(ra, dec, 7.0 + i as f32 / 100.0);
            }
            unrelated.write_to(&dir.join("unrelated.bin"))?;
        }

        let controls: Vec<_> = [30.0, 110.0, 191.5, 280.0, 353.0]
            .into_iter()
            .flat_map(|y| {
                let sky = &sky;
                [30.0, 110.0, 191.5, 280.0, 353.0]
                    .into_iter()
                    .map(move |x| {
                        let (ra, dec) = sky(x, y);
                        json!({"x": x, "y": y, "ra": ra, "dec": dec})
                    })
            })
            .collect();
        let hint = json!({"centerRaDeg": center.0, "centerDecDeg": center.1, "radiusDeg": 2.0,
            "scaleArcsecPerPixel": 8.0, "scaleTolerance": 0.2, "sipOrder": if distorted {3} else {0}});
        cases.push(json!({"id": name, "kind": "solve", "catalog": format!("{name}.bin"),
            "image": "truth.fits", "params": {"mode": "hinted", "hint": hint},
            "expected": {"controls": controls, "maxControlArcsec": 2.0, "maxRmsArcsec": 2.0, "minMatches": 40,
                "projection": if distorted {"TAN-SIP"} else {"TAN"}}}));
    }
    let mut unrelated = cases[0].clone();
    unrelated["id"] = json!("unrelated-catalog");
    unrelated["catalog"] = json!("unrelated.bin");
    unrelated["expected"] = json!({"errorCode": -32010});
    cases.push(unrelated);
    let mut blind = cases[0].clone();
    blind["id"] = json!("blind-linear");
    blind["params"] = json!({"mode": "blind", "blind": {"minScaleArcsecPerPixel": 6.0, "maxScaleArcsecPerPixel": 10.0}});
    cases.push(blind);
    write(&dir.join("blank.fits"), &vec![0.01; SIZE * SIZE], &[])?;
    let mut noise = vec![0.0; SIZE * SIZE];
    for value in &mut noise {
        *value = 0.01 + random() as f32 * 0.002;
    }
    write(&dir.join("noise.fits"), &noise, &[])?;
    for image in ["blank.fits", "noise.fits"] {
        let mut case = cases[0].clone();
        case["id"] = json!(image.trim_end_matches(".fits"));
        case["image"] = json!(image);
        case["expected"] = json!({"errorCode": -32010});
        cases.push(case);
    }
    // Known additive dark and spatially varying flat. Shifted lights exercise
    // actual registration; no normalization can conceal calibration mistakes.
    let flat: Vec<_> = (0..SIZE * SIZE)
        .map(|i| 0.8 + (i % SIZE) as f32 / (SIZE - 1) as f32 * 0.4)
        .collect();
    let card = |name: &str, value: &str| {
        seiza_fits::WriteHeaderCard::new(name, seiza_fits::HeaderValue::String(value.into()))
    };
    let exposure =
        seiza_fits::WriteHeaderCard::new("EXPTIME", seiza_fits::HeaderValue::Float(60.0));
    write(
        &dir.join("dark.fits"),
        &vec![0.02; SIZE * SIZE],
        &[card("SEIZAMST", "DARK"), exposure.clone()],
    )?;
    write(&dir.join("flat.fits"), &flat, &[card("SEIZAMST", "FLAT")])?;
    for (i, shift) in [0.0, 3.0, -2.0].into_iter().enumerate() {
        let light: Vec<_> = field(shift, shift)
            .iter()
            .zip(&flat)
            .map(|(v, f)| v * [0.9, 1.0, 1.1][i] * f + 0.02)
            .collect();
        write(
            &dir.join(format!("light-{i}.fits")),
            &light,
            std::slice::from_ref(&exposure),
        )?;
    }
    for reintegrate in [false, true] {
        cases.push(json!({"id": if reintegrate {"calibrated-reintegrated"} else {"calibrated-stack"}, "kind": "stack",
            "args": ["stack", "light-0.fits", "light-1.fits", "light-2.fits", "--dark", "dark.fits", "--flat", "flat.fits",
                "--reference", "first", "--normalization", "none", "--rejection", "none", "--weighting", "equal"],
            "reintegrate": reintegrate, "expected": {"acceptedFrames": 3, "maxPixelRmse": 0.002, "maxFluxRelativeError": 0.01}}));
    }
    // Dithered lights with no dark to subtract the sensor's hot and dead
    // pixels, which the stack must learn from the lights and repair. Drawn
    // after everything above, so the other fixtures keep their bytes.
    let shifts: [(i32, i32); 12] = [
        (0, 0),
        (3, -2),
        (-4, 5),
        (6, 1),
        (-5, -3),
        (2, 6),
        (-1, -6),
        (5, -5),
        (-6, 2),
        (4, 4),
        (-3, -4),
        (1, 3),
    ];
    // Each defect lands, frame by frame, at its sensor pixel less that
    // frame's shift on the reference grid; keep every landing clear of the
    // stars and the border, so the residual there measures repair alone.
    let landings = |x: i32, y: i32| shifts.map(|(dx, dy)| (x - dx, y - dy));
    let mut defects = Vec::new();
    while defects.len() < 20 {
        let (x, y) = (
            (20.0 + random() * 344.0) as i32,
            (20.0 + random() * 344.0) as i32,
        );
        let clear = landings(x, y).iter().all(|&(lx, ly)| {
            (20..SIZE as i32 - 20).contains(&lx)
                && (20..SIZE as i32 - 20).contains(&ly)
                && stars
                    .iter()
                    .all(|&(sx, sy, _)| (f64::from(lx) - sx).hypot(f64::from(ly) - sy) > 10.0)
        });
        let apart = defects
            .iter()
            .all(|&(dx, dy, _): &(i32, i32, f32)| (dx - x).abs() > 3 || (dy - y).abs() > 3);
        if clear && apart {
            // Saturating, bright and moderate hot pixels, and a dead one.
            let amplitude = [1.0, 0.3, 0.1, 0.05, -1.0][defects.len() % 5];
            defects.push((x, y, amplitude));
        }
    }
    let mut normal = || {
        let (u, v) = (random().max(f64::MIN_POSITIVE), random());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let mut lights = Vec::new();
    for (index, &(dx, dy)) in shifts.iter().enumerate() {
        let mut light: Vec<_> = field(f64::from(dx), f64::from(dy))
            .iter()
            .map(|value| value + (0.001 * normal()) as f32)
            .collect();
        for &(x, y, amplitude) in &defects {
            let pixel = &mut light[y as usize * SIZE + x as usize];
            *pixel = if amplitude < 0.0 {
                0.0
            } else {
                *pixel + amplitude
            };
        }
        let name = format!("dithered-{index:02}.fits");
        write(&dir.join(&name), &light, &[])?;
        lights.push(name);
    }
    let probes: Vec<_> = defects
        .iter()
        .flat_map(|&(x, y, _)| landings(x, y))
        .map(|(x, y)| json!([x, y]))
        .collect();
    std::fs::write(dir.join("defect-probes.json"), serde_json::to_vec(&probes)?)?;
    let mut args = vec![json!("stack")];
    args.extend(lights.iter().map(|name| json!(name)));
    for flag in [
        "--reference",
        "first",
        "--normalization",
        "none",
        "--rejection",
        "none",
        "--weighting",
        "equal",
    ] {
        args.push(json!(flag));
    }
    cases.push(json!({"id": "dithered-no-dark", "kind": "stack", "args": args, "probes": "defect-probes.json",
        "expected": {"acceptedFrames": shifts.len(), "maxPixelRmse": 0.002, "maxFluxRelativeError": 0.01,
            "maxDefectResidual": 0.003}}));
    std::fs::write(
        dir.join("cases.json"),
        serde_json::to_vec_pretty(&json!({"schemaVersion": 1, "cases": cases}))?,
    )?;
    Ok(())
}

/// Compare a stack with the truth; with `probes`, a JSON array of `[x, y]`
/// reference pixels, also report the largest difference within one pixel
/// of any of them.
fn check_stack(dir: &Path, output: &Path, probes: Option<&Path>) -> Result<Value> {
    let truth = seiza_image_io::open(&dir.join("truth.fits"))?;
    let actual = seiza_image_io::open(output)?;
    ensure!(
        (truth.width, truth.height, truth.channels)
            == (actual.width, actual.height, actual.channels),
        "stack dimensions changed"
    );
    let mut squared = 0.0;
    let mut count = 0;
    let mut truth_flux = 0.0;
    let mut actual_flux = 0.0;
    for y in 12..SIZE - 12 {
        for x in 12..SIZE - 12 {
            let i = y * SIZE + x;
            let a = f64::from(actual.samples[i]);
            let t = f64::from(truth.samples[i]);
            ensure!(a.is_finite(), "nonfinite output pixel");
            squared += (a - t).powi(2);
            count += 1;
            truth_flux += t - 0.01;
            actual_flux += a - 0.01;
        }
    }
    let mut metrics = json!({"pixelRmse": (squared / f64::from(count)).sqrt(), "fluxRelativeError": (actual_flux / truth_flux - 1.0).abs()});
    if let Some(probes) = probes {
        let probes: Vec<(usize, usize)> = serde_json::from_slice(&std::fs::read(probes)?)?;
        ensure!(!probes.is_empty(), "no defect probes");
        let mut residual = 0.0_f64;
        for (x, y) in probes {
            ensure!(
                (1..SIZE - 1).contains(&x) && (1..SIZE - 1).contains(&y),
                "probe off the grid"
            );
            for py in y - 1..=y + 1 {
                for px in x - 1..=x + 1 {
                    let i = py * SIZE + px;
                    let difference = f64::from(actual.samples[i]) - f64::from(truth.samples[i]);
                    ensure!(difference.is_finite(), "nonfinite output pixel at a probe");
                    residual = residual.max(difference.abs());
                }
            }
        }
        metrics["defectResidual"] = json!(residual);
    }
    Ok(metrics)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("generate") if args.len() == 3 => generate(Path::new(&args[2])),
        Some("check-stack") if (4..=5).contains(&args.len()) => {
            let probes = args.get(4).map(Path::new);
            println!(
                "{}",
                check_stack(Path::new(&args[2]), Path::new(&args[3]), probes)?
            );
            Ok(())
        }
        _ => bail!("usage: regression_fixtures generate DIR | check-stack DIR OUTPUT [PROBES]"),
    }
}
