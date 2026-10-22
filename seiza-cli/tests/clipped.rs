use seiza_fits::{F32ImageData, FitsImage, HeaderValue, WriteHeaderCard, write_f32_image};
use std::process::Command;

const WIDTH: usize = 160;
const HEIGHT: usize = 128;

/// A sparse star field over a flat sky, as a raw frame's pixels.
fn star_field() -> Vec<f32> {
    let stars = [
        (19.7_f32, 16.4_f32),
        (71.3, 28.1),
        (132.2, 34.8),
        (43.1, 49.7),
        (103.4, 58.3),
        (22.8, 70.2),
        (82.7, 76.5),
        (143.1, 87.8),
        (54.4, 96.2),
        (116.8, 104.1),
        (31.2, 113.0),
        (91.5, 118.4),
    ];
    (0..WIDTH * HEIGHT)
        .map(|index| {
            let (x, y) = ((index % WIDTH) as f32, (index / WIDTH) as f32);
            let noise = ((index * 7919) % 23) as f32 * 0.12;
            stars
                .iter()
                .enumerate()
                .map(|(star, &(sx, sy))| {
                    (900.0 + star as f32 * 130.0)
                        * (-((x - sx).powi(2) + (y - sy).powi(2)) / 3.2).exp()
                })
                .sum::<f32>()
                + 100.0
                + noise
        })
        .collect()
}

fn run(args: &[&str]) -> String {
    let result = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8_lossy(&result.stdout).into_owned()
}

/// `seiza stack --clipped` writes how much of each pixel came from frames
/// whose raw pixels reached the saturation level the frames declare.
#[test]
fn stack_writes_the_share_of_each_pixel_its_frames_clipped() {
    let directory = tempfile::tempdir().unwrap();
    let saturate = [WriteHeaderCard::new(
        "SATURATE",
        HeaderValue::Float(60000.0),
    )];
    let field = star_field();
    let mut clipped = field.clone();
    let core = [(91, 118), (92, 118), (91, 119), (92, 119)];
    for &(x, y) in &core {
        clipped[y * WIDTH + x] = 60000.0;
    }
    let first = directory.path().join("light-1.fits");
    let second = directory.path().join("light-2.fits");
    write_f32_image(&first, WIDTH, HEIGHT, F32ImageData::Mono(&field), &saturate).unwrap();
    write_f32_image(
        &second,
        WIDTH,
        HEIGHT,
        F32ImageData::Mono(&clipped),
        &saturate,
    )
    .unwrap();
    let output = directory.path().join("stack.fits");
    let map = directory.path().join("clipped.fits");
    let stdout = run(&[
        "stack",
        first.to_str().unwrap(),
        second.to_str().unwrap(),
        "--reference",
        "first",
        "--normalization",
        "none",
        "--output",
        output.to_str().unwrap(),
        "--clipped",
        map.to_str().unwrap(),
    ]);
    assert!(stdout.contains("clipped share of each pixel"), "{stdout}");
    let map = FitsImage::open(&map).unwrap();
    assert_eq!((map.width, map.height, map.planes), (WIDTH, HEIGHT, 1));
    let map = map.into_physical_f32();
    for &(x, y) in &core {
        let share = map[y * WIDTH + x];
        assert!((0.3..=0.5).contains(&share), "({x}, {y}): {share}");
    }
    assert_eq!(map[10 * WIDTH + 10], 0.0);

    // Frames that never reach their ceiling write no map.
    write_f32_image(
        directory.path().join("light-3.fits"),
        WIDTH,
        HEIGHT,
        F32ImageData::Mono(&field),
        &saturate,
    )
    .unwrap();
    let quiet = directory.path().join("quiet.fits");
    let stdout = run(&[
        "stack",
        first.to_str().unwrap(),
        directory.path().join("light-3.fits").to_str().unwrap(),
        "--output",
        directory.path().join("plain.fits").to_str().unwrap(),
        "--clipped",
        quiet.to_str().unwrap(),
    ]);
    assert!(stdout.contains("no clipped map"), "{stdout}");
    assert!(!quiet.exists());
}

/// `color-calibrate --clipped` renders the pixels the map marks neutral in
/// the calibrated FITS, and refuses a map that does not fit the image.
#[test]
fn color_calibrate_renders_clipped_pixels_neutral() {
    let directory = tempfile::tempdir().unwrap();
    let (width, height) = (64, 48);
    // A tinted, clipped core at (20, 20); sky elsewhere.
    let mut data = Vec::with_capacity(width * height * 3);
    for index in 0..width * height {
        let core = index == 20 * width + 20;
        data.extend(if core {
            [700.0_f32, 800.0, 780.0]
        } else {
            [10.0, 20.0, 15.0]
        });
    }
    let image = directory.path().join("rgb.fits");
    write_f32_image(
        &image,
        width,
        height,
        F32ImageData::RgbInterleaved(&data),
        &[],
    )
    .unwrap();
    let mut marks = vec![0.0_f32; width * height];
    marks[20 * width + 20] = 1.0;
    let map = directory.path().join("clipped.fits");
    write_f32_image(&map, width, height, F32ImageData::Mono(&marks), &[]).unwrap();
    // Without a solution calibration is skipped, but the image is written:
    // with the map, its clipped core comes out neutral.
    let output = directory.path().join("out.fits");
    let display = directory.path().join("out.png");
    run(&[
        "color-calibrate",
        image.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--display",
        display.to_str().unwrap(),
        "--clipped",
        map.to_str().unwrap(),
    ]);
    // Planar on disk: each plane holds one channel.
    let written = FitsImage::open(&output).unwrap().into_physical_f32();
    let plane = width * height;
    let core = 20 * width + 20;
    let channels = |pixel: usize| [0, 1, 2].map(|channel| written[channel * plane + pixel]);
    assert_eq!(channels(core), [800.0; 3]);
    assert_eq!(channels(0), [10.0, 20.0, 15.0]);

    // A map of another image is refused before anything is written.
    let small = directory.path().join("small.fits");
    write_f32_image(&small, 5, 5, F32ImageData::Mono(&[0.0; 25]), &[]).unwrap();
    let refused = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "color-calibrate",
            image.to_str().unwrap(),
            "--output",
            directory.path().join("refused.fits").to_str().unwrap(),
            "--display",
            directory.path().join("refused.png").to_str().unwrap(),
            "--clipped",
            small.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("does not fit"),
        "{}",
        String::from_utf8_lossy(&refused.stderr)
    );
    assert!(!directory.path().join("refused.fits").exists());
}
