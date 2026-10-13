//! Planning and registration for a canvas grown to hold a connected set of
//! overlapping frames.
//!
//! Each frame registers against the footprints of the panels it may overlap,
//! never against the whole canvas: the brightest stars of a large canvas
//! crowd into a few panels, leaving a narrow overlap elsewhere with too few
//! of them to match. A similarity places every frame; a frame lying mostly
//! within one panel also gets the polynomial warp the registration model asks
//! for, fitted against that panel.

use crate::{
    CalibrationMasters, Continue, Error, FitsFrame, FrameMetadata, LinearImage, NormalizationMap,
    NormalizationMode, ReferenceRegion, Registrar, RegistrationOptions, RegistrationResult, Result,
    SimilarityTransform, StackOptions, path_identity,
};
use seiza::DetectedStar;
use seiza_fits::HeaderValue;
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

/// A frame that covers at least this fraction of new sky starts a panel of
/// its own; dithered frames of a panel only extend its edges.
const NEW_PANEL_FRACTION: f64 = 0.25;

/// The chance a panel match may be a coincidence. Registration tries every
/// translation within its search radius, so the best of them pairs stars by
/// luck far above what any one transform would: on a real mosaic, a frame
/// sharing no sky with a panel paired 6 of its 800 brightest stars where one
/// transform expects 0.16, and in a crowded synthetic field 12 where it
/// expects 4. A true overlap between frames taken months apart paired 14 to 29.
const FALSE_MATCH_PROBABILITY: f64 = 1.0e-3;

/// The fraction of the stars a match claims the frame and panel share that
/// must actually pair. A pattern fixed to the sensor pairs far above chance
/// but only a sliver of them.
const MINIMUM_SHARED_FRACTION: f64 = 0.05;

/// A frame that neither starts a panel nor lies mostly within one keeps a
/// similarity, which cannot follow lens distortion across it, so where it
/// overlaps the frames warped onto a panel it would leave copies of each star.
/// Such a frame fits far worse than the warped frames around it: similarities
/// across this distortion fit at 1.1 to 1.4px RMS on a real mosaic, where its
/// warped frames fit at 0.04 to 0.20. One fitting more than this many times
/// the median of the frames that do not start a panel is rejected. On that
/// mosaic it rejected one S-II frame (0.43px), which cloud had already ruled
/// out, and no H-alpha or O-III frame; it guards against what the warp cannot
/// reach.
const RESIDUAL_OUTLIER_FACTOR: f64 = 5.0;

/// Below this RMS a non-starting frame is never rejected as an outlier,
/// however small the mosaic's median.
const RESIDUAL_OUTLIER_FLOOR_PIXELS: f64 = 0.3;

/// The non-starting frames a median of their residuals needs.
const RESIDUAL_OUTLIER_SAMPLE: usize = 5;

/// How many times the matching budget of stars planning pairs. An overlap
/// holds only a fraction of each frame's brightest stars, and frames taken on
/// different nights rank them differently; on a three-panel mosaic shot months
/// apart, 200 stars paired only 8 across one overlap and put the rotation half
/// a degree out, while 800 paired 29 and found it.
const PLANNING_STARS_FACTOR: usize = 4;

/// The brightest stars whose triangles seed a panel match. A narrow overlap
/// holds few of a frame's brightest stars, and fewer still in a filter as
/// faint as S-II: across one such overlap, triangles from 24 stars locked
/// onto a cluster at -0.78 degrees, pairing 22 stars at 1.86px RMS, while 40
/// or more agreed on -0.53 degrees, pairing twice as many at 1.27px, for
/// about a tenth of a second more per match.
const PLANNING_TRIANGLE_STARS: usize = 40;

/// The fraction of a frame's stars that must lie within the panel it matched
/// for a polynomial warp to be fitted against that panel. Lens distortion
/// reverses across a meridian flip and is largest at the frame's edges, where
/// panels overlap: on a real mosaic a similarity left 1.16px RMS between the
/// two sides of the pier, and a quadratic 0.06px. Fitted through a narrow
/// overlap, though, a quadratic would extrapolate across the rest of the
/// frame, so a frame starting a panel keeps its similarity.
const WARP_COVERAGE: f64 = 0.5;

/// How close a frame's own star must land to a panel star for photometry to
/// measure that star. Lens distortion bends opposite edges of two panels
/// opposite ways: on a real mosaic a similarity matched the middle of an
/// overlap and missed its ends by 17px, where apertures read empty sky, and
/// the panel's gain came out 3.2 against a true 0.97.
const PHOTOMETRY_PAIRING_PIXELS: f64 = 1.5;

/// Cleanly measured stars a panel's gain needs. An overlap holds few stars,
/// and fewer still pair up across distortion: on a real S-II mosaic one join
/// had 10. Their median beats the alternative, the ratio of the two panels'
/// noise, which there read 3.0 where the stars said about 1.1.
const PANEL_PHOTOMETRY_STARS: usize = 5;

/// Within this many degrees of a half turn from the anchor, a frame is taken
/// to come from the other side of the pier. Sessions on one rig differ by a
/// few degrees when the camera comes off between them, by 10.8 on a real
/// mosaic; a frame at a quarter turn is neither side, and a panel may start
/// from it.
const PIER_FLIP_TOLERANCE_DEGREES: f64 = 20.0;

/// Where planning put a frame, or why it could not.
pub(crate) struct PlannedRegistration {
    pub result: std::result::Result<RegistrationResult, String>,
    stamp: Option<crate::replay::SourceStamp>,
    /// Whether the frame anchored or started a panel, and so keeps a
    /// similarity by design.
    starts_panel: bool,
}

/// The planned registration of the frame read from `path`: its placement,
/// the reason it could not join the canvas, or `None` when it was not
/// planned.
pub(crate) fn registration_for<'a>(
    plans: &'a HashMap<PathBuf, PlannedRegistration>,
    path: Option<&PathBuf>,
) -> Result<Option<std::result::Result<&'a RegistrationResult, &'a str>>> {
    let Some(path) = path else { return Ok(None) };
    let Some(plan) = plans.get(&path_identity(path)) else {
        return Ok(None);
    };
    if plan.stamp != crate::replay::SourceStamp::of(path) {
        return Err(Error::Stack(format!(
            "{} changed after expansion planning",
            path.display()
        )));
    }
    Ok(Some(plan.result.as_ref().map_err(String::as_str)))
}

/// Whether a frame at `rotation_radians` to the anchor lies on the other side
/// of the pier.
fn flipped(rotation_radians: f64) -> bool {
    (rotation_radians.to_degrees().rem_euclid(360.0) - 180.0).abs() <= PIER_FLIP_TOLERANCE_DEGREES
}

/// One panel's footprint on the canvas, with the stars inside it.
#[derive(Clone, Debug)]
struct Window {
    region: ReferenceRegion,
    registrar: Registrar,
    /// The registrar's stars, on the panel's own grid.
    stars: Vec<(f64, f64)>,
    maximum_stars: usize,
    match_tolerance_pixels: f64,
    /// Roughly independent transforms registration tries: translations
    /// within the search radius one tolerance apart, and triangle candidates.
    hypotheses: f64,
}

impl Window {
    fn new(
        canvas: &LinearImage,
        region: ReferenceRegion,
        options: &RegistrationOptions,
    ) -> Result<Self> {
        Self::from_pixels(&crop(canvas, region, 0, 0)?, region, options)
    }

    /// A panel at `region` whose stars are detected in `pixels`, an image of
    /// the region's size.
    fn from_pixels(
        pixels: &LinearImage,
        region: ReferenceRegion,
        options: &RegistrationOptions,
    ) -> Result<Self> {
        let mut options = options.clone();
        // A frame overlapping this panel anywhere lies within about two of
        // its diagonals of the panel's center.
        options.maximum_drift_pixels = 2.0 * (region.width as f64).hypot(region.height as f64);
        options.maximum_stars *= PLANNING_STARS_FACTOR;
        options.triangle_stars = options.triangle_stars.max(PLANNING_TRIANGLE_STARS);
        let maximum_stars = options.maximum_stars;
        let match_tolerance_pixels = options.match_tolerance_pixels;
        let radius = options.effective_maximum_drift_pixels(region.width, region.height);
        let hypotheses =
            (radius / match_tolerance_pixels).powi(2) + options.maximum_candidates as f64;
        let registrar = Registrar::new(pixels, options)?;
        Ok(Self {
            region,
            stars: registrar.reference_star_positions(),
            registrar,
            maximum_stars,
            match_tolerance_pixels,
            hypotheses,
        })
    }

    /// Register a `width` by `height` source's stars to this panel by
    /// similarity, returning a canvas transform and the fraction of the
    /// source's stars that land within the panel.
    fn register(
        &self,
        stars: &[DetectedStar],
        width: usize,
        height: usize,
    ) -> Result<(RegistrationResult, f64)> {
        let mut result = self.registrar.register_similarity_stars(stars)?;
        let transform = result.transform;
        let inside = |(x, y): (f64, f64), width: usize, height: usize| {
            x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64
        };
        let source_stars = &stars[..stars.len().min(self.maximum_stars)];
        let source_shared = source_stars
            .iter()
            .filter(|star| {
                inside(
                    transform.apply(star.x, star.y),
                    self.region.width,
                    self.region.height,
                )
            })
            .count();
        let panel_shared = self
            .stars
            .iter()
            .filter(|&&(x, y)| inside(transform.inverse_apply(x, y), width, height))
            .count();
        // Pairs one transform finds by chance: each star it places on the
        // other image finds one of that image's stars within the tolerance as
        // often as their density allows.
        let disc = std::f64::consts::PI * self.match_tolerance_pixels.powi(2);
        let panel_density =
            self.stars.len() as f64 / (self.region.width * self.region.height) as f64;
        let source_density = source_stars.len() as f64 / (width * height) as f64;
        let chance =
            (source_shared as f64 * panel_density).max(panel_shared as f64 * source_density) * disc;
        let required = chance_ceiling(chance, self.hypotheses);
        if result.matched_stars < required {
            return Err(Error::Registration(format!(
                "{} stars matched; the best of {:.0} transforms pairs up to {} by chance",
                result.matched_stars,
                self.hypotheses,
                required - 1
            )));
        }
        // Features fixed to the sensor pair at the same pixels in every frame,
        // so a frame of another target can match at no shift or turn at all:
        // two uncalibrated lights of M81 among a Rosette mosaic paired 9 to
        // 11 of the 800 stars they claimed to share, at exactly the anchor's
        // placement. Real overlaps paired a tenth or more.
        let shared = source_shared.min(panel_shared);
        if (result.matched_stars as f64) < MINIMUM_SHARED_FRACTION * shared as f64 {
            return Err(Error::Registration(format!(
                "only {} of the {shared} stars the overlap should share matched",
                result.matched_stars
            )));
        }
        result.transform.translation_x += self.region.x as f64;
        result.transform.translation_y += self.region.y as f64;
        Ok((
            result,
            source_shared as f64 / source_stars.len().max(1) as f64,
        ))
    }

    /// Fit the registration model's polynomial warp against this panel around
    /// a canvas `registration` of the source whose `stars` these are. Leaves
    /// a similarity model's registration, or one too few stars support,
    /// unchanged.
    fn warp(&self, registration: &mut RegistrationResult, stars: &[DetectedStar]) {
        let mut local = registration.transform;
        local.translation_x -= self.region.x as f64;
        local.translation_y -= self.region.y as f64;
        if let Some((warp, pairs, rms)) = self.registrar.warp_stars(local, stars) {
            registration.warp = Some(warp.translated(self.region.x as f64, self.region.y as f64));
            registration.matched_stars = pairs;
            registration.rms_error_pixels = rms;
        }
    }
}

/// Registers frames to an expanded canvas through its panels' footprints.
#[derive(Clone, Debug)]
pub(crate) struct CanvasRegistrar {
    width: usize,
    height: usize,
    windows: Vec<Window>,
    stars: Vec<(f64, f64)>,
}

impl CanvasRegistrar {
    pub(crate) fn new(
        canvas: &LinearImage,
        regions: &[ReferenceRegion],
        options: &RegistrationOptions,
    ) -> Result<Self> {
        if regions.is_empty() {
            return Err(Error::Stack("an expanded canvas needs a panel".into()));
        }
        let windows = regions
            .iter()
            .map(|region| {
                if region.width == 0
                    || region.height == 0
                    || region.x + region.width > canvas.width
                    || region.y + region.height > canvas.height
                {
                    return Err(Error::Stack(format!(
                        "panel {region:?} lies outside the {}x{} canvas",
                        canvas.width, canvas.height
                    )));
                }
                Window::new(canvas, *region, options)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::from_windows(canvas.width, canvas.height, windows))
    }

    fn from_windows(width: usize, height: usize, windows: Vec<Window>) -> Self {
        // Overlapping panels share stars; photometry should count each once.
        let mut seen = HashSet::new();
        let stars = windows
            .iter()
            .flat_map(|window| {
                window
                    .stars
                    .iter()
                    .map(|(x, y)| (x + window.region.x as f64, y + window.region.y as f64))
            })
            .filter(|(x, y)| seen.insert((x.round() as i64, y.round() as i64)))
            .collect();
        Self {
            width,
            height,
            windows,
            stars,
        }
    }

    /// The anchor panel's registrar, on the anchor's own grid. An expanded
    /// stack registers through [`Self::register`]; this stands in where a
    /// single registrar is required, without detecting stars canvas-wide.
    pub(crate) fn anchor_registrar(&self) -> Registrar {
        self.windows[0].registrar.clone()
    }

    pub(crate) fn regions(&self) -> Vec<ReferenceRegion> {
        self.windows.iter().map(|window| window.region).collect()
    }

    /// Every panel's brightest stars, on the canvas grid.
    pub(crate) fn star_positions(&self) -> Vec<(f64, f64)> {
        self.stars.clone()
    }

    /// The best-matched registration among the panels, warped against that
    /// panel when the frame lies mostly within it.
    pub(crate) fn register(&self, source: &LinearImage) -> Result<RegistrationResult> {
        let stars = self.windows[0].registrar.detect_source_stars(source);
        let mut best: Option<(RegistrationResult, f64, &Window)> = None;
        let mut last_error = None;
        for window in &self.windows {
            match window.register(&stars, source.width, source.height) {
                Ok((result, coverage)) => {
                    if best
                        .as_ref()
                        .is_none_or(|(best, _, _)| better(&result, best))
                    {
                        best = Some((result, coverage, window));
                    }
                }
                Err(error) => last_error = Some(error),
            }
        }
        let (mut best, coverage, window) = best.ok_or_else(|| {
            last_error.unwrap_or_else(|| Error::Registration("no panel matched".into()))
        })?;
        if coverage >= WARP_COVERAGE {
            window.warp(&mut best, &stars);
        }
        best.drift_pixels = best
            .transform
            .displacement_at(self.width as f64 * 0.5, self.height as f64 * 0.5);
        Ok(best)
    }
}

/// Reject the frames that neither start a panel nor fit like the frames
/// warped onto one; see [`RESIDUAL_OUTLIER_FACTOR`].
fn reject_residual_outliers(plans: &mut HashMap<PathBuf, PlannedRegistration>) {
    let mut residuals = plans
        .values()
        .filter(|plan| !plan.starts_panel)
        .filter_map(|plan| plan.result.as_ref().ok())
        .map(|result| result.rms_error_pixels)
        .collect::<Vec<_>>();
    if residuals.len() < RESIDUAL_OUTLIER_SAMPLE {
        return;
    }
    residuals.sort_by(f64::total_cmp);
    let median = residuals[residuals.len() / 2];
    let limit = (RESIDUAL_OUTLIER_FACTOR * median).max(RESIDUAL_OUTLIER_FLOOR_PIXELS);
    for plan in plans.values_mut().filter(|plan| !plan.starts_panel) {
        if let Ok(result) = &plan.result
            && result.rms_error_pixels > limit
        {
            plan.result = Err(format!(
                "registered at {:.2}px RMS where this mosaic's frames median {median:.2}px, \
                 too loosely to stack without leaving copies of the stars it overlaps",
                result.rms_error_pixels
            ));
        }
    }
}

/// The `panel` stars within `tolerance` of a `landed` frame star.
fn paired_stars(panel: &[(f64, f64)], landed: &[(f64, f64)], tolerance: f64) -> Vec<(f64, f64)> {
    let cell = |x: f64, y: f64| {
        (
            (x / tolerance).floor() as i64,
            (y / tolerance).floor() as i64,
        )
    };
    let mut cells = HashMap::<(i64, i64), Vec<(f64, f64)>>::new();
    for &(x, y) in landed {
        cells.entry(cell(x, y)).or_default().push((x, y));
    }
    panel
        .iter()
        .copied()
        .filter(|&(x, y)| {
            let (cx, cy) = cell(x, y);
            (cx - 1..=cx + 1).any(|nx| {
                (cy - 1..=cy + 1).any(|ny| {
                    cells.get(&(nx, ny)).is_some_and(|near| {
                        near.iter()
                            .any(|&(sx, sy)| (sx - x).hypot(sy - y) <= tolerance)
                    })
                })
            })
        })
        .collect()
}

/// The fewest pairs that `hypotheses` transforms, each pairing a Poisson
/// count of mean `chance` stars by coincidence, reach with probability at
/// most [`FALSE_MATCH_PROBABILITY`].
fn chance_ceiling(chance: f64, hypotheses: f64) -> usize {
    let limit = (FALSE_MATCH_PROBABILITY / hypotheses.max(1.0)).ln();
    let chance = chance.max(1.0e-12);
    // log P(X = k) for k up to well past the mean, then tails from the top.
    let last = (chance + 20.0 * chance.sqrt() + 60.0).ceil() as usize;
    let mut log_pmf = Vec::with_capacity(last + 1);
    let mut log_factorial = 0.0;
    for k in 0..=last {
        if k > 0 {
            log_factorial += (k as f64).ln();
        }
        log_pmf.push(k as f64 * chance.ln() - chance - log_factorial);
    }
    let mut log_tail = f64::NEG_INFINITY;
    let mut ceiling = last + 1;
    for k in (0..=last).rev() {
        let (high, low) = if log_pmf[k] > log_tail {
            (log_pmf[k], log_tail)
        } else {
            (log_tail, log_pmf[k])
        };
        log_tail = high + (low - high).exp().ln_1p();
        if log_tail > limit {
            break;
        }
        ceiling = k;
    }
    ceiling
}

fn better(candidate: &RegistrationResult, incumbent: &RegistrationResult) -> bool {
    candidate.matched_stars > incumbent.matched_stars
        || (candidate.matched_stars == incumbent.matched_stars
            && candidate.rms_error_pixels < incumbent.rms_error_pixels)
}

pub(crate) struct Expansion {
    pub image: LinearImage,
    pub headers: Vec<(String, HeaderValue)>,
    pub metadata: FrameMetadata,
    pub noise: Vec<f32>,
    pub registrations: HashMap<PathBuf, PlannedRegistration>,
    pub canvas: CanvasRegistrar,
}

/// The canvas as planning grows it.
struct Plan {
    image: LinearImage,
    headers: Vec<(String, HeaderValue)>,
    /// Finite samples on the canvas.
    finite: usize,
    windows: Vec<Window>,
    /// The panel the previous frame joined, which its dithered neighbors
    /// most likely overlap too.
    last_window: usize,
    registrations: HashMap<PathBuf, PlannedRegistration>,
}

/// Each frame joins the growing canvas through actual common stars and pixels.
/// Unconnected frames are retried after growth, so input order need not follow
/// adjacency. `maximum_samples` bounds the canvas's pixels times channels.
pub(crate) fn plan(
    paths: &[PathBuf],
    calibration: &CalibrationMasters,
    options: &StackOptions,
    maximum_samples: usize,
    mut progress: impl FnMut(&Path, usize, usize, usize) -> Continue,
) -> Result<Expansion> {
    let first = paths
        .first()
        .ok_or_else(|| Error::Stack("expansion needs light frames".into()))?;
    let frame = FitsFrame::open(first)?;
    let metadata = frame.metadata();
    let (frame, _) = crate::stack::prepare_reference(frame, calibration, options)?;
    if frame.image.sample_count() > maximum_samples {
        return Err(Error::Stack(format!(
            "the anchor frame's {} samples exceed the expanded-canvas limit of {maximum_samples}",
            frame.image.sample_count()
        )));
    }
    let noise = if options.weighting.is_equal() {
        Vec::new()
    } else {
        crate::stack::measure_reference_noise(&frame.image)?
    };
    let anchor = ReferenceRegion {
        x: 0,
        y: 0,
        width: frame.image.width,
        height: frame.image.height,
    };
    let anchor = Window::new(&frame.image, anchor, &options.registration)?;
    let mut plan = Plan {
        finite: frame.image.data.iter().filter(|v| v.is_finite()).count(),
        registrations: HashMap::from([(
            path_identity(first),
            PlannedRegistration {
                stamp: crate::replay::SourceStamp::of(first),
                starts_panel: true,
                result: Ok(RegistrationResult {
                    transform: SimilarityTransform::IDENTITY,
                    // The anchor matches each of its own stars exactly.
                    matched_stars: anchor.stars.len(),
                    rms_error_pixels: 0.0,
                    drift_pixels: 0.0,
                    warp: None,
                }),
            },
        )]),
        image: frame.image,
        headers: frame.headers,
        windows: vec![anchor],
        last_window: 0,
    };
    let prepare = |path: &Path, channels: usize| -> Result<LinearImage> {
        let (frame, _) =
            crate::stack::prepare_reference(FitsFrame::open(path)?, calibration, options)?;
        if frame.image.channels != channels {
            return Err(Error::Stack(format!(
                "{} has a different channel count",
                path.display()
            )));
        }
        Ok(frame.image)
    };
    let mut pending = paths[1..]
        .iter()
        .map(|path| Pending {
            path,
            stars: Vec::new(),
            size: None,
            tried: 0,
            error: None,
        })
        .collect::<Vec<_>>();
    // Whether a frame from the other side of the pier may start a panel: only
    // once a pass finds nothing else to join.
    let mut flipped_start = false;
    while !pending.is_empty() {
        let before = pending.len();
        let mut deferred = Vec::new();
        let mut first_error = None;
        let mut waiting_for_pier_side = false;
        for mut item in pending {
            let path = item.path;
            if progress(
                path,
                plan.registrations.len(),
                plan.image.width,
                plan.image.height,
            ) == Continue::No
            {
                return Err(Error::Cancelled);
            }
            // A frame that matched no panel keeps its stars, and is read
            // again only once a panel added since then matches it.
            let mut image = None;
            if item.size.is_none() {
                let prepared = prepare(path, plan.image.channels)?;
                item.stars = plan.windows[0].registrar.detect_source_stars(&prepared);
                item.size = Some((prepared.width, prepared.height));
                image = Some(prepared);
            }
            let (width, height) = item.size.expect("the frame was measured");
            let panels = plan.windows.len();
            let joined = match plan.register(&item.stars, width, height, item.tried, options) {
                Err(error) => {
                    item.tried = panels;
                    Err(error)
                }
                Ok((registration, window, coverage)) => {
                    let image = match image {
                        Some(image) => image,
                        None => prepare(path, plan.image.channels)?,
                    };
                    let joined = plan.join(
                        &image,
                        &item.stars,
                        (registration, window, coverage),
                        flipped_start,
                        options,
                        maximum_samples,
                    );
                    // The canvas may yet grow enough for any panel to admit it.
                    item.tried = 0;
                    joined
                }
            };
            match joined {
                Ok(Some((registration, starts_panel))) => {
                    plan.registrations.insert(
                        path_identity(path),
                        PlannedRegistration {
                            result: Ok(registration),
                            stamp: crate::replay::SourceStamp::of(path),
                            starts_panel,
                        },
                    );
                }
                Ok(None) => {
                    waiting_for_pier_side = true;
                    deferred.push(item);
                }
                Err(error) => {
                    first_error.get_or_insert_with(|| format!("{}: {error}", path.display()));
                    item.error = Some(error.to_string());
                    deferred.push(item);
                }
            }
        }
        if deferred.len() == before {
            if waiting_for_pier_side && !flipped_start {
                flipped_start = true;
                pending = deferred;
                continue;
            }
            // Nothing joined the anchor: there is no mosaic to build.
            if plan.registrations.len() == 1 {
                return Err(Error::Stack(format!(
                    "cannot expand: {} frame(s) could not join the overlapping mosaic; {}",
                    deferred.len(),
                    first_error.unwrap_or_default()
                )));
            }
            // A frame of another target, or cloud, would otherwise cost the
            // whole mosaic; it is rejected when pushed, with the reason.
            for item in deferred {
                plan.registrations.insert(
                    path_identity(item.path),
                    PlannedRegistration {
                        result: Err(item.error.unwrap_or_else(|| "no panel matched".into())),
                        stamp: crate::replay::SourceStamp::of(item.path),
                        starts_panel: false,
                    },
                );
            }
            break;
        }
        flipped_start = false;
        pending = deferred;
    }
    reject_residual_outliers(&mut plan.registrations);
    let (width, height) = (plan.image.width, plan.image.height);
    for planned in plan
        .registrations
        .values_mut()
        .filter_map(|planned| planned.result.as_mut().ok())
    {
        planned.drift_pixels = planned
            .transform
            .displacement_at(width as f64 * 0.5, height as f64 * 0.5);
    }
    if progress(first, plan.registrations.len(), width, height) == Continue::No {
        return Err(Error::Cancelled);
    }
    Ok(Expansion {
        canvas: CanvasRegistrar::from_windows(width, height, plan.windows),
        image: plan.image,
        headers: plan.headers,
        metadata,
        noise,
        registrations: plan.registrations,
    })
}

/// A frame resampled onto the canvas it would grow, before committing.
struct Placement {
    width: usize,
    height: usize,
    dx: usize,
    dy: usize,
    transform: SimilarityTransform,
    region: ReferenceRegion,
    incoming: LinearImage,
    existing: LinearImage,
    /// Samples both define, and samples only the frame defines.
    common: usize,
    uncovered: usize,
}

/// A frame waiting to join the canvas.
struct Pending<'a> {
    path: &'a PathBuf,
    stars: Vec<DetectedStar>,
    /// Width and height, once read.
    size: Option<(usize, usize)>,
    /// Panels it has already failed to match.
    tried: usize,
    /// Why it last failed to join.
    error: Option<String>,
}

impl Plan {
    /// Grow the canvas to the footprint of `source`, registered to `window`
    /// with `coverage` of its stars inside it, and fill the sky it newly
    /// covers. Leaves the plan untouched on failure, and returns `None`
    /// without joining a frame that would start a panel from the other side
    /// of the pier than the anchor, unless `flipped_start` allows it.
    fn join(
        &mut self,
        source: &LinearImage,
        stars: &[DetectedStar],
        (registration, window, coverage): (RegistrationResult, usize, f64),
        flipped_start: bool,
        options: &StackOptions,
        maximum_samples: usize,
    ) -> Result<Option<(RegistrationResult, bool)>> {
        let criteria = options.acceptance;
        let finite_source = source.data.iter().filter(|v| v.is_finite()).count() as f64;
        let mut registration = registration;
        let mut refined = false;
        let placed = loop {
            let placed = self.place(source, &registration, options, maximum_samples)?;
            // A frame starting a panel sets the geometry its neighbors follow,
            // so it is refit; frames within known sky keep the panel's fit.
            let source_area = finite_source * placed.transform.scale.powi(2);
            if refined || (placed.uncovered as f64) < NEW_PANEL_FRACTION * source_area {
                break placed;
            }
            refined = true;
            match self.refine(&registration, source, stars, options) {
                Some(better) => registration = better,
                None => break placed,
            }
        };
        // A panel's first frame keeps its similarity, so its edges carry its
        // lens distortion into the seams with its neighbors. That distortion
        // reverses across a meridian flip; a panel started from the anchor's
        // side of the pier meets its neighbors on equal terms.
        if refined && !flipped_start && flipped(registration.transform.rotation_radians) {
            return Ok(None);
        }
        let Placement {
            width,
            height,
            dx,
            dy,
            transform,
            region,
            mut incoming,
            existing,
            common,
            uncovered,
        } = placed;
        let source_area = finite_source * transform.scale.powi(2);
        let overlap = common as f64 / (self.finite as f64).min(source_area).max(1.0);
        if common == 0 || overlap < f64::from(criteria.minimum_overlap_fraction) {
            return Err(Error::Stack(format!(
                "panel overlap {overlap:.3} is below {}",
                criteria.minimum_overlap_fraction
            )));
        }

        // Only a panel's first frame fills the canvas. Its siblings would add
        // slivers at most, resampled through a similarity that across a
        // meridian flip misses the stars that photometry measures; the
        // integration normalizes every frame on its own.
        let starts_panel = refined;
        if starts_panel && uncovered > 0 {
            // Match the new sky to the canvas through the overlap. The canvas
            // is a registration and photometric template, not an integrated
            // observation, so covered pixels keep their first value.
            let landed = stars
                .iter()
                .map(|star| {
                    let (x, y) = transform.apply(star.x, star.y);
                    (x - region.x as f64, y - region.y as f64)
                })
                .collect::<Vec<_>>();
            let panel_stars = self.windows[window]
                .stars
                .iter()
                .map(|&(x, y)| {
                    (
                        x + (self.windows[window].region.x + dx) as f64 - region.x as f64,
                        y + (self.windows[window].region.y + dy) as f64 - region.y as f64,
                    )
                })
                .filter(|&(x, y)| {
                    x >= 0.0 && y >= 0.0 && x < region.width as f64 && y < region.height as f64
                })
                .collect::<Vec<_>>();
            // Panels are shot on different nights, under different skies, so
            // their noise says nothing of their transparency: a gain comes
            // from stars or not at all, leaving only the background matched.
            let gains = matches!(
                options.normalization,
                NormalizationMode::LocalBackground { .. }
            )
            .then(|| {
                [
                    PHOTOMETRY_PAIRING_PIXELS,
                    options.registration.match_tolerance_pixels,
                ]
                .into_iter()
                .find_map(|tolerance| {
                    crate::normalization::photometric_gains_from(
                        &existing,
                        &incoming,
                        &paired_stars(&panel_stars, &landed, tolerance),
                        PANEL_PHOTOMETRY_STARS,
                    )
                })
                .unwrap_or_else(|| vec![1.0; incoming.channels])
            });
            let normalization = NormalizationMap::estimate_with_gains(
                &existing,
                &incoming,
                options.normalization,
                gains.as_deref(),
            )?;
            let (low, high) = normalization.gain_range();
            if low < criteria.minimum_normalization_gain
                || high > criteria.maximum_normalization_gain
            {
                return Err(Error::Stack(
                    "panel normalization gain exceeds admission limits".into(),
                ));
            }
            normalization.apply(&mut incoming)?;
        }

        // Every check has passed: commit.
        if (width, height, dx, dy) != (self.image.width, self.image.height, 0, 0) {
            self.grow(width, height, dx, dy)?;
        }
        let channels = self.image.channels;
        let mut filled = 0;
        for row in (0..region.height).filter(|_| starts_panel) {
            let start = ((region.y + row) * width + region.x) * channels;
            let target = &mut self.image.data[start..start + region.width * channels];
            let from = row * region.width * channels;
            for (target, value) in target
                .iter_mut()
                .zip(&incoming.data[from..from + region.width * channels])
            {
                if !target.is_finite() && value.is_finite() {
                    *target = *value;
                    filled += 1;
                }
            }
        }
        self.finite += filled;
        let mut joined = RegistrationResult {
            transform,
            ..registration
        };
        if !starts_panel && coverage >= WARP_COVERAGE {
            self.windows[window].warp(&mut joined, stars);
        }
        // The panel's stars come from its first frame alone. The canvas under
        // its footprint also holds its neighbours' edges, placed by their own
        // similarities and bent by distortion the other way; warps fitted to
        // such a mixture compromise between the two. On a real mosaic, frames
        // warped onto a panel built from the canvas fitted at 0.48 to 0.72px
        // RMS and left copies of each star; built from its first frame, the
        // same frames fit at 0.05px.
        if filled as f64 >= NEW_PANEL_FRACTION * source_area
            && let Ok(panel) = Window::from_pixels(&incoming, region, &options.registration)
        {
            self.windows.push(panel);
            self.last_window = self.windows.len() - 1;
        }
        Ok(Some((joined, starts_panel)))
    }

    /// Register a `width` by `height` frame's stars to the panel the previous
    /// frame joined when it lies mostly within it, or else to the panel from
    /// `first` on that matches the most stars. Returns the registration, the panel, and the fraction of
    /// the frame's stars within it.
    fn register(
        &mut self,
        stars: &[DetectedStar],
        width: usize,
        height: usize,
        first: usize,
        options: &StackOptions,
    ) -> Result<(RegistrationResult, usize, f64)> {
        let mut best: Option<(RegistrationResult, usize, f64)> = None;
        let mut last_error = None;
        let last = Some(self.last_window).filter(|&last| last >= first);
        let others = (first..self.windows.len())
            .rev()
            .filter(|&i| Some(i) != last);
        for index in last.into_iter().chain(others) {
            let result = self.windows[index]
                .register(stars, width, height)
                .and_then(|(result, coverage)| Ok((admissible(result, options)?, coverage)));
            match result {
                // A dither of the previous frame lies within its panel, and
                // that panel settles it. A frame only touching the panel's
                // edge goes on to the others: on a real mosaic, frames of
                // one panel taken after a neighbour's matched that neighbour
                // through a 14% strip by 12 stars, missing their own panel's
                // hundreds, and were placed far off.
                Ok((result, coverage)) if Some(index) == last && coverage >= WARP_COVERAGE => {
                    best = Some((result, index, coverage));
                    break;
                }
                Ok((result, coverage)) => {
                    if best
                        .as_ref()
                        .is_none_or(|(best, _, _)| better(&result, best))
                    {
                        best = Some((result, index, coverage));
                    }
                }
                Err(error) => last_error = Some(error),
            }
        }
        let best = best.ok_or_else(|| {
            last_error.unwrap_or_else(|| Error::Registration("no panel matched".into()))
        })?;
        self.last_window = best.1;
        Ok(best)
    }

    /// Where `source` lands under `registration`: the canvas that holds it,
    /// its resampled footprint, and the canvas beneath.
    fn place(
        &self,
        source: &LinearImage,
        registration: &RegistrationResult,
        options: &StackOptions,
        maximum_samples: usize,
    ) -> Result<Placement> {
        let (width, height, dx, dy) =
            union_bounds(&self.image, source, registration.transform, maximum_samples)?;
        let mut transform = registration.transform;
        transform.translation_x += dx as f64;
        transform.translation_y += dy as f64;
        let region = footprint(source, transform, width, height)
            .ok_or_else(|| Error::Stack("panel footprint is empty".into()))?;
        let incoming = crate::resample_region_to_reference_with(
            source,
            width,
            height,
            region,
            transform,
            options.interpolation,
        )?;
        let existing = crop(&self.image, region, dx, dy)?;
        let (mut common, mut uncovered) = (0, 0);
        for (a, b) in existing.data.iter().zip(&incoming.data) {
            if b.is_finite() {
                if a.is_finite() {
                    common += 1;
                } else {
                    uncovered += 1;
                }
            }
        }
        Ok(Placement {
            width,
            height,
            dx,
            dy,
            transform,
            region,
            incoming,
            existing,
            common,
            uncovered,
        })
    }

    /// A panel's overlap with a neighbor can be a narrow strip, whose few
    /// stars leave rotation and scale loose; the frame's far edge would then
    /// miss its other neighbors. Refit against every canvas star the frame
    /// covers. `None` unless more stars pair up than with the panel alone.
    fn refine(
        &self,
        registration: &RegistrationResult,
        source: &LinearImage,
        stars: &[DetectedStar],
        options: &StackOptions,
    ) -> Option<RegistrationResult> {
        let region = footprint(
            source,
            registration.transform,
            self.image.width,
            self.image.height,
        )?;
        let covered = crop(&self.image, region, 0, 0).ok()?;
        let mut initial = registration.transform;
        initial.translation_x -= region.x as f64;
        initial.translation_y -= region.y as f64;
        let mut refined = crate::registration::refine_similarity(
            &covered,
            source,
            stars,
            initial,
            &options.registration,
            options.registration.maximum_stars * PLANNING_STARS_FACTOR,
        )
        .filter(|refined| refined.matched_stars > registration.matched_stars)?;
        refined = admissible(refined, options).ok()?;
        refined.transform.translation_x += region.x as f64;
        refined.transform.translation_y += region.y as f64;
        Some(refined)
    }

    /// Move the canvas into a larger one, `dx` and `dy` from its origin.
    fn grow(&mut self, width: usize, height: usize, dx: usize, dy: usize) -> Result<()> {
        let channels = self.image.channels;
        let mut padded = blank(width, height, channels)?;
        let row = self.image.width * channels;
        for y in 0..self.image.height {
            let to = ((y + dy) * width + dx) * channels;
            padded.data[to..to + row].copy_from_slice(&self.image.data[y * row..(y + 1) * row]);
        }
        self.image = padded;
        for window in &mut self.windows {
            window.region.x += dx;
            window.region.y += dy;
        }
        for registered in self
            .registrations
            .values_mut()
            .filter_map(|planned| planned.result.as_mut().ok())
        {
            registered.transform.translation_x += dx as f64;
            registered.transform.translation_y += dy as f64;
            if let Some(warp) = &mut registered.warp {
                *warp = warp.translated(dx as f64, dy as f64);
            }
        }
        for (key, value) in &mut self.headers {
            let delta = match key.as_str() {
                "CRPIX1" => dx,
                "CRPIX2" => dy,
                _ => continue,
            };
            if let Some(pixel) = value.as_f64().filter(|v| v.is_finite()) {
                *value = HeaderValue::Float(pixel + delta as f64);
            }
        }
        Ok(())
    }
}

fn admissible(
    registration: RegistrationResult,
    options: &StackOptions,
) -> Result<RegistrationResult> {
    let criteria = options.acceptance;
    if registration.rms_error_pixels > criteria.maximum_registration_rms_pixels
        || (registration.transform.scale - 1.0).abs() > criteria.maximum_scale_deviation
        || crate::stack::rotation_deviation_degrees(registration.transform.rotation_radians)
            > criteria.maximum_rotation_degrees
    {
        return Err(Error::Stack(
            "panel transform exceeds registration admission limits".into(),
        ));
    }
    Ok(registration)
}

/// Corners of `source` under `transform`, as (left, top, right, bottom)
/// bounds of the pixels they touch.
fn corner_bounds(source: &LinearImage, transform: SimilarityTransform) -> (f64, f64, f64, f64) {
    let snap = |v: f64| {
        if (v - v.round()).abs() < 1.0e-6 {
            v.round()
        } else {
            v
        }
    };
    let mut bounds = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for (x, y) in [
        (0.0, 0.0),
        ((source.width - 1) as f64, 0.0),
        (0.0, (source.height - 1) as f64),
        ((source.width - 1) as f64, (source.height - 1) as f64),
    ] {
        let (x, y) = transform.apply(x, y);
        bounds.0 = bounds.0.min(snap(x).floor());
        bounds.1 = bounds.1.min(snap(y).floor());
        bounds.2 = bounds.2.max(snap(x).ceil() + 1.0);
        bounds.3 = bounds.3.max(snap(y).ceil() + 1.0);
    }
    bounds
}

/// The canvas size holding both `reference` and `source`, and where the
/// reference's origin lands on it.
fn union_bounds(
    reference: &LinearImage,
    source: &LinearImage,
    transform: SimilarityTransform,
    maximum_samples: usize,
) -> Result<(usize, usize, usize, usize)> {
    let (left, top, right, bottom) = corner_bounds(source, transform);
    let (left, top) = (left.min(0.0), top.min(0.0));
    let (right, bottom) = (
        right.max(reference.width as f64),
        bottom.max(reference.height as f64),
    );
    let (w, h) = (right - left, bottom - top);
    if !w.is_finite()
        || !h.is_finite()
        || w <= 0.0
        || h <= 0.0
        || w * h * reference.channels as f64 > maximum_samples as f64
    {
        return Err(Error::Stack(format!(
            "expanded canvas {w}x{h} exceeds the limit of {maximum_samples} samples"
        )));
    }
    Ok((w as usize, h as usize, (-left) as usize, (-top) as usize))
}

/// The canvas pixels `source` may cover under `transform`.
fn footprint(
    source: &LinearImage,
    transform: SimilarityTransform,
    width: usize,
    height: usize,
) -> Option<ReferenceRegion> {
    let (left, top, right, bottom) = corner_bounds(source, transform);
    let x = left.max(0.0) as usize;
    let y = top.max(0.0) as usize;
    let right = (right.max(0.0) as usize).min(width);
    let bottom = (bottom.max(0.0) as usize).min(height);
    (right > x && bottom > y).then_some(ReferenceRegion {
        x,
        y,
        width: right - x,
        height: bottom - y,
    })
}

/// `region` of a canvas about to move `dx` and `dy` from its origin: the
/// region is on the moved grid, and samples off the present canvas are `NaN`.
fn crop(
    canvas: &LinearImage,
    region: ReferenceRegion,
    dx: usize,
    dy: usize,
) -> Result<LinearImage> {
    let channels = canvas.channels;
    let mut output = blank(region.width, region.height, channels)?;
    for row in 0..region.height {
        let Some(y) = (region.y + row)
            .checked_sub(dy)
            .filter(|y| *y < canvas.height)
        else {
            continue;
        };
        let start = region.x.max(dx);
        let end = (region.x + region.width).min(canvas.width + dx);
        if end <= start {
            continue;
        }
        let from = (y * canvas.width + start - dx) * channels;
        let to = (row * region.width + start - region.x) * channels;
        let count = (end - start) * channels;
        output.data[to..to + count].copy_from_slice(&canvas.data[from..from + count]);
    }
    Ok(output)
}

fn blank(width: usize, height: usize, channels: usize) -> Result<LinearImage> {
    let count = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| Error::Stack("expanded canvas dimensions overflow".into()))?;
    let mut data = Vec::new();
    data.try_reserve_exact(count)
        .map_err(|e| Error::Stack(format!("cannot allocate expanded canvas: {e}")))?;
    data.resize(count, f32::NAN);
    LinearImage::new(width, height, channels, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_include_rotated_corners_and_negative_origins() {
        let reference = blank(40, 20, 1).unwrap();
        let source = blank(30, 10, 1).unwrap();
        for rotation in [0.0, 0.2, std::f64::consts::PI] {
            let transform = SimilarityTransform {
                scale: 1.02,
                rotation_radians: rotation,
                translation_x: -12.3,
                translation_y: -8.7,
            };
            let (w, h, dx, dy) = union_bounds(&reference, &source, transform, 10_000).unwrap();
            assert!(w >= reference.width + dx && h >= reference.height + dy);
            let mut shifted = transform;
            shifted.translation_x += dx as f64;
            shifted.translation_y += dy as f64;
            let region = footprint(&source, shifted, w, h).unwrap();
            for (x, y) in [(0.0, 0.0), (29.0, 0.0), (0.0, 9.0), (29.0, 9.0)] {
                let (x, y) = shifted.apply(x, y);
                assert!(x >= 0.0 && x < w as f64 && y >= 0.0 && y < h as f64);
                assert!(x >= region.x as f64 && x < (region.x + region.width) as f64);
                assert!(y >= region.y as f64 && y < (region.y + region.height) as f64);
            }
        }
    }

    #[test]
    fn sample_limit_counts_channels() {
        let reference = blank(10, 10, 3).unwrap();
        let source = blank(10, 10, 3).unwrap();
        let shifted = SimilarityTransform {
            translation_x: 5.0,
            ..SimilarityTransform::IDENTITY
        };
        // 15x10 pixels, three channels.
        assert!(union_bounds(&reference, &source, shifted, 15 * 10 * 3).is_ok());
        assert!(union_bounds(&reference, &source, shifted, 15 * 10 * 3 - 1).is_err());
    }

    #[test]
    fn chance_ceiling_corrects_for_the_transforms_searched() {
        // One transform expecting 0.16 pairs rarely reaches 3, but the best
        // of 85 million reaches 6; 8 is out of reach.
        assert_eq!(chance_ceiling(0.16, 1.0), 3);
        assert_eq!(chance_ceiling(0.16, 8.5e7), 8);
        // A crowded field expecting 4 pairs per transform.
        let crowded = chance_ceiling(4.0, 2.6e5);
        assert!((18..=24).contains(&crowded), "{crowded}");
        assert!(chance_ceiling(64.0, 2.6e5) > 100);
    }

    #[test]
    fn paired_stars_keep_only_panel_stars_with_a_frame_star_nearby() {
        let panel = [(10.0, 10.0), (50.0, 50.0), (90.0, 12.0)];
        let landed = [(10.8, 10.9), (58.0, 50.0), (89.0, 13.0), (0.0, 0.0)];
        assert_eq!(
            paired_stars(&panel, &landed, 1.5),
            vec![(10.0, 10.0), (90.0, 12.0)]
        );
        assert_eq!(paired_stars(&panel, &landed, 10.0).len(), 3);
    }

    #[test]
    fn residual_outliers_exclude_panel_starts_and_need_a_median() {
        let planned = |rms: f64, starts_panel: bool| PlannedRegistration {
            result: Ok(RegistrationResult {
                transform: SimilarityTransform::IDENTITY,
                matched_stars: 100,
                rms_error_pixels: rms,
                drift_pixels: 0.0,
                warp: None,
            }),
            stamp: None,
            starts_panel,
        };
        let mut plans = HashMap::new();
        for (index, rms) in [0.05, 0.06, 0.07, 0.07, 0.08, 0.09].into_iter().enumerate() {
            plans.insert(PathBuf::from(format!("{index}")), planned(rms, false));
        }
        plans.insert("start".into(), planned(1.3, true));
        plans.insert("partial".into(), planned(0.54, false));
        plans.insert("seeing".into(), planned(0.25, false));
        reject_residual_outliers(&mut plans);
        assert!(plans[Path::new("start")].result.is_ok());
        assert!(plans[Path::new("seeing")].result.is_ok(), "under the floor");
        assert!(
            plans[Path::new("partial")]
                .result
                .as_ref()
                .unwrap_err()
                .contains("0.54px RMS")
        );
        // Too few frames for a median: nothing is judged.
        let mut few = HashMap::from([
            ("a".into(), planned(0.05, false)),
            ("b".into(), planned(0.9, false)),
        ]);
        reject_residual_outliers(&mut few);
        assert!(few.values().all(|plan| plan.result.is_ok()));
    }

    #[test]
    fn crop_reads_a_moving_canvas() {
        let canvas = LinearImage::new(3, 2, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let region = ReferenceRegion {
            x: 0,
            y: 0,
            width: 3,
            height: 2,
        };
        let moved = crop(&canvas, region, 1, 1).unwrap();
        assert!(moved.data[..4].iter().all(|v| v.is_nan()));
        assert_eq!(&moved.data[4..], &[1.0, 2.0]);
        assert_eq!(crop(&canvas, region, 0, 0).unwrap().data, canvas.data);
    }
}
