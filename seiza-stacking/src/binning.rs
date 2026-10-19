//! How a stack binned by [`crate::StackOptions::bin_factor`] reads the
//! pixel-valued options and constants that registration, normalization and
//! mosaic planning use.
//!
//! Every such value keeps its meaning in the sensor's own (native) pixels,
//! whatever the bin factor, so a host writes the same numbers at any factor.
//! [`Scale`] converts them once, where the working grid is built: lengths
//! divide by the factor and areas by its square, with a floor where the
//! binned stars are too undersampled for the plain quotient. Measurements
//! reported back, such as registration RMS and drift, are multiplied back to
//! native pixels. Fractions, counts, sigmas and angles do not change.

use crate::{NormalizationMode, RegistrationOptions};
use seiza::DetectConfig;

/// Native pixels per working pixel along each axis: the bin factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Scale {
    bin: u32,
}

impl Scale {
    /// The sensor's own sampling, where every value means what it says.
    pub(crate) const NATIVE: Self = Self { bin: 1 };

    pub(crate) fn new(bin: u32) -> Self {
        Self { bin: bin.max(1) }
    }

    /// The bin factor.
    pub(crate) fn bin(self) -> u32 {
        self.bin
    }

    fn factor(self) -> f64 {
        f64::from(self.bin)
    }

    /// A length of `native` pixels, in working pixels.
    pub(crate) fn length(self, native: f64) -> f64 {
        native / self.factor()
    }

    /// A length of `native` pixels in working pixels, but no shorter than
    /// `floor` working pixels once binned: a pairing radius below about a
    /// pixel drops pairs to the centroid bias of undersampled stars. Never
    /// longer than `native` itself, so a value already tighter than the
    /// floor keeps its meaning, and at the native scale it is unchanged.
    pub(crate) fn floored_length(self, native: f64, floor: f64) -> f64 {
        if self.bin == 1 {
            return native;
        }
        self.length(native).max(floor.min(native))
    }

    /// A length measured in working pixels, in native pixels.
    pub(crate) fn native_length(self, working: f64) -> f64 {
        working * self.factor()
    }

    /// Registration options in working pixels: the match tolerance, floored
    /// at a pixel, and the drift floor. The drift fraction is relative to the
    /// working frame, so it stands.
    pub(crate) fn registration(self, native: &RegistrationOptions) -> RegistrationOptions {
        let mut options = native.clone();
        options.match_tolerance_pixels = self.floored_length(native.match_tolerance_pixels, 1.0);
        options.maximum_drift_pixels = self.length(native.maximum_drift_pixels);
        options
    }

    /// A normalization mode in working pixels: its tile, no smaller than the
    /// 16 pixels every tile needs.
    pub(crate) fn normalization(self, native: NormalizationMode) -> NormalizationMode {
        let tile = |tile_size: usize| (tile_size / self.bin as usize).max(16.min(tile_size));
        match native {
            NormalizationMode::Local { tile_size } => NormalizationMode::Local {
                tile_size: tile(tile_size),
            },
            NormalizationMode::LocalBackground { tile_size } => {
                NormalizationMode::LocalBackground {
                    tile_size: tile(tile_size),
                }
            }
            other => other,
        }
    }

    /// The star detector's settings on the working grid: its background
    /// tile, at least 16 pixels; its smallest and largest component.
    ///
    /// A star of 2 native pixels FWHM binned by 4 covers a pixel or two, so
    /// the smallest component falls with the factor: 3 pixels natively, 2 at
    /// a factor of 2 and 1 from 3 up. Hot pixels were repaired before
    /// binning, and what is left of one is averaged down by the factor
    /// squared, so a lone pixel no longer passes for a star. Whether
    /// elongation and area filters should be bypassed for factors of 3 and
    /// 4 is still open; this rule stands until real frames at a FWHM near 2
    /// pixels settle it.
    pub(crate) fn detection(self, config: DetectConfig) -> DetectConfig {
        if self.bin == 1 {
            return config;
        }
        let bin = self.bin;
        DetectConfig {
            tile_size: (config.tile_size / bin).max(16),
            min_area: match bin {
                2 => config.min_area.min(2),
                _ => 1,
            },
            max_area: (config.max_area / (bin * bin)).max(1),
            ..config
        }
    }

    /// The edge margin within which a star's centroid is biased and left
    /// out: 6 native pixels, but never under 2 working pixels.
    pub(crate) fn centroid_margin(self, native: isize) -> isize {
        if self.bin == 1 {
            return native;
        }
        (native as f64 / self.factor()).ceil().max(2.0) as isize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_native_scale_changes_nothing() {
        let scale = Scale::NATIVE;
        let options = RegistrationOptions {
            match_tolerance_pixels: 0.4,
            maximum_drift_pixels: 123.0,
            ..RegistrationOptions::default()
        };
        let converted = scale.registration(&options);
        assert_eq!(converted.match_tolerance_pixels, 0.4);
        assert_eq!(converted.maximum_drift_pixels, 123.0);
        let tile = NormalizationMode::LocalBackground { tile_size: 17 };
        assert_eq!(scale.normalization(tile), tile);
        let config = DetectConfig::default();
        let detection = scale.detection(config.clone());
        assert_eq!(
            (detection.tile_size, detection.min_area, detection.max_area),
            (config.tile_size, config.min_area, config.max_area)
        );
        assert_eq!(scale.centroid_margin(6), 6);
        assert_eq!(scale.floored_length(0.3, 1.0), 0.3);
    }

    #[test]
    fn lengths_divide_with_floors_and_areas_divide_by_the_square() {
        let options = RegistrationOptions::default();
        let two = Scale::new(2).registration(&options);
        assert_eq!(two.match_tolerance_pixels, 1.25);
        assert_eq!(two.maximum_drift_pixels, 128.0);
        // 2.5 / 4 is under the floor.
        let four = Scale::new(4).registration(&options);
        assert_eq!(four.match_tolerance_pixels, 1.0);
        assert_eq!(four.maximum_drift_pixels, 64.0);
        // A tolerance already under the floor is not widened to it.
        assert_eq!(Scale::new(4).floored_length(0.5, 1.0), 0.5);

        assert_eq!(
            Scale::new(4).normalization(NormalizationMode::Local { tile_size: 256 }),
            NormalizationMode::Local { tile_size: 64 }
        );
        assert_eq!(
            Scale::new(4).normalization(NormalizationMode::LocalBackground { tile_size: 32 }),
            NormalizationMode::LocalBackground { tile_size: 16 }
        );

        let config = DetectConfig::default();
        let two = Scale::new(2).detection(config.clone());
        assert_eq!((two.tile_size, two.min_area, two.max_area), (32, 2, 5_000));
        let four = Scale::new(4).detection(config);
        assert_eq!(
            (four.tile_size, four.min_area, four.max_area),
            (16, 1, 1_250)
        );
        assert_eq!(Scale::new(3).detection(DetectConfig::default()).min_area, 1);

        assert_eq!(Scale::new(2).centroid_margin(6), 3);
        assert_eq!(Scale::new(4).centroid_margin(6), 2);
        assert_eq!(Scale::new(4).native_length(0.5), 2.0);
    }
}
