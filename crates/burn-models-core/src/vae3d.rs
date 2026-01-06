//! 3D VAE for Video Generation
//!
//! A 3D Variational Autoencoder that compresses videos spatially and temporally.
//! Used in video generation models like CogVideoX, Wan, Mochi, etc.
//!
//! # Architecture
//!
//! ```text
//! Video [B, C, T, H, W]
//!     │
//!     ▼ Encoder (3D conv + temporal downsample)
//! Latent [B, Z, T/4, H/8, W/8]
//!     │
//!     ▼ Decoder (3D conv + temporal upsample)
//! Video [B, C, T, H, W]
//! ```
//!
//! # Compression
//!
//! - Temporal: 4x (16 frames → 4 latent frames)
//! - Spatial: 8x (512x512 → 64x64)
//! - Total: 256x compression

use burn::prelude::*;

/// Configuration for 3D VAE
#[derive(Debug, Clone)]
pub struct Vae3dConfig {
    /// Input channels (usually 3 for RGB)
    pub in_channels: usize,
    /// Latent channels
    pub latent_channels: usize,
    /// Base channel multiplier
    pub base_channels: usize,
    /// Channel multipliers for each level
    pub channel_mults: Vec<usize>,
    /// Temporal compression factor
    pub temporal_compression: usize,
    /// Spatial compression factor
    pub spatial_compression: usize,
    /// Number of residual blocks per level
    pub num_res_blocks: usize,
}

impl Default for Vae3dConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            latent_channels: 4,
            base_channels: 128,
            channel_mults: vec![1, 2, 4, 4],
            temporal_compression: 4,
            spatial_compression: 8,
            num_res_blocks: 2,
        }
    }
}

impl Vae3dConfig {
    /// CogVideoX-style VAE
    pub fn cogvideox() -> Self {
        Self {
            in_channels: 3,
            latent_channels: 16,
            base_channels: 128,
            channel_mults: vec![1, 2, 4, 4],
            temporal_compression: 4,
            spatial_compression: 8,
            num_res_blocks: 2,
        }
    }

    /// Compute output shape for encoder
    pub fn encoded_shape(&self, time: usize, height: usize, width: usize) -> (usize, usize, usize) {
        (
            time / self.temporal_compression,
            height / self.spatial_compression,
            width / self.spatial_compression,
        )
    }
}

/// Temporal downsampling (average pool in time)
pub fn temporal_downsample<B: Backend>(x: Tensor<B, 5>, factor: usize) -> Tensor<B, 5> {
    let [batch, channels, time, height, width] = x.dims();
    let new_time = time / factor;
    let device = x.device();

    // Reshape to [batch, channels, new_time, factor, height, width]
    // Then average over factor dimension
    // Simplified: just slice every factor'th frame

    let mut frames = Vec::new();
    for t in 0..new_time {
        let frame = x.clone().slice([
            0..batch,
            0..channels,
            t * factor..t * factor + 1,
            0..height,
            0..width,
        ]);
        frames.push(frame);
    }

    if frames.is_empty() {
        Tensor::zeros([batch, channels, 0, height, width], &device)
    } else {
        Tensor::cat(frames, 2)
    }
}

/// Temporal upsampling (repeat frames)
pub fn temporal_upsample<B: Backend>(x: Tensor<B, 5>, factor: usize) -> Tensor<B, 5> {
    // Repeat each frame 'factor' times
    // [B, C, T, H, W] -> [B, C, T*factor, H, W]
    x.repeat_dim(2, factor)
}

/// Spatial downsampling for 3D tensors
pub fn spatial_downsample_3d<B: Backend>(x: Tensor<B, 5>, factor: usize) -> Tensor<B, 5> {
    let [batch, channels, time, height, width] = x.dims();
    let new_height = height / factor;
    let new_width = width / factor;

    // Simplified: slice to downsample
    // Real implementation would use strided conv or avg pool
    let mut result_frames = Vec::new();
    for t in 0..time {
        let frame = x.clone().slice([
            0..batch,
            0..channels,
            t..t + 1,
            0..height,
            0..width,
        ]);
        // Simple subsample (every factor'th pixel)
        let frame_3d = frame.reshape([batch, channels, height, width]);
        let subsampled = subsample_2d(frame_3d, factor);
        let frame_5d = subsampled.reshape([batch, channels, 1, new_height, new_width]);
        result_frames.push(frame_5d);
    }

    if result_frames.is_empty() {
        let device = x.device();
        Tensor::zeros([batch, channels, 0, new_height, new_width], &device)
    } else {
        Tensor::cat(result_frames, 2)
    }
}

fn subsample_2d<B: Backend>(x: Tensor<B, 4>, factor: usize) -> Tensor<B, 4> {
    let [batch, channels, height, width] = x.dims();
    let new_height = height / factor;
    let new_width = width / factor;

    // Simple subsample - take every factor'th element
    // Real implementation would do proper resampling
    let mut rows = Vec::new();
    for h in 0..new_height {
        let row = x.clone().slice([
            0..batch,
            0..channels,
            h * factor..h * factor + 1,
            0..width,
        ]);
        // Subsample width
        let mut cols = Vec::new();
        for w in 0..new_width {
            let col = row.clone().slice([
                0..batch,
                0..channels,
                0..1,
                w * factor..w * factor + 1,
            ]);
            cols.push(col);
        }
        let subsampled_row = Tensor::cat(cols, 3);
        rows.push(subsampled_row);
    }

    if rows.is_empty() {
        let device = x.device();
        Tensor::zeros([batch, channels, new_height, new_width], &device)
    } else {
        Tensor::cat(rows, 2).reshape([batch, channels, new_height, new_width])
    }
}

/// Spatial upsampling for 3D tensors (nearest neighbor)
pub fn spatial_upsample_3d<B: Backend>(x: Tensor<B, 5>, factor: usize) -> Tensor<B, 5> {
    let [batch, channels, time, height, width] = x.dims();
    let new_height = height * factor;
    let new_width = width * factor;

    let mut result_frames = Vec::new();
    for t in 0..time {
        let frame = x.clone().slice([
            0..batch,
            0..channels,
            t..t + 1,
            0..height,
            0..width,
        ]);
        let frame_4d = frame.reshape([batch, channels, height, width]);

        // Upsample height then width using repeat
        let upsampled_h = frame_4d
            .unsqueeze_dim::<5>(3)
            .repeat_dim(3, factor)
            .reshape([batch, channels, new_height, width]);

        let upsampled = upsampled_h
            .unsqueeze_dim::<5>(4)
            .repeat_dim(4, factor)
            .reshape([batch, channels, new_height, new_width]);

        let frame_5d = upsampled.reshape([batch, channels, 1, new_height, new_width]);
        result_frames.push(frame_5d);
    }

    if result_frames.is_empty() {
        let device = x.device();
        Tensor::zeros([batch, channels, 0, new_height, new_width], &device)
    } else {
        Tensor::cat(result_frames, 2)
    }
}

/// 3D VAE Encoder output
pub struct Vae3dEncoderOutput<B: Backend> {
    /// Latent mean
    pub mean: Tensor<B, 5>,
    /// Latent log variance
    pub logvar: Tensor<B, 5>,
}

impl<B: Backend> Vae3dEncoderOutput<B> {
    /// Sample from the latent distribution
    pub fn sample(&self) -> Tensor<B, 5> {
        let std = (self.logvar.clone() * 0.5).exp();
        let noise = Tensor::random_like(&self.mean, burn::tensor::Distribution::Normal(0.0, 1.0));
        self.mean.clone() + std * noise
    }

    /// Get deterministic latent (just the mean)
    pub fn deterministic(&self) -> Tensor<B, 5> {
        self.mean.clone()
    }
}

/// Statistics for video VAE operations
#[derive(Debug, Clone)]
pub struct Vae3dStats {
    pub input_shape: [usize; 5],
    pub latent_shape: [usize; 5],
    pub compression_ratio: f32,
    pub temporal_compression: usize,
    pub spatial_compression: usize,
}

impl Vae3dStats {
    pub fn new(config: &Vae3dConfig, batch: usize, time: usize, height: usize, width: usize) -> Self {
        let (lat_t, lat_h, lat_w) = config.encoded_shape(time, height, width);

        let input_elements = batch * config.in_channels * time * height * width;
        let latent_elements = batch * config.latent_channels * lat_t * lat_h * lat_w;

        Self {
            input_shape: [batch, config.in_channels, time, height, width],
            latent_shape: [batch, config.latent_channels, lat_t, lat_h, lat_w],
            compression_ratio: input_elements as f32 / latent_elements as f32,
            temporal_compression: config.temporal_compression,
            spatial_compression: config.spatial_compression,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_config_default() {
        let config = Vae3dConfig::default();
        assert_eq!(config.in_channels, 3);
        assert_eq!(config.temporal_compression, 4);
        assert_eq!(config.spatial_compression, 8);
    }

    #[test]
    fn test_encoded_shape() {
        let config = Vae3dConfig::default();
        let (t, h, w) = config.encoded_shape(16, 512, 512);
        assert_eq!(t, 4);  // 16 / 4
        assert_eq!(h, 64); // 512 / 8
        assert_eq!(w, 64); // 512 / 8
    }

    #[test]
    fn test_temporal_downsample() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 5>::zeros([1, 3, 16, 64, 64], &device);

        let down = temporal_downsample(x, 4);
        assert_eq!(down.dims(), [1, 3, 4, 64, 64]);
    }

    #[test]
    fn test_temporal_upsample() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 5>::zeros([1, 4, 4, 64, 64], &device);

        let up = temporal_upsample(x, 4);
        assert_eq!(up.dims(), [1, 4, 16, 64, 64]);
    }

    #[test]
    fn test_spatial_downsample_3d() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 5>::zeros([1, 3, 4, 64, 64], &device);

        let down = spatial_downsample_3d(x, 2);
        assert_eq!(down.dims(), [1, 3, 4, 32, 32]);
    }

    #[test]
    fn test_spatial_upsample_3d() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 5>::zeros([1, 4, 4, 32, 32], &device);

        let up = spatial_upsample_3d(x, 2);
        assert_eq!(up.dims(), [1, 4, 4, 64, 64]);
    }

    #[test]
    fn test_vae3d_stats() {
        let config = Vae3dConfig::default();
        let stats = Vae3dStats::new(&config, 1, 16, 512, 512);

        assert_eq!(stats.input_shape, [1, 3, 16, 512, 512]);
        assert_eq!(stats.latent_shape, [1, 4, 4, 64, 64]);
        // 3*16*512*512 / (4*4*64*64) = 12,582,912 / 65,536 = 192
        assert!(stats.compression_ratio > 100.0);
    }

    #[test]
    fn test_encoder_output_sample() {
        let device = Default::default();
        let mean = Tensor::<TestBackend, 5>::zeros([1, 4, 4, 64, 64], &device);
        let logvar = Tensor::<TestBackend, 5>::zeros([1, 4, 4, 64, 64], &device);

        let output = Vae3dEncoderOutput { mean, logvar };

        let sampled = output.sample();
        assert_eq!(sampled.dims(), [1, 4, 4, 64, 64]);

        let det = output.deterministic();
        assert_eq!(det.dims(), [1, 4, 4, 64, 64]);
    }
}
