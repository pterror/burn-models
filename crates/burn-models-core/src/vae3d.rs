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
use burn::module::Param;

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

/// 3D Convolution parameters
#[derive(Debug, Clone)]
pub struct Conv3dParams {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: [usize; 3], // [T, H, W]
    pub stride: [usize; 3],
    pub padding: [usize; 3],
}

impl Conv3dParams {
    pub fn new(in_ch: usize, out_ch: usize) -> Self {
        Self {
            in_channels: in_ch,
            out_channels: out_ch,
            kernel_size: [3, 3, 3],
            stride: [1, 1, 1],
            padding: [1, 1, 1],
        }
    }

    pub fn with_kernel(mut self, kernel: [usize; 3]) -> Self {
        self.kernel_size = kernel;
        self
    }

    pub fn with_stride(mut self, stride: [usize; 3]) -> Self {
        self.stride = stride;
        self
    }

    pub fn with_padding(mut self, padding: [usize; 3]) -> Self {
        self.padding = padding;
        self
    }
}

/// 3D Convolution layer
///
/// Implements 3D convolution using the im2col (image to column) approach:
/// 1. Extract all overlapping 3D patches from input
/// 2. Reshape patches into columns
/// 3. Matrix multiply with weight matrix
/// 4. Reshape result to output volume
#[derive(Module, Debug)]
pub struct Conv3d<B: Backend> {
    /// Convolution weights [out_channels, in_channels * kernel_t * kernel_h * kernel_w]
    pub weight: Param<Tensor<B, 2>>,
    /// Bias [out_channels]
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// Input channels
    #[module(skip)]
    pub in_channels: usize,
    /// Output channels
    #[module(skip)]
    pub out_channels: usize,
    /// Kernel size [T, H, W]
    #[module(skip)]
    pub kernel_size: [usize; 3],
    /// Stride [T, H, W]
    #[module(skip)]
    pub stride: [usize; 3],
    /// Padding [T, H, W]
    #[module(skip)]
    pub padding: [usize; 3],
}

impl<B: Backend> Conv3d<B> {
    pub fn new(params: &Conv3dParams, device: &B::Device) -> Self {
        let [k_t, k_h, k_w] = params.kernel_size;
        let kernel_elements = k_t * k_h * k_w;
        let fan_in = params.in_channels * kernel_elements;

        // Kaiming/He initialization
        let bound = (1.0 / fan_in as f64).sqrt() as f32;
        let weight = Tensor::random(
            [params.out_channels, fan_in],
            burn::tensor::Distribution::Uniform((-bound).into(), bound.into()),
            device,
        );
        let bias = Tensor::zeros([params.out_channels], device);

        Self {
            weight: Param::from_tensor(weight),
            bias: Some(Param::from_tensor(bias)),
            in_channels: params.in_channels,
            out_channels: params.out_channels,
            kernel_size: params.kernel_size,
            stride: params.stride,
            padding: params.padding,
        }
    }

    /// Forward pass using im2col approach
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [batch, _in_ch, _time, _height, _width] = x.dims();
        let [k_t, k_h, k_w] = self.kernel_size;
        let [s_t, s_h, s_w] = self.stride;
        let [p_t, p_h, p_w] = self.padding;

        // Pad input if needed
        let x_padded = if p_t > 0 || p_h > 0 || p_w > 0 {
            pad_5d(x, self.padding)
        } else {
            x
        };
        let [_, _, pad_t, pad_h, pad_w] = x_padded.dims();

        // Calculate output dimensions
        let out_t = (pad_t - k_t) / s_t + 1;
        let out_h = (pad_h - k_h) / s_h + 1;
        let out_w = (pad_w - k_w) / s_w + 1;

        // im2col: extract patches and reshape to columns
        // Output shape: [batch, in_ch * k_t * k_h * k_w, out_t * out_h * out_w]
        let cols = im2col_3d(
            x_padded,
            self.kernel_size,
            self.stride,
            [out_t, out_h, out_w],
        );

        // Matrix multiply: [out_ch, in_ch*k] @ [batch, in_ch*k, out_positions]
        let weight = self.weight.val();
        let [out_ch, in_k] = weight.dims();

        // cols is [batch, in_ch*k, out_positions]
        // weight is [out_ch, in_ch*k]
        // We want [batch, out_ch, out_positions]

        // Expand weight to [batch, out_ch, in_ch*k] for batched matmul
        let weight_expanded = weight
            .unsqueeze_dim::<3>(0)
            .expand([batch, out_ch, in_k]);

        // Batched matmul: [batch, out_ch, in_ch*k] @ [batch, in_ch*k, out_positions] = [batch, out_ch, out_positions]
        let out = weight_expanded.matmul(cols);

        // Add bias if present
        let out = if let Some(ref bias) = self.bias {
            let bias_expanded = bias.val().reshape([1, out_ch, 1]);
            out + bias_expanded
        } else {
            out
        };

        // Reshape to [batch, out_ch, out_t, out_h, out_w]
        out.reshape([batch, out_ch, out_t, out_h, out_w])
    }
}

/// Pad a 5D tensor [B, C, T, H, W] with zeros
fn pad_5d<B: Backend>(x: Tensor<B, 5>, padding: [usize; 3]) -> Tensor<B, 5> {
    let [batch, channels, time, height, width] = x.dims();
    let [p_t, p_h, p_w] = padding;
    let device = x.device();

    let new_t = time + 2 * p_t;
    let new_h = height + 2 * p_h;
    let new_w = width + 2 * p_w;

    // Create padded tensor
    let mut padded = Tensor::zeros([batch, channels, new_t, new_h, new_w], &device);

    // Copy original data to center using slice_assign
    padded = padded.slice_assign(
        [
            0..batch,
            0..channels,
            p_t..p_t + time,
            p_h..p_h + height,
            p_w..p_w + width,
        ],
        x,
    );

    padded
}

/// Extract 3D patches from input (im2col for 3D convolution)
///
/// Input: [batch, in_channels, T, H, W]
/// Output: [batch, in_channels * k_t * k_h * k_w, out_t * out_h * out_w]
fn im2col_3d<B: Backend>(
    x: Tensor<B, 5>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    out_size: [usize; 3],
) -> Tensor<B, 3> {
    let [batch, in_ch, _, _, _] = x.dims();
    let [k_t, k_h, k_w] = kernel_size;
    let [s_t, s_h, s_w] = stride;
    let [out_t, out_h, out_w] = out_size;

    let kernel_elements = k_t * k_h * k_w;
    let out_positions = out_t * out_h * out_w;
    let col_size = in_ch * kernel_elements;

    let device = x.device();

    // Collect all patches
    let mut patches = Vec::with_capacity(out_positions);

    for ot in 0..out_t {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let t_start = ot * s_t;
                let h_start = oh * s_h;
                let w_start = ow * s_w;

                // Extract patch [batch, in_ch, k_t, k_h, k_w]
                let patch = x.clone().slice([
                    0..batch,
                    0..in_ch,
                    t_start..t_start + k_t,
                    h_start..h_start + k_h,
                    w_start..w_start + k_w,
                ]);

                // Flatten to [batch, in_ch * k_t * k_h * k_w]
                let patch_flat = patch.reshape([batch, col_size]);
                // Add position dimension [batch, col_size, 1]
                let patch_col = patch_flat.unsqueeze_dim::<3>(2);
                patches.push(patch_col);
            }
        }
    }

    if patches.is_empty() {
        Tensor::zeros([batch, col_size, 0], &device)
    } else {
        // Concatenate along position dimension: [batch, col_size, out_positions]
        Tensor::cat(patches, 2)
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
    fn test_conv3d_params() {
        let params = Conv3dParams::new(3, 64)
            .with_kernel([3, 3, 3])
            .with_stride([1, 2, 2])
            .with_padding([1, 1, 1]);

        assert_eq!(params.in_channels, 3);
        assert_eq!(params.out_channels, 64);
        assert_eq!(params.stride, [1, 2, 2]);
    }

    #[test]
    fn test_conv3d_forward_shape() {
        let device = Default::default();

        // 3x3x3 conv with stride 1, padding 1 (same padding)
        let params = Conv3dParams::new(3, 16)
            .with_kernel([3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding([1, 1, 1]);
        let conv = Conv3d::<TestBackend>::new(&params, &device);

        let x = Tensor::<TestBackend, 5>::zeros([2, 3, 8, 16, 16], &device);
        let y = conv.forward(x);

        // Same padding: output shape should match input spatial dims
        assert_eq!(y.dims(), [2, 16, 8, 16, 16]);
    }

    #[test]
    fn test_conv3d_forward_stride() {
        let device = Default::default();

        // 3x3x3 conv with stride 2 in spatial dims
        let params = Conv3dParams::new(4, 8)
            .with_kernel([3, 3, 3])
            .with_stride([1, 2, 2])
            .with_padding([1, 1, 1]);
        let conv = Conv3d::<TestBackend>::new(&params, &device);

        let x = Tensor::<TestBackend, 5>::zeros([1, 4, 4, 16, 16], &device);
        let y = conv.forward(x);

        // Stride 2 in H,W halves spatial dimensions
        assert_eq!(y.dims(), [1, 8, 4, 8, 8]);
    }

    #[test]
    fn test_conv3d_computes_values() {
        let device = Default::default();

        let params = Conv3dParams::new(1, 1)
            .with_kernel([1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding([0, 0, 0]);
        let conv = Conv3d::<TestBackend>::new(&params, &device);

        // Input with known values
        let x = Tensor::<TestBackend, 5>::ones([1, 1, 2, 2, 2], &device);
        let y = conv.forward(x);

        // Output should not be all zeros (has bias and computed values)
        let y_data: Vec<f32> = y.into_data().to_vec().unwrap();
        let sum: f32 = y_data.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "Conv3d output should not be all zeros");
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
