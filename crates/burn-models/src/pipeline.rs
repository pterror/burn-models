//! Diffusion pipeline trait and implementations

// Many sampling loops use step_idx to index timestep_tensors AND pass to sampler.step()
#![allow(clippy::needless_range_loop)]

use burn::prelude::*;
use burn::tensor::Int;

use burn_models_clip::{ClipConfig, ClipTextEncoder, ClipTokenizer, END_OF_TEXT};
use burn_models_samplers::{DdimConfig, DdimSampler, NoiseSchedule, apply_guidance};
use burn_models_unet::{UNet, UNetConfig};
use burn_models_vae::{Decoder, DecoderConfig, Encoder, EncoderConfig};

/// Compute sinusoidal size embedding for SDXL
///
/// Used for encoding image dimensions (width, height, crop coordinates)
/// into the model's conditioning. Returns a 256-dim embedding.
fn compute_size_embedding<B: Backend>(value: usize, device: &B::Device) -> Tensor<B, 1> {
    let half_dim = 128;
    let value = value as f32;
    let mut emb = vec![0.0f32; 256];
    for i in 0..half_dim {
        let freq = (-((i as f32) / half_dim as f32) * (10000.0f32).ln()).exp();
        emb[i] = (value * freq).sin();
        emb[i + half_dim] = (value * freq).cos();
    }
    Tensor::from_data(TensorData::new(emb, [256]), device)
}

/// Configuration for sampling
#[derive(Debug, Clone)]
pub struct SampleConfig {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            steps: 50,
            guidance_scale: 7.5,
            seed: None,
        }
    }
}

// ============================================================================
// Step Callback Types
// ============================================================================

/// What to output at each sampling step
#[derive(Debug, Clone, Copy, Default)]
pub enum StepOutput {
    /// No output, minimal overhead
    #[default]
    None,
    /// Raw latent tensor (~0.1ms GPU→CPU copy)
    Latent,
    /// Latent visualized as RGB without VAE decode (~0.1ms)
    LatentPreview,
    /// Full VAE decode to image (~50-100ms, expensive!)
    Decoded,
}

/// Information passed to step callback
pub struct StepInfo<B: Backend> {
    /// Current step (0-indexed)
    pub step: usize,
    /// Total number of steps
    pub total_steps: usize,
    /// Current timestep value
    pub timestep: usize,
    /// Output based on StepOutput setting
    pub output: Option<Tensor<B, 4>>,
}

/// Latent format for preview coefficients
///
/// Coefficients from ComfyUI's latent_formats.py:
/// https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py
/// Licensed under GPL-3.0
#[derive(Debug, Clone, Copy, Default)]
pub enum LatentFormat {
    // 4-channel models
    /// Stable Diffusion 1.x
    #[default]
    SD15,
    /// Stable Diffusion XL
    SDXL,
    /// SDXL Playground 2.5
    SDXLPlayground25,
    /// SD 4x Upscaler
    SDX4,
    /// Stable Cascade stage B
    SCB,

    // 12-channel models
    /// Mochi video
    Mochi,

    // 16-channel models
    /// Stable Diffusion 3
    SD3,
    /// Flux.1
    Flux,
    /// Stable Cascade Prior
    SCPrior,
    /// HunyuanVideo
    HunyuanVideo,
    /// Wan 2.1
    Wan21,

    // 32-channel models
    /// HunyuanVideo 1.5
    HunyuanVideo15,
    /// Flux.2 (Kontext)
    Flux2,

    // 48-channel models
    /// Wan 2.2
    Wan22,

    // 128-channel models
    /// LTXV video
    LTXV,
}

impl LatentFormat {
    /// Number of latent channels for this format
    pub fn channels(&self) -> usize {
        match self {
            Self::SD15 | Self::SDXL | Self::SDXLPlayground25 | Self::SDX4 | Self::SCB => 4,
            Self::Mochi => 12,
            Self::SD3 | Self::Flux | Self::SCPrior | Self::HunyuanVideo | Self::Wan21 => 16,
            Self::HunyuanVideo15 | Self::Flux2 => 32,
            Self::Wan22 => 48,
            Self::LTXV => 128,
        }
    }

    /// Get RGB projection coefficients (channels x 3) and bias
    #[rustfmt::skip]
    fn rgb_coefs(&self) -> (Vec<f32>, [f32; 3]) {
        match self {
            // 4-channel models
            Self::SD15 => (vec![
                 0.3512,  0.2297,  0.3227,
                 0.3250,  0.4974,  0.2350,
                -0.2829,  0.1762,  0.2721,
                -0.2120, -0.2616, -0.7177,
            ], [0.0, 0.0, 0.0]),

            Self::SDXL => (vec![
                 0.3651,  0.4232,  0.4341,
                -0.2533, -0.0042,  0.1068,
                 0.1076,  0.1111, -0.0362,
                -0.3165, -0.2492, -0.2188,
            ], [0.1084, -0.0175, -0.0011]),

            Self::SDXLPlayground25 => (vec![
                 0.3920,  0.4054,  0.4549,
                -0.2634, -0.0196,  0.0653,
                 0.0568,  0.1687, -0.0755,
                -0.3112, -0.2359, -0.2076,
            ], [0.0, 0.0, 0.0]),

            Self::SDX4 => (vec![
                -0.2340, -0.3863, -0.3257,
                 0.0994,  0.0885, -0.0908,
                -0.2833, -0.2349, -0.3741,
                 0.2523, -0.0055, -0.1651,
            ], [0.0, 0.0, 0.0]),

            Self::SCB => (vec![
                 0.1121,  0.2006,  0.1023,
                -0.2093, -0.0222, -0.0195,
                -0.3087, -0.1535,  0.0366,
                 0.0290, -0.1574, -0.4078,
            ], [0.0, 0.0, 0.0]),

            // 12-channel models
            Self::Mochi => (vec![
                -0.0069, -0.0045,  0.0018,
                 0.0154, -0.0692, -0.0274,
                 0.0333,  0.0019,  0.0206,
                -0.1390,  0.0628,  0.1678,
                -0.0725,  0.0134, -0.1898,
                 0.0074, -0.0270, -0.0209,
                -0.0176, -0.0277, -0.0221,
                 0.5294,  0.5204,  0.3852,
                -0.0326, -0.0446, -0.0143,
                -0.0659,  0.0153, -0.0153,
                 0.0185, -0.0217,  0.0014,
                -0.0396, -0.0495, -0.0281,
            ], [-0.0940, -0.1418, -0.1453]),

            // 16-channel models
            Self::SD3 => (vec![
                -0.0645,  0.0177,  0.1052,
                 0.0028,  0.0312,  0.0650,
                 0.1848,  0.0762,  0.0360,
                 0.0944,  0.0360,  0.0889,
                 0.0897,  0.0506, -0.0364,
                -0.0020,  0.1203,  0.0284,
                 0.0855,  0.0118,  0.0283,
                -0.0539,  0.1160,  0.1077,
                -0.0057,  0.0116,  0.0700,
                -0.0412,  0.0281, -0.0039,
                 0.1106,  0.1171,  0.1220,
                -0.0248,  0.0682, -0.0481,
                 0.0815,  0.0846,  0.1207,
                -0.0120, -0.0055, -0.1463,
                 0.0020,  0.0523,  0.1994,
                 0.0339,  0.0254,  0.0459,
            ], [0.2394, 0.2135, 0.1925]),

            Self::Flux => (vec![
                -0.0346,  0.0244,  0.0681,
                 0.0034,  0.0210,  0.0687,
                 0.0437,  0.0689,  0.0471,
                 0.0440,  0.0967,  0.0728,
                 0.0386,  0.0445, -0.0601,
                 0.0444,  0.0921,  0.0335,
                 0.0986,  0.0331,  0.0743,
                -0.0279,  0.0553,  0.0452,
                -0.0060,  0.0298,  0.0636,
                 0.0349,  0.0784,  0.0276,
                 0.0732,  0.0735,  0.0148,
                 0.0091,  0.0420,  0.0073,
                 0.0719,  0.0669,  0.0757,
                 0.0270,  0.0658,  0.0031,
                -0.0156,  0.0353,  0.0604,
                 0.0472,  0.0316,  0.0701,
            ], [-0.0329, -0.0718, -0.0851]),

            Self::SCPrior => (vec![
                -0.0326, -0.0204, -0.0127,
                -0.1592, -0.0427,  0.0216,
                 0.0873,  0.0638, -0.0020,
                -0.0602,  0.0442,  0.1304,
                 0.0800, -0.0313, -0.1796,
                -0.0810, -0.0638, -0.1581,
                 0.1791,  0.1180,  0.0967,
                 0.0740,  0.1416,  0.0432,
                -0.1745, -0.1888, -0.1373,
                 0.2412,  0.1577,  0.0928,
                 0.1908,  0.0998,  0.0682,
                 0.0209,  0.0365, -0.0092,
                 0.0448, -0.0650, -0.1728,
                -0.1658, -0.1045, -0.1308,
                 0.0542,  0.1545,  0.1325,
                -0.0352, -0.1672, -0.2541,
            ], [0.0, 0.0, 0.0]),

            Self::HunyuanVideo => (vec![
                -0.0395, -0.0331,  0.0445,
                 0.0696,  0.0795,  0.0518,
                 0.0135, -0.0945, -0.0282,
                 0.0108, -0.0250, -0.0765,
                -0.0209,  0.0032,  0.0224,
                -0.0804, -0.0254, -0.0639,
                -0.0991,  0.0271, -0.0669,
                -0.0646, -0.0422, -0.0400,
                -0.0696, -0.0595, -0.0894,
                -0.0799, -0.0208, -0.0375,
                 0.1166,  0.1627,  0.0962,
                 0.1165,  0.0432,  0.0407,
                -0.2315, -0.1920, -0.1355,
                -0.0270,  0.0401, -0.0821,
                -0.0616, -0.0997, -0.0727,
                 0.0249, -0.0469, -0.1703,
            ], [0.0259, -0.0192, -0.0761]),

            Self::Wan21 => (vec![
                -0.1299, -0.1692,  0.2932,
                 0.0671,  0.0406,  0.0442,
                 0.3568,  0.2548,  0.1747,
                 0.0372,  0.2344,  0.1420,
                 0.0313,  0.0189, -0.0328,
                 0.0296, -0.0956, -0.0665,
                -0.3477, -0.4059, -0.2925,
                 0.0166,  0.1902,  0.1975,
                -0.0412,  0.0267, -0.1364,
                -0.1293,  0.0740,  0.1636,
                 0.0680,  0.3019,  0.1128,
                 0.0032,  0.0581,  0.0639,
                -0.1251,  0.0927,  0.1699,
                 0.0060, -0.0633,  0.0005,
                 0.3477,  0.2275,  0.2950,
                 0.1984,  0.0913,  0.1861,
            ], [-0.1835, -0.0868, -0.3360]),

            // 32-channel models
            Self::HunyuanVideo15 => (vec![
                 0.0568, -0.0521, -0.0131,
                 0.0014,  0.0735,  0.0326,
                 0.0186,  0.0531, -0.0138,
                -0.0031,  0.0051,  0.0288,
                 0.0110,  0.0556,  0.0432,
                -0.0041, -0.0023, -0.0485,
                 0.0530,  0.0413,  0.0253,
                 0.0283,  0.0251,  0.0339,
                 0.0277, -0.0372, -0.0093,
                 0.0393,  0.0944,  0.1131,
                 0.0020,  0.0251,  0.0037,
                -0.0017,  0.0012,  0.0234,
                 0.0468,  0.0436,  0.0203,
                 0.0354,  0.0439, -0.0233,
                 0.0090,  0.0123,  0.0346,
                 0.0382,  0.0029,  0.0217,
                 0.0261, -0.0300,  0.0030,
                -0.0088, -0.0220, -0.0283,
                -0.0272, -0.0121, -0.0363,
                -0.0664, -0.0622,  0.0144,
                 0.0414,  0.0479,  0.0529,
                 0.0355,  0.0612, -0.0247,
                 0.0147,  0.0264,  0.0174,
                 0.0438,  0.0038,  0.0542,
                 0.0431, -0.0573, -0.0033,
                -0.0162, -0.0211, -0.0406,
                -0.0487, -0.0295, -0.0393,
                 0.0005, -0.0109,  0.0253,
                 0.0296,  0.0591,  0.0353,
                 0.0119,  0.0181, -0.0306,
                -0.0085, -0.0362,  0.0229,
                 0.0005, -0.0106,  0.0242,
            ], [0.0456, -0.0202, -0.0644]),

            Self::Flux2 => (vec![
                 0.0058,  0.0113,  0.0073,
                 0.0495,  0.0443,  0.0836,
                -0.0099,  0.0096,  0.0644,
                 0.2144,  0.3009,  0.3652,
                 0.0166, -0.0039, -0.0054,
                 0.0157,  0.0103, -0.0160,
                -0.0398,  0.0902, -0.0235,
                -0.0052,  0.0095,  0.0109,
                -0.3527, -0.2712, -0.1666,
                -0.0301, -0.0356, -0.0180,
                -0.0107,  0.0078,  0.0013,
                 0.0746,  0.0090, -0.0941,
                 0.0156,  0.0169,  0.0070,
                -0.0034, -0.0040, -0.0114,
                 0.0032,  0.0181,  0.0080,
                -0.0939, -0.0008,  0.0186,
                 0.0018,  0.0043,  0.0104,
                 0.0284,  0.0056, -0.0127,
                -0.0024, -0.0022, -0.0030,
                 0.1207, -0.0026,  0.0065,
                 0.0128,  0.0101,  0.0142,
                 0.0137, -0.0072, -0.0007,
                 0.0095,  0.0092, -0.0059,
                 0.0000, -0.0077, -0.0049,
                -0.0465, -0.0204, -0.0312,
                 0.0095,  0.0012, -0.0066,
                 0.0290, -0.0034,  0.0025,
                 0.0220,  0.0169, -0.0048,
                -0.0332, -0.0457, -0.0468,
                -0.0085,  0.0389,  0.0609,
                -0.0076,  0.0003, -0.0043,
                -0.0111, -0.0460, -0.0614,
            ], [-0.0329, -0.0718, -0.0851]),

            // 48-channel models
            Self::Wan22 => (vec![
                 0.0119,  0.0103,  0.0046,
                -0.1062, -0.0504,  0.0165,
                 0.0140,  0.0409,  0.0491,
                -0.0813, -0.0677,  0.0607,
                 0.0656,  0.0851,  0.0808,
                 0.0264,  0.0463,  0.0912,
                 0.0295,  0.0326,  0.0590,
                -0.0244, -0.0270,  0.0025,
                 0.0443, -0.0102,  0.0288,
                -0.0465, -0.0090, -0.0205,
                 0.0359,  0.0236,  0.0082,
                -0.0776,  0.0854,  0.1048,
                 0.0564,  0.0264,  0.0561,
                 0.0006,  0.0594,  0.0418,
                -0.0319, -0.0542, -0.0637,
                -0.0268,  0.0024,  0.0260,
                 0.0539,  0.0265,  0.0358,
                -0.0359, -0.0312, -0.0287,
                -0.0285, -0.1032, -0.1237,
                 0.1041,  0.0537,  0.0622,
                -0.0086, -0.0374, -0.0051,
                 0.0390,  0.0670,  0.2863,
                 0.0069,  0.0144,  0.0082,
                 0.0006, -0.0167,  0.0079,
                 0.0313, -0.0574, -0.0232,
                -0.1454, -0.0902, -0.0481,
                 0.0714,  0.0827,  0.0447,
                -0.0304, -0.0574, -0.0196,
                 0.0401,  0.0384,  0.0204,
                -0.0758, -0.0297, -0.0014,
                 0.0568,  0.1307,  0.1372,
                -0.0055, -0.0310, -0.0380,
                 0.0239, -0.0305,  0.0325,
                -0.0663, -0.0673, -0.0140,
                -0.0416, -0.0047, -0.0023,
                 0.0166,  0.0112, -0.0093,
                -0.0211,  0.0011,  0.0331,
                 0.1833,  0.1466,  0.2250,
                -0.0368,  0.0370,  0.0295,
                -0.3441, -0.3543, -0.2008,
                -0.0479, -0.0489, -0.0420,
                -0.0660, -0.0153,  0.0800,
                -0.0101,  0.0068,  0.0156,
                -0.0690, -0.0452, -0.0927,
                -0.0145,  0.0041,  0.0015,
                 0.0421,  0.0451,  0.0373,
                 0.0504, -0.0483, -0.0356,
                -0.0837,  0.0168,  0.0055,
            ], [0.0317, -0.0878, -0.1388]),

            // 128-channel models
            Self::LTXV => (vec![
                 0.0112, -0.0006, -0.0100,
                 0.0860,  0.0658,  0.0010,
                -0.0126, -0.0076, -0.0041,
                 0.0094, -0.0022,  0.0026,
                 0.0038,  0.0128,  0.0092,
                 0.0210, -0.0053,  0.0034,
                -0.0089, -0.0197, -0.0188,
                -0.0132, -0.0105,  0.0020,
                -0.0015, -0.0070, -0.0076,
                -0.0017,  0.0005, -0.0034,
                 0.0136,  0.0047, -0.0020,
                 0.0103,  0.0077,  0.0139,
                -0.0161, -0.0062,  0.0012,
                 0.0073,  0.0156,  0.0004,
                 0.0010, -0.0030, -0.0148,
                 0.0191,  0.0109,  0.0123,
                 0.0045,  0.0000, -0.0069,
                -0.0005,  0.0033,  0.0078,
                 0.0339,  0.0334,  0.0375,
                -0.0230, -0.0025, -0.0031,
                 0.0503,  0.0388,  0.0335,
                -0.0041, -0.0011,  0.0016,
                -0.1269, -0.1311, -0.2101,
                 0.0263,  0.0142, -0.0036,
                -0.0049,  0.0088,  0.0078,
                -0.0017, -0.0049, -0.0052,
                -0.0021,  0.0024,  0.0094,
                -0.0225, -0.0213, -0.0151,
                -0.0158, -0.0106, -0.0065,
                -0.0047,  0.0050, -0.0067,
                 0.0120,  0.0207,  0.0162,
                -0.0064, -0.0085, -0.0095,
                 0.0073, -0.0099, -0.0230,
                -0.0009,  0.0063,  0.0096,
                -0.0372, -0.0371, -0.0567,
                -0.1337, -0.1072, -0.0538,
                -0.0054,  0.0081,  0.0088,
                -0.1525, -0.2144, -0.2184,
                 0.0314,  0.0070, -0.0098,
                 0.0022, -0.0090, -0.0210,
                 0.0038, -0.0059, -0.0150,
                -0.0043, -0.0129, -0.0160,
                -0.0055, -0.0108, -0.0030,
                -0.0065,  0.0031, -0.0102,
                -0.0050, -0.0072, -0.0009,
                -0.0086, -0.0024,  0.0011,
                -0.0090, -0.0096,  0.0016,
                 0.0051,  0.0121,  0.0200,
                 0.0138,  0.0117,  0.0082,
                -0.0105, -0.0116, -0.0041,
                -0.0284, -0.0313, -0.0221,
                 0.0029,  0.0365,  0.0187,
                -0.0167, -0.0167, -0.0045,
                 0.0488,  0.0401,  0.0087,
                -0.0151, -0.0006,  0.0030,
                -0.0176, -0.0081,  0.0131,
                -0.0093,  0.0108, -0.0063,
                 0.0031,  0.0005,  0.0123,
                -0.0228, -0.0230, -0.0260,
                -0.0248, -0.0154, -0.0221,
                -0.0236,  0.0011,  0.0124,
                -0.0079, -0.0012, -0.0061,
                -0.0115, -0.0013,  0.0063,
                -0.0542,  0.0266,  0.0063,
                 0.0044, -0.0073, -0.0105,
                -0.0045,  0.0016,  0.0144,
                 0.0137,  0.0089,  0.0041,
                -0.0101,  0.0090,  0.0157,
                -0.0056,  0.0012,  0.0081,
                -0.0037, -0.0054,  0.0013,
                 0.0295,  0.0214,  0.0304,
                -0.0349, -0.0243, -0.0253,
                -0.0341, -0.0224, -0.0106,
                -0.0173, -0.0132, -0.0107,
                -0.0021, -0.0086, -0.0030,
                 0.0012, -0.0042, -0.0069,
                 0.0009, -0.0067, -0.0001,
                 0.0160, -0.0101, -0.0289,
                 0.0012,  0.0102,  0.0189,
                 0.0173,  0.0003,  0.0138,
                -0.0135, -0.0036,  0.0007,
                 0.0047, -0.0052,  0.0024,
                -0.0059, -0.0062, -0.0018,
                 0.0155,  0.0146,  0.0020,
                 0.0075,  0.0016, -0.0082,
                 0.0191,  0.0016, -0.0040,
                -0.0057, -0.0027, -0.0041,
                 0.0017,  0.0146,  0.0258,
                -0.0008,  0.0023,  0.0045,
                 0.0116,  0.0089, -0.0073,
                 0.0076,  0.0027,  0.0114,
                 0.0052,  0.0037,  0.0140,
                -0.0184, -0.0225, -0.0245,
                 0.0006, -0.0058, -0.0148,
                -0.0161, -0.0086, -0.0145,
                 0.0205,  0.0207,  0.0064,
                 0.0034, -0.0112, -0.0164,
                -0.0015, -0.0105,  0.0017,
                 0.0281,  0.0235,  0.0328,
                -0.0185, -0.0128, -0.0088,
                -0.0081, -0.0108, -0.0175,
                -0.0039,  0.0162,  0.0334,
                -0.0075, -0.0142, -0.0062,
                 0.0035, -0.0114, -0.0106,
                 0.0115,  0.0039,  0.0028,
                 0.0072, -0.0015, -0.0038,
                 0.0022, -0.0088, -0.0096,
                 0.0241,  0.0217,  0.0281,
                -0.0054, -0.0243, -0.0178,
                 0.0074,  0.0105,  0.0127,
                 0.0063,  0.0063,  0.0192,
                 0.0164,  0.0095,  0.0067,
                 0.0172,  0.0236,  0.0233,
                -0.0146, -0.0098, -0.0116,
                 0.0144,  0.0144,  0.0066,
                -0.0068,  0.0189,  0.0146,
                 0.0061,  0.0035, -0.0027,
                -0.0027, -0.0059, -0.0092,
                 0.0102,  0.0074, -0.0076,
                -0.0133,  0.0193, -0.0009,
                 0.0024, -0.0048, -0.0158,
                 0.0262,  0.0260,  0.0202,
                 0.0157,  0.0185,  0.0027,
                -0.0022,  0.0047, -0.0224,
                -0.0075,  0.0074,  0.0144,
                -0.0084, -0.0080,  0.0098,
                 0.0383,  0.0097, -0.0193,
                -0.0146, -0.0067,  0.0040,
            ], [-0.0571, -0.1657, -0.2512]),
        }
    }

    /// Get format that matches a given channel count (for auto-detection)
    pub fn for_channels(channels: usize) -> Self {
        match channels {
            4 => Self::SD15,
            12 => Self::Mochi,
            16 => Self::SD3,
            32 => Self::HunyuanVideo15,
            48 => Self::Wan22,
            128 => Self::LTXV,
            _ => Self::SD15, // fallback
        }
    }
}

/// Convert latent to RGB preview without VAE decode
///
/// Uses learned coefficients to project latent channels to RGB.
/// Much faster than VAE decode (~0.1ms vs ~50ms).
///
/// Coefficients from ComfyUI's latent_formats.py (GPL-3.0):
/// https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py
pub fn latent_to_preview<B: Backend>(latent: Tensor<B, 4>, format: LatentFormat) -> Tensor<B, 4> {
    let [b, c, h, w] = latent.dims();
    let device = latent.device();

    // Use format's coefficients, or auto-detect based on channel count
    let effective_format = if format.channels() == c {
        format
    } else {
        LatentFormat::for_channels(c)
    };

    let (coefs_vec, bias) = effective_format.rgb_coefs();
    let channels = effective_format.channels();

    let coefs: Tensor<B, 2> = Tensor::from_data(TensorData::new(coefs_vec, [channels, 3]), &device);

    // einsum "cxy,cr -> rxy": project latent channels to 3 RGB channels
    let flat = latent.reshape([b, channels, h * w]); // [B, C, H*W]
    let flat_t = flat.swap_dims(1, 2); // [B, H*W, C]
    let coefs_3d = coefs.unsqueeze::<3>().repeat_dim(0, b); // [B, C, 3]
    let rgb_flat = flat_t.matmul(coefs_3d); // [B, H*W, 3]
    let rgb = rgb_flat.swap_dims(1, 2).reshape([b, 3, h, w]); // [B, 3, H, W]

    // Apply bias and normalize to [0, 255]
    let bias_tensor: Tensor<B, 1> = Tensor::from_data(TensorData::new(bias.to_vec(), [3]), &device);
    let rgb_biased = rgb + bias_tensor.reshape([1, 3, 1, 1]);
    let normalized = (rgb_biased + 0.5) * 255.0;
    normalized.clamp(0.0, 255.0)
}

/// Unified interface for diffusion pipelines
pub trait DiffusionPipeline<B: Backend> {
    type Conditioning;

    /// Encode text prompt into conditioning
    fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> Self::Conditioning;

    /// Sample latent from conditioning
    fn sample_latent(
        &self,
        conditioning: &Self::Conditioning,
        config: &SampleConfig,
    ) -> Tensor<B, 4>;

    /// Decode latent to image
    fn decode(&self, latent: Tensor<B, 4>) -> Tensor<B, 4>;

    /// Full pipeline: prompt -> image
    fn generate(&self, prompt: &str, negative_prompt: &str, config: &SampleConfig) -> Tensor<B, 4> {
        let conditioning = self.encode_prompt(prompt, negative_prompt);
        let latent = self.sample_latent(&conditioning, config);
        self.decode(latent)
    }
}

/// SD 1.x conditioning (text embeddings)
pub struct Sd1xConditioning<B: Backend> {
    /// Conditional text embedding [batch, seq_len, embed_dim]
    pub cond: Tensor<B, 3>,
    /// Unconditional text embedding [batch, seq_len, embed_dim]
    pub uncond: Tensor<B, 3>,
}

/// Stable Diffusion 1.x Pipeline
pub struct StableDiffusion1x<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub text_encoder: ClipTextEncoder<B>,
    pub unet: UNet<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    pub device: B::Device,
}

impl<B: Backend> StableDiffusion1x<B> {
    /// Create a new SD 1.x pipeline with default configs
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let clip_config = ClipConfig::sd1x();
        let unet_config = UNetConfig::sd1x();
        let vae_config = DecoderConfig::sd();

        Self {
            tokenizer,
            text_encoder: ClipTextEncoder::new(&clip_config, device),
            unet: UNet::new(&unet_config, device),
            vae_decoder: Decoder::new(&vae_config, device),
            scheduler: NoiseSchedule::sd1x(device),
            device: device.clone(),
        }
    }

    /// Create with custom configs
    pub fn with_configs(
        tokenizer: ClipTokenizer,
        clip_config: &ClipConfig,
        unet_config: &UNetConfig,
        vae_config: &DecoderConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            tokenizer,
            text_encoder: ClipTextEncoder::new(clip_config, device),
            unet: UNet::new(unet_config, device),
            vae_decoder: Decoder::new(vae_config, device),
            scheduler: NoiseSchedule::sd1x(device),
            device: device.clone(),
        }
    }

    /// Encode a single text prompt to CLIP embeddings
    fn encode_text(&self, text: &str) -> Tensor<B, 3> {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>(); // [1, 77]

        self.text_encoder.forward(token_tensor)
    }

    /// Find the position of the end-of-sequence token in the token list
    #[allow(dead_code)]
    fn find_eos_position(tokens: &[u32]) -> usize {
        tokens
            .iter()
            .position(|&t| t == END_OF_TEXT)
            .unwrap_or(tokens.len() - 1)
    }
}

impl<B: Backend> DiffusionPipeline<B> for StableDiffusion1x<B> {
    type Conditioning = Sd1xConditioning<B>;

    fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> Self::Conditioning {
        let cond = self.encode_text(prompt);
        let uncond = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        Sd1xConditioning { cond, uncond }
    }

    fn sample_latent(
        &self,
        conditioning: &Self::Conditioning,
        config: &SampleConfig,
    ) -> Tensor<B, 4> {
        let latent_height = config.height / 8;
        let latent_width = config.width / 8;

        // Create DDIM sampler
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sd1x(&self.device), ddim_config);

        // Initialize with random noise
        let mut latent = sampler.init_latent(1, 4, latent_height, latent_width, &self.device);

        // Precompute all timestep tensors to avoid CPU->GPU transfer in hot loop
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Sampling loop
        for step_idx in 0..sampler.num_steps() {
            let t = timestep_tensors[step_idx].clone();

            // Predict noise for unconditional
            let noise_uncond =
                self.unet
                    .forward(latent.clone(), t.clone(), conditioning.uncond.clone());

            // Predict noise for conditional
            let noise_cond = self
                .unet
                .forward(latent.clone(), t, conditioning.cond.clone());

            // Apply classifier-free guidance
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            // DDIM step
            latent = sampler.step(latent, noise_pred, step_idx);
        }

        latent
    }

    fn decode(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
        self.vae_decoder.decode_to_image(latent)
    }
}

impl<B: Backend> StableDiffusion1x<B> {
    /// Sample latent with step callback for progress reporting
    ///
    /// The callback is called after each step with step info and optional output.
    /// Use `StepOutput::None` for minimal overhead, or `StepOutput::LatentPreview`
    /// for cheap visual feedback.
    pub fn sample_latent_with_callback<F>(
        &self,
        conditioning: &Sd1xConditioning<B>,
        config: &SampleConfig,
        step_output: StepOutput,
        mut callback: F,
    ) -> Tensor<B, 4>
    where
        F: FnMut(StepInfo<B>),
    {
        let latent_height = config.height / 8;
        let latent_width = config.width / 8;

        // Create DDIM sampler
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sd1x(&self.device), ddim_config);

        // Initialize with random noise
        let mut latent = sampler.init_latent(1, 4, latent_height, latent_width, &self.device);

        // Precompute all timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        let timesteps = sampler.timesteps();
        let total_steps = sampler.num_steps();

        // Sampling loop
        for step_idx in 0..total_steps {
            let t = timestep_tensors[step_idx].clone();

            // Predict noise for unconditional
            let noise_uncond =
                self.unet
                    .forward(latent.clone(), t.clone(), conditioning.uncond.clone());

            // Predict noise for conditional
            let noise_cond = self
                .unet
                .forward(latent.clone(), t, conditioning.cond.clone());

            // Apply classifier-free guidance
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            // DDIM step
            latent = sampler.step(latent, noise_pred, step_idx);

            // Generate output based on step_output setting
            let output = match step_output {
                StepOutput::None => None,
                StepOutput::Latent => Some(latent.clone()),
                StepOutput::LatentPreview => {
                    Some(latent_to_preview(latent.clone(), LatentFormat::SD15))
                }
                StepOutput::Decoded => Some(self.vae_decoder.decode_to_image(latent.clone())),
            };

            // Call the callback
            callback(StepInfo {
                step: step_idx,
                total_steps,
                timestep: timesteps[step_idx],
                output,
            });
        }

        latent
    }

    /// Generate image with step callback
    pub fn generate_with_callback<F>(
        &self,
        prompt: &str,
        negative_prompt: &str,
        config: &SampleConfig,
        step_output: StepOutput,
        callback: F,
    ) -> Tensor<B, 4>
    where
        F: FnMut(StepInfo<B>),
    {
        let conditioning =
            <Self as DiffusionPipeline<B>>::encode_prompt(self, prompt, negative_prompt);
        let latent = self.sample_latent_with_callback(&conditioning, config, step_output, callback);
        self.vae_decoder.decode_to_image(latent)
    }
}

/// Configuration for img2img sampling
#[derive(Debug, Clone)]
pub struct Img2ImgConfig {
    pub steps: usize,
    pub guidance_scale: f64,
    /// Strength of the transformation (0.0 = no change, 1.0 = full regeneration)
    pub strength: f64,
    pub seed: Option<u64>,
}

impl Default for Img2ImgConfig {
    fn default() -> Self {
        Self {
            steps: 50,
            guidance_scale: 7.5,
            strength: 0.75,
            seed: None,
        }
    }
}

/// Stable Diffusion 1.x Img2Img Pipeline
pub struct StableDiffusion1xImg2Img<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub text_encoder: ClipTextEncoder<B>,
    pub unet: UNet<B>,
    pub vae_encoder: Encoder<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusion1xImg2Img<B> {
    /// Create a new SD 1.x img2img pipeline with default configs
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let clip_config = ClipConfig::sd1x();
        let unet_config = UNetConfig::sd1x();
        let encoder_config = EncoderConfig::sd();
        let decoder_config = DecoderConfig::sd();

        Self {
            tokenizer,
            text_encoder: ClipTextEncoder::new(&clip_config, device),
            unet: UNet::new(&unet_config, device),
            vae_encoder: Encoder::new(&encoder_config, device),
            vae_decoder: Decoder::new(&decoder_config, device),
            scheduler: NoiseSchedule::sd1x(device),
            device: device.clone(),
        }
    }

    /// Encode a single text prompt
    fn encode_text(&self, text: &str) -> Tensor<B, 3> {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>();
        self.text_encoder.forward(token_tensor)
    }

    /// Generate image from input image and prompt
    ///
    /// # Arguments
    /// * `image` - Input image tensor [1, 3, H, W] with values in [0, 255]
    /// * `prompt` - Text prompt
    /// * `negative_prompt` - Negative prompt
    /// * `config` - Img2img configuration
    pub fn generate(
        &self,
        image: Tensor<B, 4>,
        prompt: &str,
        negative_prompt: &str,
        config: &Img2ImgConfig,
    ) -> Tensor<B, 4> {
        // Encode prompts
        let cond = self.encode_text(prompt);
        let uncond = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        // Encode image to latent
        let init_latent = self.vae_encoder.encode_deterministic(image);

        // Calculate start step based on strength
        let num_inference_steps = config.steps;
        let start_step = ((1.0 - config.strength) * num_inference_steps as f64) as usize;

        // Create sampler
        let ddim_config = DdimConfig {
            num_inference_steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sd1x(&self.device), ddim_config);

        // Add noise to init_latent at the start timestep
        let start_timestep = if start_step < sampler.timesteps().len() {
            sampler.timesteps()[start_step]
        } else {
            0
        };

        let noise = Tensor::random(
            init_latent.shape(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &self.device,
        );

        // Get alpha for noise addition
        let alpha_t = self.scheduler.alpha_cumprod_at(start_timestep);
        let sqrt_alpha = alpha_t.clone().sqrt();
        let sqrt_one_minus_alpha = (alpha_t.neg() + 1.0).sqrt();

        let mut latent =
            init_latent * sqrt_alpha.unsqueeze() + noise * sqrt_one_minus_alpha.unsqueeze();

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Denoising loop from start_step
        for step_idx in start_step..sampler.num_steps() {
            let t = timestep_tensors[step_idx].clone();

            let noise_uncond = self.unet.forward(latent.clone(), t.clone(), uncond.clone());
            let noise_cond = self.unet.forward(latent.clone(), t, cond.clone());
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            latent = sampler.step(latent, noise_pred, step_idx);
        }

        // Decode
        self.vae_decoder.decode_to_image(latent)
    }
}

// ============================================================================
// SD 1.x Inpainting Pipeline
// ============================================================================

/// Inpainting configuration
#[derive(Debug, Clone)]
pub struct InpaintConfig {
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
}

impl Default for InpaintConfig {
    fn default() -> Self {
        Self {
            steps: 50,
            guidance_scale: 7.5,
            seed: None,
        }
    }
}

/// Stable Diffusion 1.x Inpainting Pipeline
///
/// Performs masked image editing - regenerates only the masked regions
/// while preserving unmasked areas.
pub struct StableDiffusion1xInpaint<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub text_encoder: ClipTextEncoder<B>,
    pub unet: UNet<B>,
    pub vae_encoder: Encoder<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusion1xInpaint<B> {
    /// Create a new SD 1.x inpainting pipeline
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let clip_config = ClipConfig::sd1x();
        let unet_config = UNetConfig::sd1x();
        let encoder_config = EncoderConfig::sd();
        let decoder_config = DecoderConfig::sd();

        Self {
            tokenizer,
            text_encoder: ClipTextEncoder::new(&clip_config, device),
            unet: UNet::new(&unet_config, device),
            vae_encoder: Encoder::new(&encoder_config, device),
            vae_decoder: Decoder::new(&decoder_config, device),
            scheduler: NoiseSchedule::sd1x(device),
            device: device.clone(),
        }
    }

    /// Encode a single text prompt to embeddings
    fn encode_text(&self, text: &str) -> Tensor<B, 3> {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>();
        self.text_encoder.forward(token_tensor)
    }

    /// Inpaint masked regions of an image
    ///
    /// # Arguments
    /// * `image` - Input image tensor [1, 3, H, W] with values in [0, 255]
    /// * `mask` - Binary mask tensor [1, 1, H, W] where 1 = regenerate, 0 = preserve
    /// * `prompt` - Text prompt for regenerated regions
    /// * `negative_prompt` - Negative prompt
    /// * `config` - Inpainting configuration
    pub fn inpaint(
        &self,
        image: Tensor<B, 4>,
        mask: Tensor<B, 4>,
        prompt: &str,
        negative_prompt: &str,
        config: &InpaintConfig,
    ) -> Tensor<B, 4> {
        let [_, _, img_h, img_w] = image.dims();

        // Encode prompts
        let cond = self.encode_text(prompt);
        let uncond = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        // Encode image to latent
        let init_latent = self.vae_encoder.encode_deterministic(image);

        // Downsample mask to latent size
        let latent_mask = self.downsample_mask(mask, img_h / 8, img_w / 8);

        // Create sampler
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let scheduler = NoiseSchedule::sd1x(&self.device);
        let sampler = DdimSampler::new(scheduler, ddim_config);

        // Initialize latent with noise
        let [_, c, h, w] = init_latent.dims();
        let mut latent = sampler.init_latent(1, c, h, w, &self.device);

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Inpainting loop
        for step_idx in 0..sampler.num_steps() {
            let timestep = sampler.timesteps()[step_idx];
            let t = timestep_tensors[step_idx].clone();

            // Predict noise
            let noise_uncond = self.unet.forward(latent.clone(), t.clone(), uncond.clone());
            let noise_cond = self.unet.forward(latent.clone(), t.clone(), cond.clone());
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            // DDIM step for generated regions
            latent = sampler.step(latent, noise_pred, step_idx);

            // Blend: replace unmasked regions with noised original
            let alpha_t = self.scheduler.alpha_cumprod_at(timestep);
            let sqrt_alpha = alpha_t.clone().sqrt();
            let sqrt_one_minus_alpha = (alpha_t.neg() + 1.0).sqrt();

            let noise = Tensor::random(
                init_latent.shape(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &self.device,
            );
            let noised_original = init_latent.clone() * sqrt_alpha.unsqueeze()
                + noise * sqrt_one_minus_alpha.unsqueeze();

            // mask = 1 means regenerate (use latent), mask = 0 means preserve (use noised_original)
            latent = latent.clone() * latent_mask.clone()
                + noised_original * (latent_mask.clone().neg() + 1.0);
        }

        // Final blend in latent space (without noise)
        latent = latent.clone() * latent_mask.clone() + init_latent * (latent_mask.neg() + 1.0);

        // Decode
        self.vae_decoder.decode_to_image(latent)
    }

    /// Downsample mask from image space to latent space using nearest-neighbor sampling
    fn downsample_mask(
        &self,
        mask: Tensor<B, 4>,
        target_h: usize,
        target_w: usize,
    ) -> Tensor<B, 4> {
        let [b, c, h, w] = mask.dims();

        // Simple nearest-neighbor downsampling via slicing
        // Take every 8th pixel
        let scale_h = h / target_h;
        let scale_w = w / target_w;

        let mut result = Vec::with_capacity(b * c * target_h * target_w);
        let data = mask.into_data();
        let values: Vec<f32> = data.to_vec().unwrap();

        for batch in 0..b {
            for channel in 0..c {
                for th in 0..target_h {
                    for tw in 0..target_w {
                        let src_h = th * scale_h;
                        let src_w = tw * scale_w;
                        let idx = batch * c * h * w + channel * h * w + src_h * w + src_w;
                        result.push(values[idx]);
                    }
                }
            }
        }

        Tensor::from_data(
            TensorData::new(result, [b, c, target_h, target_w]),
            &self.device,
        )
    }
}

// ============================================================================
// SDXL Pipeline
// ============================================================================

use burn_models_clip::{OpenClipConfig, OpenClipTextEncoder};
use burn_models_unet::{UNetXL, UNetXLConfig};

/// SDXL sampling configuration
#[derive(Debug, Clone)]
pub struct SdxlSampleConfig {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
}

impl Default for SdxlSampleConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            steps: 30,
            guidance_scale: 7.5,
            seed: None,
        }
    }
}

/// SDXL conditioning (dual text embeddings + pooled)
pub struct SdxlConditioning<B: Backend> {
    /// Conditional context [batch, seq_len, 2048]
    pub cond_context: Tensor<B, 3>,
    /// Unconditional context [batch, seq_len, 2048]
    pub uncond_context: Tensor<B, 3>,
    /// Conditional pooled embedding [batch, pooled_dim]
    pub cond_pooled: Tensor<B, 2>,
    /// Unconditional pooled embedding [batch, pooled_dim]
    pub uncond_pooled: Tensor<B, 2>,
}

/// Stable Diffusion XL Pipeline
///
/// Uses dual text encoders (CLIP + OpenCLIP) for conditioning
pub struct StableDiffusionXL<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub clip_encoder: ClipTextEncoder<B>,
    pub open_clip_encoder: OpenClipTextEncoder<B>,
    pub unet: UNetXL<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusionXL<B> {
    /// Create a new SDXL pipeline with default configs
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let clip_config = ClipConfig::sd1x(); // SDXL uses same CLIP architecture
        let open_clip_config = OpenClipConfig::sdxl();
        let unet_config = UNetXLConfig::sdxl_base();
        let vae_config = DecoderConfig::sd();

        Self {
            tokenizer,
            clip_encoder: ClipTextEncoder::new(&clip_config, device),
            open_clip_encoder: OpenClipTextEncoder::new(&open_clip_config, device),
            unet: UNetXL::new(&unet_config, device),
            vae_decoder: Decoder::new(&vae_config, device),
            scheduler: NoiseSchedule::sdxl(device),
            device: device.clone(),
        }
    }

    /// Create with custom configs
    pub fn with_configs(
        tokenizer: ClipTokenizer,
        clip_config: &ClipConfig,
        open_clip_config: &OpenClipConfig,
        unet_config: &UNetXLConfig,
        vae_config: &DecoderConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            tokenizer,
            clip_encoder: ClipTextEncoder::new(clip_config, device),
            open_clip_encoder: OpenClipTextEncoder::new(open_clip_config, device),
            unet: UNetXL::new(unet_config, device),
            vae_decoder: Decoder::new(vae_config, device),
            scheduler: NoiseSchedule::sdxl(device),
            device: device.clone(),
        }
    }

    /// Encode text using both CLIP and OpenCLIP encoders, returning concatenated context and pooled embedding
    fn encode_text(&self, text: &str) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>(); // [1, 77]

        // CLIP encoder output [1, 77, 768]
        let clip_hidden = self.clip_encoder.forward(token_tensor.clone());

        // OpenCLIP encoder outputs [1, 77, 1280] and pooled [1, 1280]
        let eos_pos = tokens.iter().position(|&t| t == END_OF_TEXT).unwrap_or(76);
        let (open_clip_hidden, pooled) = self
            .open_clip_encoder
            .forward_with_pooled(token_tensor, &[eos_pos]);

        // Concatenate CLIP and OpenCLIP hidden states [1, 77, 768 + 1280 = 2048]
        let context = Tensor::cat(vec![clip_hidden, open_clip_hidden], 2);

        (context, pooled)
    }

    /// Create add_embed from pooled text embedding
    ///
    /// SDXL add_embed includes:
    /// - Pooled text embedding (1280)
    /// - Original size (height, width) - 256 each
    /// - Crop coords (top, left) - 256 each
    /// - Target size (height, width) - 256 each
    pub fn create_add_embed(
        &self,
        pooled: Tensor<B, 2>,
        original_size: (usize, usize),
        crop_coords: (usize, usize),
        target_size: (usize, usize),
    ) -> Tensor<B, 2> {
        let [_batch, _] = pooled.dims();

        // Time embeddings for size/coord conditioning (each 256 dim)
        let orig_h_emb = self.size_embedding(original_size.0);
        let orig_w_emb = self.size_embedding(original_size.1);
        let crop_t_emb = self.size_embedding(crop_coords.0);
        let crop_l_emb = self.size_embedding(crop_coords.1);
        let target_h_emb = self.size_embedding(target_size.0);
        let target_w_emb = self.size_embedding(target_size.1);

        // Concatenate: pooled (1280) + size embeddings (6 * 256 = 1536) = 2816
        Tensor::cat(
            vec![
                pooled,
                orig_h_emb.unsqueeze::<2>(),
                orig_w_emb.unsqueeze::<2>(),
                crop_t_emb.unsqueeze::<2>(),
                crop_l_emb.unsqueeze::<2>(),
                target_h_emb.unsqueeze::<2>(),
                target_w_emb.unsqueeze::<2>(),
            ],
            1,
        )
    }

    /// Compute a sinusoidal embedding for a size value
    fn size_embedding(&self, value: usize) -> Tensor<B, 1> {
        compute_size_embedding(value, &self.device)
    }

    /// Encode prompt for SDXL
    pub fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> SdxlConditioning<B> {
        let (cond_context, cond_pooled) = self.encode_text(prompt);
        let (uncond_context, uncond_pooled) = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        SdxlConditioning {
            cond_context,
            uncond_context,
            cond_pooled,
            uncond_pooled,
        }
    }

    /// Sample latent from conditioning
    pub fn sample_latent(
        &self,
        conditioning: &SdxlConditioning<B>,
        config: &SdxlSampleConfig,
    ) -> Tensor<B, 4> {
        let latent_height = config.height / 8;
        let latent_width = config.width / 8;

        // Create DDIM sampler
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sdxl(&self.device), ddim_config);

        // Initialize with random noise
        let mut latent = sampler.init_latent(1, 4, latent_height, latent_width, &self.device);

        // Create add_embed for conditioning
        let cond_add_embed = self.create_add_embed(
            conditioning.cond_pooled.clone(),
            (config.height, config.width),
            (0, 0),
            (config.height, config.width),
        );
        let uncond_add_embed = self.create_add_embed(
            conditioning.uncond_pooled.clone(),
            (config.height, config.width),
            (0, 0),
            (config.height, config.width),
        );

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Sampling loop
        for step_idx in 0..sampler.num_steps() {
            let t = timestep_tensors[step_idx].clone();

            // Predict noise for unconditional
            let noise_uncond = self.unet.forward(
                latent.clone(),
                t.clone(),
                conditioning.uncond_context.clone(),
                uncond_add_embed.clone(),
            );

            // Predict noise for conditional
            let noise_cond = self.unet.forward(
                latent.clone(),
                t,
                conditioning.cond_context.clone(),
                cond_add_embed.clone(),
            );

            // Apply classifier-free guidance
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            // DDIM step
            latent = sampler.step(latent, noise_pred, step_idx);
        }

        latent
    }

    /// Decode latent to image
    pub fn decode(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
        // SDXL uses different VAE scaling factor (0.13025 vs 0.18215)
        let latent = latent / 0.13025;
        self.vae_decoder.forward(latent)
    }

    /// Full generation pipeline
    pub fn generate(
        &self,
        prompt: &str,
        negative_prompt: &str,
        config: &SdxlSampleConfig,
    ) -> Tensor<B, 4> {
        let conditioning = self.encode_prompt(prompt, negative_prompt);
        let latent = self.sample_latent(&conditioning, config);
        self.decode(latent)
    }
}

/// SDXL img2img configuration
#[derive(Debug, Clone)]
pub struct SdxlImg2ImgConfig {
    pub steps: usize,
    pub guidance_scale: f64,
    /// Strength of the transformation (0.0 = no change, 1.0 = full regeneration)
    pub strength: f64,
    pub seed: Option<u64>,
}

impl Default for SdxlImg2ImgConfig {
    fn default() -> Self {
        Self {
            steps: 30,
            guidance_scale: 7.5,
            strength: 0.75,
            seed: None,
        }
    }
}

/// Stable Diffusion XL Img2Img Pipeline
pub struct StableDiffusionXLImg2Img<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub clip_encoder: ClipTextEncoder<B>,
    pub open_clip_encoder: OpenClipTextEncoder<B>,
    pub unet: UNetXL<B>,
    pub vae_encoder: Encoder<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusionXLImg2Img<B> {
    /// Create a new SDXL img2img pipeline
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let clip_config = ClipConfig::sd1x();
        let open_clip_config = OpenClipConfig::sdxl();
        let unet_config = UNetXLConfig::sdxl_base();
        let encoder_config = EncoderConfig::sd();
        let decoder_config = DecoderConfig::sd();

        Self {
            tokenizer,
            clip_encoder: ClipTextEncoder::new(&clip_config, device),
            open_clip_encoder: OpenClipTextEncoder::new(&open_clip_config, device),
            unet: UNetXL::new(&unet_config, device),
            vae_encoder: Encoder::new(&encoder_config, device),
            vae_decoder: Decoder::new(&decoder_config, device),
            scheduler: NoiseSchedule::sdxl(device),
            device: device.clone(),
        }
    }

    /// Encode text using both encoders
    fn encode_text(&self, text: &str) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>();

        let clip_hidden = self.clip_encoder.forward(token_tensor.clone());
        let eos_pos = tokens.iter().position(|&t| t == END_OF_TEXT).unwrap_or(76);
        let (open_clip_hidden, pooled) = self
            .open_clip_encoder
            .forward_with_pooled(token_tensor, &[eos_pos]);

        let context = Tensor::cat(vec![clip_hidden, open_clip_hidden], 2);
        (context, pooled)
    }

    /// Create add_embed from pooled text embedding
    fn create_add_embed(
        &self,
        pooled: Tensor<B, 2>,
        original_size: (usize, usize),
        crop_coords: (usize, usize),
        target_size: (usize, usize),
    ) -> Tensor<B, 2> {
        let orig_h_emb = self.size_embedding(original_size.0);
        let orig_w_emb = self.size_embedding(original_size.1);
        let crop_t_emb = self.size_embedding(crop_coords.0);
        let crop_l_emb = self.size_embedding(crop_coords.1);
        let target_h_emb = self.size_embedding(target_size.0);
        let target_w_emb = self.size_embedding(target_size.1);

        Tensor::cat(
            vec![
                pooled,
                orig_h_emb.unsqueeze::<2>(),
                orig_w_emb.unsqueeze::<2>(),
                crop_t_emb.unsqueeze::<2>(),
                crop_l_emb.unsqueeze::<2>(),
                target_h_emb.unsqueeze::<2>(),
                target_w_emb.unsqueeze::<2>(),
            ],
            1,
        )
    }

    /// Compute a sinusoidal embedding for a size value
    fn size_embedding(&self, value: usize) -> Tensor<B, 1> {
        compute_size_embedding(value, &self.device)
    }

    /// Generate image from input image and prompt
    pub fn generate(
        &self,
        image: Tensor<B, 4>,
        prompt: &str,
        negative_prompt: &str,
        config: &SdxlImg2ImgConfig,
    ) -> Tensor<B, 4> {
        let [_, _, height, width] = image.dims();

        // Encode prompts
        let (cond_context, cond_pooled) = self.encode_text(prompt);
        let (uncond_context, uncond_pooled) = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        // Encode image to latent (SDXL scale factor: 0.13025)
        let init_latent = self.vae_encoder.encode_deterministic(image) * (0.13025 / 0.18215);

        // Calculate start step based on strength
        let num_inference_steps = config.steps;
        let start_step = ((1.0 - config.strength) * num_inference_steps as f64) as usize;

        // Create sampler
        let ddim_config = DdimConfig {
            num_inference_steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sdxl(&self.device), ddim_config);

        // Add noise to init_latent at the start timestep
        let start_timestep = if start_step < sampler.timesteps().len() {
            sampler.timesteps()[start_step]
        } else {
            0
        };

        let noise = Tensor::random(
            init_latent.shape(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &self.device,
        );

        let alpha_t = self.scheduler.alpha_cumprod_at(start_timestep);
        let sqrt_alpha = alpha_t.clone().sqrt();
        let sqrt_one_minus_alpha = (alpha_t.neg() + 1.0).sqrt();

        let mut latent =
            init_latent * sqrt_alpha.unsqueeze() + noise * sqrt_one_minus_alpha.unsqueeze();

        // Create add_embed
        let cond_add_embed =
            self.create_add_embed(cond_pooled, (height, width), (0, 0), (height, width));
        let uncond_add_embed =
            self.create_add_embed(uncond_pooled, (height, width), (0, 0), (height, width));

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Denoising loop
        for step_idx in start_step..sampler.num_steps() {
            let t = timestep_tensors[step_idx].clone();

            let noise_uncond = self.unet.forward(
                latent.clone(),
                t.clone(),
                uncond_context.clone(),
                uncond_add_embed.clone(),
            );
            let noise_cond = self.unet.forward(
                latent.clone(),
                t,
                cond_context.clone(),
                cond_add_embed.clone(),
            );
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            latent = sampler.step(latent, noise_pred, step_idx);
        }

        // Decode (SDXL scale factor)
        let latent = latent / 0.13025;
        self.vae_decoder.forward(latent)
    }
}

// ============================================================================
// SDXL Inpainting Pipeline
// ============================================================================

/// SDXL Inpainting configuration
#[derive(Debug, Clone)]
pub struct SdxlInpaintConfig {
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
}

impl Default for SdxlInpaintConfig {
    fn default() -> Self {
        Self {
            steps: 30,
            guidance_scale: 7.5,
            seed: None,
        }
    }
}

/// Stable Diffusion XL Inpainting Pipeline
///
/// Performs masked image editing using SDXL - regenerates only the masked regions
/// while preserving unmasked areas.
pub struct StableDiffusionXLInpaint<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub clip_encoder: ClipTextEncoder<B>,
    pub open_clip_encoder: OpenClipTextEncoder<B>,
    pub unet: UNetXL<B>,
    pub vae_encoder: Encoder<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusionXLInpaint<B> {
    /// Create a new SDXL inpainting pipeline
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let clip_config = ClipConfig::sd1x();
        let open_clip_config = OpenClipConfig::sdxl();
        let unet_config = UNetXLConfig::sdxl_base();
        let encoder_config = EncoderConfig::sdxl();
        let decoder_config = DecoderConfig::sdxl();

        Self {
            tokenizer,
            clip_encoder: ClipTextEncoder::new(&clip_config, device),
            open_clip_encoder: OpenClipTextEncoder::new(&open_clip_config, device),
            unet: UNetXL::new(&unet_config, device),
            vae_encoder: Encoder::new(&encoder_config, device),
            vae_decoder: Decoder::new(&decoder_config, device),
            scheduler: NoiseSchedule::sdxl(device),
            device: device.clone(),
        }
    }

    /// Encode text using dual encoders (same as SDXL base)
    fn encode_text(&self, text: &str) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>();

        // CLIP hidden states (penultimate layer)
        let clip_hidden = self.clip_encoder.forward_penultimate(token_tensor.clone());

        // OpenCLIP hidden states and pooled
        let eos_pos = tokens.iter().position(|&t| t == END_OF_TEXT).unwrap_or(76);
        let (open_clip_hidden, pooled) = self
            .open_clip_encoder
            .forward_with_pooled(token_tensor, &[eos_pos]);

        // Concatenate hidden states
        let context = Tensor::cat(vec![clip_hidden, open_clip_hidden], 2);

        (context, pooled)
    }

    /// Compute a sinusoidal embedding for a size value
    fn size_embedding(&self, value: usize) -> Tensor<B, 1> {
        compute_size_embedding(value, &self.device)
    }

    /// Create add_embed tensor from pooled embedding and size/crop information
    fn create_add_embed(
        &self,
        pooled: Tensor<B, 2>,
        original_size: (usize, usize),
        crop_coords: (usize, usize),
        target_size: (usize, usize),
    ) -> Tensor<B, 2> {
        let orig_h_emb = self.size_embedding(original_size.0);
        let orig_w_emb = self.size_embedding(original_size.1);
        let crop_t_emb = self.size_embedding(crop_coords.0);
        let crop_l_emb = self.size_embedding(crop_coords.1);
        let target_h_emb = self.size_embedding(target_size.0);
        let target_w_emb = self.size_embedding(target_size.1);

        Tensor::cat(
            vec![
                pooled,
                orig_h_emb.unsqueeze::<2>(),
                orig_w_emb.unsqueeze::<2>(),
                crop_t_emb.unsqueeze::<2>(),
                crop_l_emb.unsqueeze::<2>(),
                target_h_emb.unsqueeze::<2>(),
                target_w_emb.unsqueeze::<2>(),
            ],
            1,
        )
    }

    /// Inpaint masked regions of an image
    ///
    /// # Arguments
    /// * `image` - Input image tensor [1, 3, H, W] with values in [0, 255]
    /// * `mask` - Binary mask tensor [1, 1, H, W] where 1 = regenerate, 0 = preserve
    /// * `prompt` - Text prompt for regenerated regions
    /// * `negative_prompt` - Negative prompt
    /// * `config` - Inpainting configuration
    pub fn inpaint(
        &self,
        image: Tensor<B, 4>,
        mask: Tensor<B, 4>,
        prompt: &str,
        negative_prompt: &str,
        config: &SdxlInpaintConfig,
    ) -> Tensor<B, 4> {
        let [_, _, img_h, img_w] = image.dims();

        // Encode prompts
        let (cond_context, cond_pooled) = self.encode_text(prompt);
        let (uncond_context, uncond_pooled) = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        // Create add_embed
        let cond_add_embed =
            self.create_add_embed(cond_pooled, (img_h, img_w), (0, 0), (img_h, img_w));
        let uncond_add_embed =
            self.create_add_embed(uncond_pooled, (img_h, img_w), (0, 0), (img_h, img_w));

        // Encode image to latent
        let init_latent = self.vae_encoder.encode_deterministic_sdxl(image);

        // Downsample mask to latent size
        let latent_mask = self.downsample_mask(mask, img_h / 8, img_w / 8);

        // Create sampler
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let scheduler = NoiseSchedule::sdxl(&self.device);
        let sampler = DdimSampler::new(scheduler, ddim_config);

        // For blending, we need the noise schedule
        let blend_scheduler = NoiseSchedule::sdxl(&self.device);

        // Initialize latent with noise
        let [_, c, h, w] = init_latent.dims();
        let mut latent = sampler.init_latent(1, c, h, w, &self.device);

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Inpainting loop
        for step_idx in 0..sampler.num_steps() {
            let timestep = sampler.timesteps()[step_idx];
            let t = timestep_tensors[step_idx].clone();

            // Predict noise
            let noise_uncond = self.unet.forward(
                latent.clone(),
                t.clone(),
                uncond_context.clone(),
                uncond_add_embed.clone(),
            );
            let noise_cond = self.unet.forward(
                latent.clone(),
                t.clone(),
                cond_context.clone(),
                cond_add_embed.clone(),
            );
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            // DDIM step for generated regions
            latent = sampler.step(latent, noise_pred, step_idx);

            // Blend: replace unmasked regions with noised original
            let alpha_t = blend_scheduler.alpha_cumprod_at(timestep);
            let sqrt_alpha = alpha_t.clone().sqrt();
            let sqrt_one_minus_alpha = (alpha_t.neg() + 1.0).sqrt();

            let noise = Tensor::random(
                init_latent.shape(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &self.device,
            );
            let noised_original = init_latent.clone() * sqrt_alpha.unsqueeze()
                + noise * sqrt_one_minus_alpha.unsqueeze();

            // mask = 1 means regenerate, mask = 0 means preserve
            latent = latent.clone() * latent_mask.clone()
                + noised_original * (latent_mask.clone().neg() + 1.0);
        }

        // Final blend in latent space (without noise)
        latent = latent.clone() * latent_mask.clone() + init_latent * (latent_mask.neg() + 1.0);

        // Decode with SDXL scaling
        self.vae_decoder.decode_to_image_sdxl(latent)
    }

    /// Downsample mask from image space to latent space
    fn downsample_mask(
        &self,
        mask: Tensor<B, 4>,
        target_h: usize,
        target_w: usize,
    ) -> Tensor<B, 4> {
        let [b, c, h, w] = mask.dims();

        let scale_h = h / target_h;
        let scale_w = w / target_w;

        let mut result = Vec::with_capacity(b * c * target_h * target_w);
        let data = mask.into_data();
        let values: Vec<f32> = data.to_vec().unwrap();

        for batch in 0..b {
            for channel in 0..c {
                for th in 0..target_h {
                    for tw in 0..target_w {
                        let src_h = th * scale_h;
                        let src_w = tw * scale_w;
                        let idx = batch * c * h * w + channel * h * w + src_h * w + src_w;
                        result.push(values[idx]);
                    }
                }
            }
        }

        Tensor::from_data(
            TensorData::new(result, [b, c, target_h, target_w]),
            &self.device,
        )
    }
}

// ============================================================================
// SDXL Refiner
// ============================================================================

/// SDXL Refiner configuration
#[derive(Debug, Clone)]
pub struct RefinerConfig {
    /// Number of refinement steps
    pub steps: usize,
    /// Guidance scale for the refiner
    pub guidance_scale: f64,
    /// At what fraction of total steps to hand off from base to refiner (0.0-1.0)
    /// E.g., 0.8 means refiner takes over at 80% of the way through denoising
    pub denoise_start: f64,
}

impl Default for RefinerConfig {
    fn default() -> Self {
        Self {
            steps: 20,
            guidance_scale: 7.5,
            denoise_start: 0.8,
        }
    }
}

/// SDXL Refiner Pipeline
///
/// Uses only OpenCLIP text encoder (no CLIP) and a different UNet architecture.
/// Designed to refine the output of the SDXL base model.
pub struct StableDiffusionXLRefiner<B: Backend> {
    pub tokenizer: ClipTokenizer,
    pub text_encoder: OpenClipTextEncoder<B>,
    pub unet: UNetXL<B>,
    pub vae_decoder: Decoder<B>,
    pub scheduler: NoiseSchedule<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusionXLRefiner<B> {
    /// Create a new SDXL Refiner pipeline
    pub fn new(tokenizer: ClipTokenizer, device: &B::Device) -> Self {
        let open_clip_config = OpenClipConfig::sdxl();
        let unet_config = UNetXLConfig::sdxl_refiner();
        let vae_config = DecoderConfig::sdxl();

        Self {
            tokenizer,
            text_encoder: OpenClipTextEncoder::new(&open_clip_config, device),
            unet: UNetXL::new(&unet_config, device),
            vae_decoder: Decoder::new(&vae_config, device),
            scheduler: NoiseSchedule::sdxl(device),
            device: device.clone(),
        }
    }

    /// Encode text using OpenCLIP only
    fn encode_text(&self, text: &str) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>();

        let eos_pos = tokens.iter().position(|&t| t == END_OF_TEXT).unwrap_or(76);
        self.text_encoder
            .forward_with_pooled(token_tensor, &[eos_pos])
    }

    /// Create add_embed for refiner (includes aesthetic score)
    fn create_add_embed(
        &self,
        pooled: Tensor<B, 2>,
        original_size: (usize, usize),
        crop_coords: (usize, usize),
        target_size: (usize, usize),
        aesthetic_score: f64,
    ) -> Tensor<B, 2> {
        let orig_h_emb = self.size_embedding(original_size.0);
        let orig_w_emb = self.size_embedding(original_size.1);
        let crop_t_emb = self.size_embedding(crop_coords.0);
        let crop_l_emb = self.size_embedding(crop_coords.1);
        let target_h_emb = self.size_embedding(target_size.0);
        let target_w_emb = self.size_embedding(target_size.1);
        let aesthetic_emb = self.aesthetic_embedding(aesthetic_score);

        // Refiner add_embed: pooled + sizes + aesthetic = 2560
        Tensor::cat(
            vec![
                pooled,
                orig_h_emb.unsqueeze::<2>(),
                orig_w_emb.unsqueeze::<2>(),
                crop_t_emb.unsqueeze::<2>(),
                crop_l_emb.unsqueeze::<2>(),
                target_h_emb.unsqueeze::<2>(),
                target_w_emb.unsqueeze::<2>(),
                aesthetic_emb.unsqueeze::<2>(),
            ],
            1,
        )
    }

    /// Compute a sinusoidal embedding for a size value
    fn size_embedding(&self, value: usize) -> Tensor<B, 1> {
        compute_size_embedding(value, &self.device)
    }

    /// Compute a sinusoidal embedding for an aesthetic score value
    fn aesthetic_embedding(&self, score: f64) -> Tensor<B, 1> {
        // Aesthetic score embedding (same as size embedding but for score)
        let half_dim = 128;
        let value = score as f32;
        let mut emb = vec![0.0f32; 256];
        for i in 0..half_dim {
            let freq = (-((i as f32) / half_dim as f32) * (10000.0f32).ln()).exp();
            emb[i] = (value * freq).sin();
            emb[i + half_dim] = (value * freq).cos();
        }
        Tensor::from_data(TensorData::new(emb, [256]), &self.device)
    }

    /// Refine a latent from the base model
    ///
    /// Takes a partially denoised latent and continues the denoising process.
    pub fn refine(
        &self,
        latent: Tensor<B, 4>,
        prompt: &str,
        negative_prompt: &str,
        config: &RefinerConfig,
    ) -> Tensor<B, 4> {
        let [_, _, height, width] = latent.dims();
        let image_height = height * 8;
        let image_width = width * 8;

        // Encode prompts
        let (cond_context, cond_pooled) = self.encode_text(prompt);
        let (uncond_context, uncond_pooled) = self.encode_text(if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        });

        // Create add_embed with high aesthetic score
        let aesthetic_score = 6.0; // High aesthetic score for positive
        let neg_aesthetic_score = 2.5; // Low aesthetic score for negative

        let cond_add_embed = self.create_add_embed(
            cond_pooled,
            (image_height, image_width),
            (0, 0),
            (image_height, image_width),
            aesthetic_score,
        );
        let uncond_add_embed = self.create_add_embed(
            uncond_pooled,
            (image_height, image_width),
            (0, 0),
            (image_height, image_width),
            neg_aesthetic_score,
        );

        // Create sampler
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sdxl(&self.device), ddim_config);

        // Start from denoise_start
        let start_step = ((1.0 - config.denoise_start) * config.steps as f64) as usize;
        let mut latent = latent;

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Refinement loop
        for step_idx in start_step..sampler.num_steps() {
            let t = timestep_tensors[step_idx].clone();

            let noise_uncond = self.unet.forward(
                latent.clone(),
                t.clone(),
                uncond_context.clone(),
                uncond_add_embed.clone(),
            );
            let noise_cond = self.unet.forward(
                latent.clone(),
                t,
                cond_context.clone(),
                cond_add_embed.clone(),
            );
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            latent = sampler.step(latent, noise_pred, step_idx);
        }

        latent
    }

    /// Decode latent to image
    pub fn decode(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
        self.vae_decoder.decode_to_image_sdxl(latent)
    }
}

// ============================================================================
// SDXL Base + Refiner Combined Workflow
// ============================================================================

/// Configuration for combined base + refiner workflow
#[derive(Debug, Clone)]
pub struct BaseRefinerConfig {
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
    /// Total number of denoising steps (split between base and refiner)
    pub steps: usize,
    /// Base model guidance scale
    pub base_guidance_scale: f64,
    /// Refiner model guidance scale
    pub refiner_guidance_scale: f64,
    /// Fraction of steps for base model (e.g., 0.8 = base does 80% of steps)
    pub refiner_start: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for BaseRefinerConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            steps: 40,
            base_guidance_scale: 7.5,
            refiner_guidance_scale: 7.5,
            refiner_start: 0.8,
            seed: None,
        }
    }
}

/// Combined SDXL Base + Refiner Pipeline
///
/// Runs the base model for initial denoising, then hands off to the refiner
/// for final quality improvements. This is the recommended workflow for
/// highest quality SDXL generation.
pub struct StableDiffusionXLWithRefiner<B: Backend> {
    pub base: StableDiffusionXL<B>,
    pub refiner: StableDiffusionXLRefiner<B>,
    device: B::Device,
}

impl<B: Backend> StableDiffusionXLWithRefiner<B> {
    /// Create a new combined pipeline
    ///
    /// Requires two tokenizers (same vocabulary, different instances) because
    /// the base and refiner pipelines each need their own tokenizer.
    pub fn new(
        base_tokenizer: ClipTokenizer,
        refiner_tokenizer: ClipTokenizer,
        device: &B::Device,
    ) -> Self {
        Self {
            base: StableDiffusionXL::new(base_tokenizer, device),
            refiner: StableDiffusionXLRefiner::new(refiner_tokenizer, device),
            device: device.clone(),
        }
    }

    /// Create from pre-constructed base and refiner pipelines
    pub fn from_pipelines(
        base: StableDiffusionXL<B>,
        refiner: StableDiffusionXLRefiner<B>,
        device: &B::Device,
    ) -> Self {
        Self {
            base,
            refiner,
            device: device.clone(),
        }
    }

    /// Generate image using base + refiner workflow
    pub fn generate(
        &self,
        prompt: &str,
        negative_prompt: &str,
        config: &BaseRefinerConfig,
    ) -> Tensor<B, 4> {
        // Calculate step splits
        let base_steps = ((config.refiner_start) * config.steps as f64) as usize;

        // Run base model for initial denoising
        let base_config = SdxlSampleConfig {
            width: config.width,
            height: config.height,
            steps: config.steps, // Full schedule, but we'll stop early
            guidance_scale: config.base_guidance_scale,
            seed: config.seed,
        };

        let latent = self.sample_base_partial(prompt, negative_prompt, &base_config, base_steps);

        // Run refiner for final quality
        let refiner_config = RefinerConfig {
            steps: config.steps,
            guidance_scale: config.refiner_guidance_scale,
            denoise_start: config.refiner_start,
        };

        let refined = self
            .refiner
            .refine(latent, prompt, negative_prompt, &refiner_config);

        // Decode to image
        self.refiner.decode(refined)
    }

    /// Run base model and stop after specified number of steps
    fn sample_base_partial(
        &self,
        prompt: &str,
        negative_prompt: &str,
        config: &SdxlSampleConfig,
        stop_at_step: usize,
    ) -> Tensor<B, 4> {
        let conditioning = self.base.encode_prompt(prompt, negative_prompt);

        let latent_height = config.height / 8;
        let latent_width = config.width / 8;

        // Create DDIM sampler with full schedule
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(NoiseSchedule::sdxl(&self.device), ddim_config);

        // Initialize with random noise
        let mut latent = sampler.init_latent(1, 4, latent_height, latent_width, &self.device);

        // Create add_embed for conditioning
        let cond_add_embed = self.base.create_add_embed(
            conditioning.cond_pooled.clone(),
            (config.height, config.width),
            (0, 0),
            (config.height, config.width),
        );
        let uncond_add_embed = self.base.create_add_embed(
            conditioning.uncond_pooled.clone(),
            (config.height, config.width),
            (0, 0),
            (config.height, config.width),
        );

        // Precompute timestep tensors
        let timestep_tensors: Vec<Tensor<B, 1>> = sampler
            .timesteps()
            .iter()
            .map(|&t| Tensor::<B, 1>::from_data(TensorData::new(vec![t as f32], [1]), &self.device))
            .collect();

        // Sampling loop - stop at specified step
        for step_idx in 0..stop_at_step {
            let t = timestep_tensors[step_idx].clone();

            // Predict noise for unconditional
            let noise_uncond = self.base.unet.forward(
                latent.clone(),
                t.clone(),
                conditioning.uncond_context.clone(),
                uncond_add_embed.clone(),
            );

            // Predict noise for conditional
            let noise_cond = self.base.unet.forward(
                latent.clone(),
                t,
                conditioning.cond_context.clone(),
                cond_add_embed.clone(),
            );

            // Apply classifier-free guidance
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            // DDIM step
            latent = sampler.step(latent, noise_pred, step_idx);
        }

        latent
    }

    /// Generate with separate prompts for base and refiner
    ///
    /// Useful when you want different prompts for initial generation vs refinement
    pub fn generate_with_prompts(
        &self,
        base_prompt: &str,
        base_negative: &str,
        refiner_prompt: &str,
        refiner_negative: &str,
        config: &BaseRefinerConfig,
    ) -> Tensor<B, 4> {
        let base_steps = ((config.refiner_start) * config.steps as f64) as usize;

        let base_config = SdxlSampleConfig {
            width: config.width,
            height: config.height,
            steps: config.steps,
            guidance_scale: config.base_guidance_scale,
            seed: config.seed,
        };

        let latent = self.sample_base_partial(base_prompt, base_negative, &base_config, base_steps);

        let refiner_config = RefinerConfig {
            steps: config.steps,
            guidance_scale: config.refiner_guidance_scale,
            denoise_start: config.refiner_start,
        };

        let refined =
            self.refiner
                .refine(latent, refiner_prompt, refiner_negative, &refiner_config);
        self.refiner.decode(refined)
    }
}

/// Helper to convert output tensor to image bytes (RGB, 0-255)
pub fn tensor_to_rgb<B: Backend>(tensor: Tensor<B, 4>) -> Vec<u8> {
    let [_, _, h, w] = tensor.dims();

    // Clamp to [0, 255] and convert
    let tensor = tensor.clamp(0.0, 255.0);

    // Get data as f32 (convert if backend uses different precision)
    let data = tensor.into_data();
    let floats: Vec<f32> = data.convert::<f32>().to_vec().unwrap();

    // Convert to u8 RGB (assuming tensor is [1, 3, H, W])
    let mut rgb = Vec::with_capacity(h * w * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let idx = c * h * w + y * w + x;
                rgb.push(floats[idx] as u8);
            }
        }
    }

    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_config_default() {
        let config = SampleConfig::default();
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);
        assert_eq!(config.steps, 50);
        assert_eq!(config.guidance_scale, 7.5);
    }
}
