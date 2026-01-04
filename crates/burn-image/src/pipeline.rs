//! Diffusion pipeline trait and implementations

use burn::prelude::*;

use burn_image_clip::{ClipConfig, ClipTextEncoder, ClipTokenizer, END_OF_TEXT};
use burn_image_samplers::{apply_guidance, DdimConfig, DdimSampler, NoiseSchedule};
use burn_image_unet::{UNet, UNetConfig};
use burn_image_vae::{Decoder, DecoderConfig};

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

/// Unified interface for diffusion pipelines
pub trait DiffusionPipeline<B: Backend> {
    type Conditioning;

    /// Encode text prompt into conditioning
    fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> Self::Conditioning;

    /// Sample latent from conditioning
    fn sample_latent(&self, conditioning: &Self::Conditioning, config: &SampleConfig) -> Tensor<B, 4>;

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
    device: B::Device,
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

    /// Encode a single text prompt
    fn encode_text(&self, text: &str) -> Tensor<B, 3> {
        let tokens = self.tokenizer.encode_padded(text, 77);
        let token_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), [77]),
            &self.device,
        );
        let token_tensor = token_tensor.unsqueeze::<2>(); // [1, 77]

        self.text_encoder.forward(token_tensor)
    }

    /// Find EOS token position
    fn find_eos_position(tokens: &[u32]) -> usize {
        tokens.iter().position(|&t| t == END_OF_TEXT).unwrap_or(tokens.len() - 1)
    }
}

impl<B: Backend> DiffusionPipeline<B> for StableDiffusion1x<B> {
    type Conditioning = Sd1xConditioning<B>;

    fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> Self::Conditioning {
        let cond = self.encode_text(prompt);
        let uncond = self.encode_text(if negative_prompt.is_empty() { "" } else { negative_prompt });

        Sd1xConditioning { cond, uncond }
    }

    fn sample_latent(&self, conditioning: &Self::Conditioning, config: &SampleConfig) -> Tensor<B, 4> {
        let latent_height = config.height / 8;
        let latent_width = config.width / 8;

        // Create DDIM sampler
        let ddim_config = DdimConfig {
            num_inference_steps: config.steps,
            eta: 0.0,
        };
        let sampler = DdimSampler::new(
            NoiseSchedule::sd1x(&self.device),
            ddim_config,
        );

        // Initialize with random noise
        let mut latent = sampler.init_latent(1, 4, latent_height, latent_width, &self.device);

        // Sampling loop
        for step_idx in 0..sampler.num_steps() {
            let timestep = sampler.timesteps()[step_idx];
            let t = Tensor::<B, 1>::from_data(
                TensorData::new(vec![timestep as f32], [1]),
                &self.device,
            );

            // Predict noise for unconditional
            let noise_uncond = self.unet.forward(
                latent.clone(),
                t.clone(),
                conditioning.uncond.clone(),
            );

            // Predict noise for conditional
            let noise_cond = self.unet.forward(
                latent.clone(),
                t,
                conditioning.cond.clone(),
            );

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

/// Helper to convert output tensor to image bytes (RGB, 0-255)
pub fn tensor_to_rgb<B: Backend>(tensor: Tensor<B, 4>) -> Vec<u8> {
    let [_, _, h, w] = tensor.dims();

    // Clamp to [0, 255] and convert
    let tensor = tensor.clamp(0.0, 255.0);

    // Get data as f32
    let data = tensor.into_data();
    let floats: Vec<f32> = data.to_vec().unwrap();

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
