//! Diffusion pipeline trait and implementations

use burn::prelude::*;

use burn_image_clip::{ClipConfig, ClipTextEncoder, ClipTokenizer, END_OF_TEXT};
use burn_image_samplers::{apply_guidance, DdimConfig, DdimSampler, NoiseSchedule};
use burn_image_unet::{UNet, UNetConfig};
use burn_image_vae::{Decoder, DecoderConfig, Encoder, EncoderConfig};

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
        let token_tensor = Tensor::<B, 1>::from_data(
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
        let uncond = self.encode_text(if negative_prompt.is_empty() { "" } else { negative_prompt });

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

        let mut latent = init_latent * sqrt_alpha.unsqueeze() + noise * sqrt_one_minus_alpha.unsqueeze();

        // Denoising loop from start_step
        for step_idx in start_step..sampler.num_steps() {
            let timestep = sampler.timesteps()[step_idx];
            let t = Tensor::<B, 1>::from_data(
                TensorData::new(vec![timestep as f32], [1]),
                &self.device,
            );

            let noise_uncond = self.unet.forward(latent.clone(), t.clone(), uncond.clone());
            let noise_cond = self.unet.forward(latent.clone(), t, cond.clone());
            let noise_pred = apply_guidance(noise_uncond, noise_cond, config.guidance_scale);

            latent = sampler.step(latent, noise_pred, step_idx);
        }

        // Decode
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
