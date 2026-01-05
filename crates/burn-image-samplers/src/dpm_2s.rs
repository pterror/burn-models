//! DPM++ 2S Ancestral samplers
//!
//! Second-order singlestep DPM-Solver++ with ancestral sampling.
//! The "2S" indicates second-order singlestep (vs multistep).

use burn::prelude::*;

use crate::scheduler::{NoiseSchedule, compute_sigmas, get_ancestral_step, sampler_timesteps};
use crate::guidance::apply_cfg_plus_plus;

/// Configuration for DPM++ 2S Ancestral sampler
#[derive(Debug, Clone)]
pub struct Dpm2sAncestralConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Use Karras sigmas
    pub use_karras_sigmas: bool,
    /// S_noise parameter for noise scaling
    pub s_noise: f32,
}

impl Default for Dpm2sAncestralConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 25,
            use_karras_sigmas: true,
            s_noise: 1.0,
        }
    }
}

/// DPM++ 2S Ancestral Sampler
///
/// Second-order singlestep DPM-Solver++ with noise injection.
/// Uses a midpoint method with ancestral sampling for creative outputs.
pub struct Dpm2sAncestralSampler<B: Backend> {
    config: Dpm2sAncestralConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Dpm2sAncestralSampler<B> {
    /// Create a new DPM++ 2S Ancestral sampler
    pub fn new(config: Dpm2sAncestralConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(schedule, &timesteps, config.use_karras_sigmas);

        Self {
            config,
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Perform one DPM++ 2S Ancestral step
    ///
    /// This is a singlestep method, requiring two model evaluations per step.
    /// The first evaluation is at the current point, the second at the midpoint.
    pub fn step(
        &self,
        model_output: Tensor<B, 4>,
        model_output_2: Option<Tensor<B, 4>>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        if sigma_next == 0.0 {
            return sample.clone() - model_output * sigma;
        }

        let (sigma_down, sigma_up) = get_ancestral_step(sigma, sigma_next, 1.0);

        // Denoised prediction (x0)
        let denoised = sample.clone() - model_output.clone() * sigma;

        // Compute midpoint sigma
        let sigma_mid = (sigma * sigma_down).sqrt();

        let result = if let Some(model_output_mid) = model_output_2 {
            // Full second-order step with midpoint evaluation
            let denoised_mid = sample.clone() - model_output_mid * sigma_mid;

            // Second-order update
            let t = -sigma.ln();
            let t_next = -sigma_down.ln();
            let h = t_next - t;

            let d0 = denoised.clone();
            let d1 = denoised_mid;
            let coeff = h / 2.0;

            let x_next = (sample.clone() / sigma) * sigma_down
                + d0 * (1.0 - sigma_down / sigma) * (1.0 - coeff)
                + d1 * (1.0 - sigma_down / sigma) * coeff;
            x_next
        } else {
            // First-order fallback (Euler)
            let derivative = (sample.clone() - denoised.clone()) / sigma;
            sample.clone() + derivative * (sigma_down - sigma)
        };

        // Add ancestral noise
        if sigma_up > 0.0 {
            let device = result.device();
            let shape = result.dims();
            let noise: Tensor<B, 4> = Tensor::random(
                shape,
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            result + noise * (sigma_up * self.config.s_noise)
        } else {
            result
        }
    }

    /// Get the midpoint sigma for the second model evaluation
    pub fn get_midpoint_sigma(&self, timestep_idx: usize) -> f32 {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];
        let (sigma_down, _) = get_ancestral_step(sigma, sigma_next, 1.0);
        (sigma * sigma_down).sqrt()
    }
}

/// DPM++ 2S Ancestral CFG++ Sampler
///
/// Combines DPM++ 2S Ancestral with CFG++ guidance.
pub struct Dpm2sAncestralCfgPlusPlusSampler<B: Backend> {
    config: Dpm2sAncestralConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    /// Guidance rescale factor
    pub guidance_rescale: f32,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Dpm2sAncestralCfgPlusPlusSampler<B> {
    /// Create a new DPM++ 2S Ancestral CFG++ sampler
    pub fn new(config: Dpm2sAncestralConfig, schedule: &NoiseSchedule<B>, guidance_rescale: f32) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(schedule, &timesteps, config.use_karras_sigmas);

        Self {
            config,
            timesteps,
            sigmas,
            guidance_rescale,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Apply CFG++ guidance in denoised space
    pub fn apply_cfg_plus_plus_guidance(
        &self,
        noise_pred_uncond: Tensor<B, 4>,
        noise_pred_cond: Tensor<B, 4>,
        sample: Tensor<B, 4>,
        sigma: f32,
        guidance_scale: f32,
    ) -> Tensor<B, 4> {
        apply_cfg_plus_plus(
            noise_pred_uncond,
            noise_pred_cond,
            sample,
            sigma,
            guidance_scale,
            self.guidance_rescale,
        )
    }

    /// Perform one step with pre-computed guided x0
    pub fn step(
        &self,
        x0_guided: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        if sigma_next == 0.0 {
            return x0_guided;
        }

        // Ancestral step
        let (sigma_down, sigma_up) = get_ancestral_step(sigma, sigma_next, 1.0);

        // Euler step
        let derivative = (sample.clone() - x0_guided.clone()) / sigma;
        let result = sample + derivative * (sigma_down - sigma);

        // Add noise
        if sigma_up > 0.0 {
            let device = result.device();
            let shape = result.dims();
            let noise: Tensor<B, 4> = Tensor::random(
                shape,
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            result + noise * (sigma_up * self.config.s_noise)
        } else {
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpm2s_ancestral_config_default() {
        let config = Dpm2sAncestralConfig::default();
        assert_eq!(config.num_inference_steps, 25);
        assert!(config.use_karras_sigmas);
        assert_eq!(config.s_noise, 1.0);
    }
}
