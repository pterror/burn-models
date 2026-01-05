//! DPM++ 2M variant samplers
//!
//! Includes DPM++ 2M CFG++ and DPM++ 2M SDE Heun variants.

use burn::prelude::*;
use std::collections::VecDeque;

use crate::scheduler::{NoiseSchedule, compute_sigmas, get_ancestral_step, sampler_timesteps};
use crate::guidance::apply_cfg_plus_plus;

/// Configuration for DPM++ 2M CFG++ sampler
#[derive(Debug, Clone)]
pub struct Dpm2mCfgPlusPlusConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Guidance rescale factor (0.0 = standard, 0.7 = recommended)
    pub guidance_rescale: f32,
    /// Use Karras sigmas
    pub use_karras_sigmas: bool,
}

impl Default for Dpm2mCfgPlusPlusConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 20,
            guidance_rescale: 0.7,
            use_karras_sigmas: true,
        }
    }
}

/// DPM++ 2M CFG++ Sampler
///
/// Second-order multistep DPM-Solver++ with CFG++ guidance.
/// CFG++ applies guidance in the denoised space for better quality.
pub struct Dpm2mCfgPlusPlusSampler<B: Backend> {
    config: Dpm2mCfgPlusPlusConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    /// Previous denoised prediction for multistep
    prev_denoised: Option<Tensor<B, 4>>,
}

impl<B: Backend> Dpm2mCfgPlusPlusSampler<B> {
    /// Create a new DPM++ 2M CFG++ sampler
    pub fn new(config: Dpm2mCfgPlusPlusConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(schedule, &timesteps, config.use_karras_sigmas);

        Self {
            config,
            timesteps,
            sigmas,
            prev_denoised: None,
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset the sampler state
    pub fn reset(&mut self) {
        self.prev_denoised = None;
    }

    /// Apply CFG++ guidance
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
            self.config.guidance_rescale,
        )
    }

    /// Perform one DPM++ 2M CFG++ step
    pub fn step(
        &mut self,
        denoised: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        if sigma_next == 0.0 {
            self.prev_denoised = None;
            return denoised;
        }

        let t = -sigma.ln();
        let t_next = -sigma_next.ln();
        let h = t_next - t;

        let result = if let Some(ref prev) = self.prev_denoised {
            // Second order (multistep)
            let h_last = if timestep_idx > 0 {
                -self.sigmas[timestep_idx - 1].ln() + sigma.ln()
            } else {
                h
            };

            let r = h / h_last;
            let denoised_d = denoised.clone() + (denoised.clone() - prev.clone()) * (r / 2.0);

            let sigma_ratio = sigma_next / sigma;
            sample.clone() * sigma_ratio + denoised_d * (1.0 - sigma_ratio)
        } else {
            // First order (Euler)
            let sigma_ratio = sigma_next / sigma;
            sample.clone() * sigma_ratio + denoised.clone() * (1.0 - sigma_ratio)
        };

        self.prev_denoised = Some(denoised);
        result
    }
}

/// Configuration for DPM++ 2M SDE Heun sampler
#[derive(Debug, Clone)]
pub struct Dpm2mSdeHeunConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Use Karras sigmas
    pub use_karras_sigmas: bool,
    /// Eta for noise injection
    pub eta: f32,
    /// S_noise parameter
    pub s_noise: f32,
}

impl Default for Dpm2mSdeHeunConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 20,
            use_karras_sigmas: true,
            eta: 1.0,
            s_noise: 1.0,
        }
    }
}

/// DPM++ 2M SDE Heun Sampler
///
/// Combines DPM++ 2M multistep with SDE noise injection and
/// Heun's method for the SDE correction step.
pub struct Dpm2mSdeHeunSampler<B: Backend> {
    config: Dpm2mSdeHeunConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    /// History of denoised predictions
    denoised_history: VecDeque<Tensor<B, 4>>,
}

impl<B: Backend> Dpm2mSdeHeunSampler<B> {
    /// Create a new DPM++ 2M SDE Heun sampler
    pub fn new(config: Dpm2mSdeHeunConfig, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(schedule, &timesteps, config.use_karras_sigmas);

        Self {
            config,
            timesteps,
            sigmas,
            denoised_history: VecDeque::new(),
        }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset the sampler state
    pub fn reset(&mut self) {
        self.denoised_history.clear();
    }

    /// Perform one DPM++ 2M SDE Heun step
    ///
    /// This combines the DPM++ 2M multistep with SDE noise injection
    /// and uses Heun's method for improved accuracy.
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        model_output_heun: Option<Tensor<B, 4>>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Denoised prediction
        let denoised = sample.clone() - model_output.clone() * sigma;

        // Store for multistep
        self.denoised_history.push_front(denoised.clone());
        if self.denoised_history.len() > 2 {
            self.denoised_history.pop_back();
        }

        if sigma_next == 0.0 {
            return denoised;
        }

        let (sigma_down, sigma_up) = get_ancestral_step(sigma, sigma_next, self.config.eta);

        let t = -sigma.ln();
        let t_down = -sigma_down.max(1e-10).ln();
        let h = t_down - t;

        // Compute the update
        let result = if self.denoised_history.len() >= 2 {
            // Second order with multistep correction
            let prev_denoised = self.denoised_history.get(1).unwrap();

            let h_last = if timestep_idx > 0 {
                -self.sigmas[timestep_idx - 1].ln() + sigma.ln()
            } else {
                h
            };

            let r = h / h_last;
            let denoised_d = denoised.clone() + (denoised.clone() - prev_denoised.clone()) * (r / 2.0);

            let sigma_ratio = sigma_down / sigma;
            sample.clone() * sigma_ratio + denoised_d * (1.0 - sigma_ratio)
        } else if let Some(output_heun) = model_output_heun {
            // Heun's method for first step
            // First, Euler predictor
            let derivative_1 = (sample.clone() - denoised.clone()) / sigma;
            let sample_euler = sample.clone() + derivative_1.clone() * (sigma_down - sigma);

            // Heun corrector
            let denoised_2 = sample_euler.clone() - output_heun * sigma_down;
            let derivative_2 = (sample_euler.clone() - denoised_2) / sigma_down.max(1e-8);

            // Average derivatives
            let derivative_avg = (derivative_1 + derivative_2) * 0.5;
            sample.clone() + derivative_avg * (sigma_down - sigma)
        } else {
            // First order (Euler)
            let sigma_ratio = sigma_down / sigma;
            sample.clone() * sigma_ratio + denoised.clone() * (1.0 - sigma_ratio)
        };

        // Add noise for SDE
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
    fn test_dpm2m_cfg_plus_plus_config_default() {
        let config = Dpm2mCfgPlusPlusConfig::default();
        assert_eq!(config.num_inference_steps, 20);
        assert_eq!(config.guidance_rescale, 0.7);
        assert!(config.use_karras_sigmas);
    }

    #[test]
    fn test_dpm2m_sde_heun_config_default() {
        let config = Dpm2mSdeHeunConfig::default();
        assert_eq!(config.num_inference_steps, 20);
        assert_eq!(config.eta, 1.0);
        assert!(config.use_karras_sigmas);
    }
}
