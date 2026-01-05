//! Euler CFG++ samplers
//!
//! CFG++ (Classifier-Free Guidance Plus Plus) is an improved guidance method
//! that applies guidance in the denoised space rather than noise space,
//! leading to better image quality and fewer artifacts.

use burn::prelude::*;

use crate::scheduler::NoiseSchedule;

/// Configuration for Euler CFG++ sampler
#[derive(Debug, Clone)]
pub struct EulerCfgPlusPlusConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Guidance rescale factor (0.0 = standard CFG, 0.7 = recommended)
    pub guidance_rescale: f32,
}

impl Default for EulerCfgPlusPlusConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 30,
            guidance_rescale: 0.7,
        }
    }
}

/// Euler CFG++ Sampler
///
/// Uses CFG++ guidance which applies classifier-free guidance in the
/// denoised prediction space rather than the noise prediction space.
/// This reduces artifacts and improves image quality at high guidance scales.
pub struct EulerCfgPlusPlusSampler<B: Backend> {
    config: EulerCfgPlusPlusConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> EulerCfgPlusPlusSampler<B> {
    /// Create a new Euler CFG++ sampler
    pub fn new(config: EulerCfgPlusPlusConfig, schedule: &NoiseSchedule<B>) -> Self {
        let num_train_steps = schedule.num_train_steps;
        let step_ratio = num_train_steps / config.num_inference_steps;

        let timesteps: Vec<usize> = (0..config.num_inference_steps)
            .rev()
            .map(|i| (i * step_ratio).min(num_train_steps - 1))
            .collect();

        let sigmas = Self::compute_sigmas(schedule, &timesteps);

        Self {
            config,
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    fn compute_sigmas(schedule: &NoiseSchedule<B>, timesteps: &[usize]) -> Vec<f32> {
        let mut sigmas: Vec<f32> = timesteps
            .iter()
            .map(|&t| {
                let alpha_cumprod = schedule.alpha_cumprod_at(t);
                let alpha_data = alpha_cumprod.into_data();
                let alpha: f32 = alpha_data.to_vec().unwrap()[0];
                ((1.0 - alpha) / alpha).sqrt()
            })
            .collect();
        sigmas.push(0.0);
        sigmas
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Apply CFG++ guidance
    ///
    /// CFG++ applies guidance in the denoised space:
    /// x0_guided = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
    /// Then optionally rescales to prevent over-saturation
    pub fn apply_cfg_plus_plus(
        &self,
        noise_pred_uncond: Tensor<B, 4>,
        noise_pred_cond: Tensor<B, 4>,
        sample: Tensor<B, 4>,
        sigma: f32,
        guidance_scale: f32,
    ) -> Tensor<B, 4> {
        // Convert noise predictions to x0 predictions
        let x0_uncond = sample.clone() - noise_pred_uncond.clone() * sigma;
        let x0_cond = sample.clone() - noise_pred_cond.clone() * sigma;

        // Apply guidance in x0 space
        let x0_guided = x0_uncond.clone() + (x0_cond.clone() - x0_uncond.clone()) * guidance_scale;

        // Optionally rescale to prevent over-saturation
        if self.config.guidance_rescale > 0.0 {
            let std_cond = Self::compute_std(&x0_cond);
            let std_guided = Self::compute_std(&x0_guided);

            if std_guided > 1e-6 {
                let rescale_factor =
                    std_cond / std_guided * self.config.guidance_rescale + (1.0 - self.config.guidance_rescale);
                x0_guided * rescale_factor
            } else {
                x0_guided
            }
        } else {
            x0_guided
        }
    }

    fn compute_std(tensor: &Tensor<B, 4>) -> f32 {
        // Compute standard deviation across all elements
        let flattened = tensor.clone().flatten::<1>(0, 3);
        let var = flattened.clone().var(0);
        let std_tensor = var.sqrt();
        let std_data = std_tensor.into_data();
        std_data.to_vec::<f32>().unwrap()[0]
    }

    /// Perform one Euler CFG++ step
    pub fn step(
        &self,
        x0_guided: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Convert x0 back to noise prediction
        let noise_pred = (sample.clone() - x0_guided.clone()) / sigma;

        // Euler step
        let dt = sigma_next - sigma;
        let denoised = x0_guided;
        let derivative = (sample.clone() - denoised) / sigma;

        sample + derivative * dt
    }
}

/// Euler Ancestral CFG++ Sampler
///
/// Combines CFG++ guidance with ancestral sampling (noise injection).
pub struct EulerAncestralCfgPlusPlusSampler<B: Backend> {
    config: EulerCfgPlusPlusConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> EulerAncestralCfgPlusPlusSampler<B> {
    /// Create a new Euler Ancestral CFG++ sampler
    pub fn new(config: EulerCfgPlusPlusConfig, schedule: &NoiseSchedule<B>) -> Self {
        let num_train_steps = schedule.num_train_steps;
        let step_ratio = num_train_steps / config.num_inference_steps;

        let timesteps: Vec<usize> = (0..config.num_inference_steps)
            .rev()
            .map(|i| (i * step_ratio).min(num_train_steps - 1))
            .collect();

        let sigmas = Self::compute_sigmas(schedule, &timesteps);

        Self {
            config,
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    fn compute_sigmas(schedule: &NoiseSchedule<B>, timesteps: &[usize]) -> Vec<f32> {
        let mut sigmas: Vec<f32> = timesteps
            .iter()
            .map(|&t| {
                let alpha_cumprod = schedule.alpha_cumprod_at(t);
                let alpha_data = alpha_cumprod.into_data();
                let alpha: f32 = alpha_data.to_vec().unwrap()[0];
                ((1.0 - alpha) / alpha).sqrt()
            })
            .collect();
        sigmas.push(0.0);
        sigmas
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Apply CFG++ guidance (same as non-ancestral version)
    pub fn apply_cfg_plus_plus(
        &self,
        noise_pred_uncond: Tensor<B, 4>,
        noise_pred_cond: Tensor<B, 4>,
        sample: Tensor<B, 4>,
        sigma: f32,
        guidance_scale: f32,
    ) -> Tensor<B, 4> {
        let x0_uncond = sample.clone() - noise_pred_uncond.clone() * sigma;
        let x0_cond = sample.clone() - noise_pred_cond.clone() * sigma;

        let x0_guided = x0_uncond.clone() + (x0_cond.clone() - x0_uncond.clone()) * guidance_scale;

        if self.config.guidance_rescale > 0.0 {
            let std_cond = Self::compute_std(&x0_cond);
            let std_guided = Self::compute_std(&x0_guided);

            if std_guided > 1e-6 {
                let rescale_factor =
                    std_cond / std_guided * self.config.guidance_rescale + (1.0 - self.config.guidance_rescale);
                x0_guided * rescale_factor
            } else {
                x0_guided
            }
        } else {
            x0_guided
        }
    }

    fn compute_std(tensor: &Tensor<B, 4>) -> f32 {
        let flattened = tensor.clone().flatten::<1>(0, 3);
        let var = flattened.clone().var(0);
        let std_tensor = var.sqrt();
        let std_data = std_tensor.into_data();
        std_data.to_vec::<f32>().unwrap()[0]
    }

    /// Perform one Euler Ancestral CFG++ step
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

        // Compute sigma_up and sigma_down for ancestral sampling
        let sigma_up = (sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2)).sqrt();
        let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).sqrt();

        // Euler step to sigma_down
        let derivative = (sample.clone() - x0_guided.clone()) / sigma;
        let sample_down = sample + derivative * (sigma_down - sigma);

        // Add noise
        let device = sample_down.device();
        let shape = sample_down.dims();
        let noise: Tensor<B, 4> = Tensor::random(
            shape,
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        sample_down + noise * sigma_up
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_cfg_plus_plus_config_default() {
        let config = EulerCfgPlusPlusConfig::default();
        assert_eq!(config.num_inference_steps, 30);
        assert_eq!(config.guidance_rescale, 0.7);
    }
}
