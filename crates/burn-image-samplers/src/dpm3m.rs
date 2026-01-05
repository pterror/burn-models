//! DPM++ 3M SDE sampler
//!
//! Third-order DPM-Solver++ with SDE (stochastic differential equation) formulation.
//! Provides better quality than 2M variants for some prompts.

use burn::prelude::*;
use std::collections::VecDeque;

use crate::scheduler::NoiseSchedule;

/// Configuration for DPM++ 3M SDE sampler
#[derive(Debug, Clone)]
pub struct Dpm3mSdeConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Use Karras noise schedule
    pub use_karras_sigmas: bool,
    /// Eta for noise injection (0 = ODE, 1 = full SDE)
    pub eta: f32,
    /// S_noise parameter for noise scaling
    pub s_noise: f32,
}

impl Default for Dpm3mSdeConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 20,
            use_karras_sigmas: true,
            eta: 1.0,
            s_noise: 1.0,
        }
    }
}

/// DPM++ 3M SDE sampler
///
/// Third-order DPM-Solver++ with stochastic noise injection.
/// Uses three past model outputs for higher accuracy.
pub struct Dpm3mSdeSampler<B: Backend> {
    config: Dpm3mSdeConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    /// History of model outputs
    model_outputs: VecDeque<Tensor<B, 4>>,
}

impl<B: Backend> Dpm3mSdeSampler<B> {
    /// Create a new DPM++ 3M SDE sampler
    pub fn new(config: Dpm3mSdeConfig, schedule: &NoiseSchedule<B>) -> Self {
        let num_train_steps = schedule.num_train_steps;
        let step_ratio = num_train_steps / config.num_inference_steps;

        let timesteps: Vec<usize> = (0..config.num_inference_steps)
            .rev()
            .map(|i| (i * step_ratio).min(num_train_steps - 1))
            .collect();

        let sigmas = Self::compute_sigmas(schedule, &timesteps, config.use_karras_sigmas);

        Self {
            config,
            timesteps,
            sigmas,
            model_outputs: VecDeque::new(),
        }
    }

    fn compute_sigmas(schedule: &NoiseSchedule<B>, timesteps: &[usize], use_karras: bool) -> Vec<f32> {
        let mut sigmas: Vec<f32> = timesteps
            .iter()
            .map(|&t| {
                let alpha_cumprod = schedule.alpha_cumprod_at(t);
                let alpha_data = alpha_cumprod.into_data();
                let alpha: f32 = alpha_data.to_vec().unwrap()[0];
                ((1.0 - alpha) / alpha).sqrt()
            })
            .collect();

        if use_karras {
            let sigma_min = *sigmas.last().unwrap_or(&0.0);
            let sigma_max = *sigmas.first().unwrap_or(&1.0);
            let n = sigmas.len();
            let rho = 7.0;

            sigmas = (0..n)
                .map(|i| {
                    let t = i as f32 / (n - 1).max(1) as f32;
                    (sigma_max.powf(1.0 / rho) + t * (sigma_min.powf(1.0 / rho) - sigma_max.powf(1.0 / rho))).powf(rho)
                })
                .collect();
        }

        sigmas.push(0.0);
        sigmas
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset the model output history
    pub fn reset(&mut self) {
        self.model_outputs.clear();
    }

    /// Compute sigma_down and sigma_up for SDE
    fn get_ancestral_step(&self, sigma: f32, sigma_next: f32) -> (f32, f32) {
        if sigma_next == 0.0 {
            return (0.0, 0.0);
        }

        let sigma_up = (sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2))
            .sqrt()
            .min(sigma_next)
            * self.config.eta;
        let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).sqrt();

        (sigma_down, sigma_up)
    }

    /// Perform one DPM++ 3M SDE step
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Convert to denoised
        let denoised = sample.clone() - model_output.clone() * sigma;

        // Store for multi-step
        self.model_outputs.push_front(denoised.clone());
        if self.model_outputs.len() > 3 {
            self.model_outputs.pop_back();
        }

        if sigma_next == 0.0 {
            return denoised;
        }

        let (sigma_down, sigma_up) = self.get_ancestral_step(sigma, sigma_next);

        // Compute step based on available history
        let t = -sigma.ln();
        let t_next = -sigma_down.max(1e-10).ln();
        let h = t_next - t;

        let result = if self.model_outputs.len() == 1 {
            // First order (Euler)
            let derivative = (sample.clone() - denoised.clone()) / sigma;
            sample.clone() + derivative * (sigma_down - sigma)
        } else if self.model_outputs.len() == 2 {
            // Second order (DPM++ 2M style)
            let denoised_1 = self.model_outputs.get(1).unwrap();

            let h_1 = if timestep_idx > 0 {
                -self.sigmas[timestep_idx].ln() + self.sigmas[timestep_idx - 1].ln()
            } else {
                h
            };

            let r = h / h_1;
            let d0 = denoised.clone();
            let d1 = denoised_1.clone();

            // Second order correction
            let denoised_d = d0.clone() + (d0.clone() - d1) * (r / 2.0);

            let derivative = (sample.clone() - denoised_d.clone()) / sigma;
            sample.clone() + derivative * (sigma_down - sigma)
        } else {
            // Third order (DPM++ 3M)
            let denoised_1 = self.model_outputs.get(1).unwrap();
            let denoised_2 = self.model_outputs.get(2).unwrap();

            let h_1 = if timestep_idx > 0 {
                -self.sigmas[timestep_idx].ln() + self.sigmas[timestep_idx - 1].ln()
            } else {
                h
            };
            let h_2 = if timestep_idx > 1 {
                -self.sigmas[timestep_idx - 1].ln() + self.sigmas[timestep_idx - 2].ln()
            } else {
                h_1
            };

            let r0 = h / h_1;
            let r1 = h_1 / h_2;

            let d0 = denoised.clone();
            let d1 = denoised_1.clone();
            let d2 = denoised_2.clone();

            // Third order correction using polynomial interpolation
            let d1_0 = d0.clone() - d1.clone();
            let d1_1 = d1.clone() - d2.clone();
            let d2_0 = d1_0.clone() - d1_1.clone() * r0;

            let denoised_d = d0.clone()
                + d1_0.clone() * (r0 / 2.0)
                + d2_0 * (r0 * r0 / 6.0);

            let derivative = (sample.clone() - denoised_d.clone()) / sigma;
            sample.clone() + derivative * (sigma_down - sigma)
        };

        // Add noise for SDE
        if sigma_up > 0.0 && self.config.eta > 0.0 {
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
    fn test_dpm3m_sde_config_default() {
        let config = Dpm3mSdeConfig::default();
        assert_eq!(config.num_inference_steps, 20);
        assert!(config.use_karras_sigmas);
        assert_eq!(config.eta, 1.0);
    }
}
