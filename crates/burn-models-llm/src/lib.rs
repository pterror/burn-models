//! LLM implementations for burn-models
//!
//! This crate provides implementations of popular Large Language Models
//! using the Burn deep learning framework.
//!
//! # Supported Models
//!
//! - **LLaMA**: LLaMA 2 (7B, 13B, 70B) and LLaMA 3 (8B, 70B)
//!
//! # Example
//!
//! ```ignore
//! use burn_models_llm::llama::{Llama, LlamaConfig};
//!
//! // Create a tiny model for testing
//! let config = LlamaConfig::tiny();
//! let model = config.init::<MyBackend>(&device);
//!
//! // Forward pass
//! let output = model.forward(input_ids, None);
//!
//! // Generate text
//! let generated = model.generate(prompt, 100, 0.8);
//! ```

pub mod deepseek;
pub mod deepseek_loader;
pub mod gemma;
pub mod gemma_loader;
pub mod llama;
pub mod llama_loader;
pub mod mistral;
pub mod mistral_loader;
pub mod mixtral;
pub mod phi;
pub mod phi_loader;
pub mod qwen;
pub mod qwen_loader;

pub use deepseek::{DeepSeek, DeepSeekConfig, DeepSeekOutput, DeepSeekRuntime};
pub use deepseek_loader::{load_deepseek, DeepSeekLoadError};
pub use gemma::{Gemma, GemmaConfig, GemmaOutput, GemmaRuntime};
pub use gemma_loader::{load_gemma, GemmaLoadError};
pub use phi::{Phi, PhiConfig, PhiOutput, PhiRuntime};
pub use phi_loader::{load_phi, PhiLoadError};
pub use llama::{Llama, LlamaConfig, LlamaOutput, LlamaRuntime};
pub use llama_loader::{load_llama, LlamaLoadError};
pub use mistral::{Mistral, MistralConfig, MistralOutput, MistralRuntime};
pub use mistral_loader::{load_mistral, MistralLoadError};
pub use mixtral::{Mixtral, MixtralConfig, MixtralOutput, MixtralRuntime};
pub use qwen::{Qwen, QwenConfig, QwenOutput, QwenRuntime};
pub use qwen_loader::{load_qwen, QwenLoadError};
