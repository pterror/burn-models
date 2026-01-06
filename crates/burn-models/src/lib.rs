//! burn-models: Model Inference in Pure Rust
//!
//! A pure Rust implementation of popular AI models using the Burn deep learning framework.
//!
//! # Supported Models
//!
//! ## Image Generation
//! - Stable Diffusion 1.x and SDXL
//! - ControlNet, IP-Adapter, T2I-Adapter
//! - LoRA and Textual Inversion
//!
//! ## Large Language Models
//! - LLaMA 2/3 (7B, 13B, 70B, 8B)
//! - Mixtral 8x7B/8x22B (Mixture of Experts)
//!
//! # Backend Selection
//!
//! Choose a backend via feature flags:
//! - `ndarray`: CPU backend (no GPU required)
//! - `tch`: PyTorch backend via libtorch (CUDA, MPS support)
//! - `wgpu`: WebGPU backend (cross-platform GPU)
//! - `cuda`: Native CUDA backend (NVIDIA only)
//!
//! # Examples
//!
//! ## Stable Diffusion
//!
//! ```toml
//! [dependencies]
//! burn-models = { version = "0.1", features = ["wgpu"] }
//! ```
//!
//! ```ignore
//! use burn_models::{backends::Wgpu, StableDiffusionXL, SdxlSampleConfig};
//!
//! let device = burn_models::backends::default_device();
//! let tokenizer = burn_models::clip::ClipTokenizer::from_file("vocab.txt")?;
//! let pipeline = StableDiffusionXL::<Wgpu>::new(tokenizer, &device);
//!
//! let config = SdxlSampleConfig::default();
//! let image = pipeline.generate("a sunset over mountains", "", &config);
//! ```
//!
//! ## LLaMA
//!
//! ```ignore
//! use burn_models::llm::{Llama, LlamaConfig, load_llama};
//!
//! let config = LlamaConfig::llama3_8b();
//! let (model, runtime) = load_llama::<MyBackend>("model.safetensors", &config, &device)?;
//!
//! let output = model.forward(input_ids, &runtime, None);
//! ```

pub use burn_models_clip as clip;
pub use burn_models_convert as convert;
pub use burn_models_core as core;
pub use burn_models_llm as llm;
pub use burn_models_samplers as samplers;
pub use burn_models_unet as unet;
pub use burn_models_vae as vae;

// Re-export LoRA types for convenience
pub use burn_models_convert::lora_loader::{LoraFormat, LoraLoadError, load_lora};
pub use burn_models_core::lora::{LoraConvWeight, LoraModel, LoraWeight, LoraWeightType};

// Re-export ControlNet types
pub use burn_models_convert::controlnet_loader::{
    ControlNetInfo, ControlNetLoadError, ControlNetType, load_controlnet_info,
};
pub use burn_models_unet::controlnet::{
    ControlNet, ControlNetConfig, ControlNetOutput, ControlNetPreprocessor,
};

// Re-export IP-Adapter types
pub use burn_models_unet::ip_adapter::{
    ImageProjection, IpAdapter, IpAdapterConfig, combine_embeddings,
};

// Re-export T2I-Adapter types
pub use burn_models_unet::t2i_adapter::{
    T2IAdapter, T2IAdapterConfig, T2IAdapterOutput, T2IAdapterType,
};

// Re-export textual inversion / embedding types
pub use burn_models_convert::embedding_loader::{
    EmbeddingFormat, EmbeddingLoadError, load_embedding,
};
pub use burn_models_core::textual_inversion::{
    EmbeddingError, EmbeddingManager, TextualInversionEmbedding, find_placeholder_tokens,
};

// Re-export precision types (configuration only - see precision module docs)
pub use burn_models_core::precision::{PrecisionConfig, PrecisionMode};

pub mod backends;
pub mod batch;
pub mod memory;
pub mod offload;
pub mod pipeline;

pub use batch::{BatchConfig, BatchResult};
pub use offload::{ModelComponent, OffloadConfig, OffloadState, OffloadStrategy, PipelinePhase};

pub use pipeline::{
    // SDXL Base + Refiner Workflow
    BaseRefinerConfig,
    DiffusionPipeline,
    Img2ImgConfig,
    // Inpainting
    InpaintConfig,
    LatentFormat,
    // SDXL Refiner
    RefinerConfig,
    SampleConfig,
    Sd1xConditioning,
    // SDXL
    SdxlConditioning,
    SdxlImg2ImgConfig,
    // SDXL Inpainting
    SdxlInpaintConfig,
    SdxlSampleConfig,
    StableDiffusion1x,
    StableDiffusion1xImg2Img,
    StableDiffusion1xInpaint,
    StableDiffusionXL,
    StableDiffusionXLImg2Img,
    StableDiffusionXLInpaint,
    StableDiffusionXLRefiner,
    StableDiffusionXLWithRefiner,
    StepInfo,
    // Step callback types
    StepOutput,
    latent_to_preview,
    tensor_to_rgb,
};

pub use memory::{MemoryConfig, TiledVae};
