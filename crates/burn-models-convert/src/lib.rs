//! Weight Loading and Serialization
//!
//! This crate handles loading model weights from safetensors format
//! and provides serialization utilities for Burn models.
//!
//! # Weight Loading
//!
//! Load weights from safetensors files:
//!
//! ```ignore
//! use burn_models_convert::{SafeTensorFile, LoadError};
//!
//! let file = SafeTensorFile::open("model.safetensors")?;
//! let tensor = file.load_tensor::<f32>("model.encoder.weight")?;
//! ```
//!
//! # Specialized Loaders
//!
//! - [`load_lora`] - Load LoRA weights (Kohya, Diffusers formats)
//! - [`load_controlnet_info`] - Detect ControlNet type and config
//! - [`load_embedding`] - Load textual inversion embeddings
//!
//! # Serialization
//!
//! Save and load Burn models with configurable precision:
//!
//! ```ignore
//! use burn_models_convert::{BinFileRecorder, FullPrecisionSettings};
//!
//! // Save model
//! let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
//! recorder.record(model, "model.bin")?;
//!
//! // Load model
//! let model = recorder.load("model.bin", &device)?;
//! ```
//!
//! # Precision Options
//!
//! - [`FullPrecisionSettings`] - f32 weights
//! - [`HalfPrecisionSettings`] - f16/bf16 weights

pub mod controlnet_loader;
pub mod embedding_loader;
pub mod loader;
pub mod lora_loader;
pub mod mapping;
pub mod serialize;

pub use controlnet_loader::{load_controlnet_info, ControlNetInfo, ControlNetLoadError, ControlNetType};
pub use embedding_loader::{load_embedding, EmbeddingFormat, EmbeddingLoadError};
pub use loader::{LoadError, SafeTensorFile};
pub use lora_loader::{load_lora, LoraFormat, LoraLoadError};
pub use serialize::{
    bytes_recorder, full_precision_recorder, half_precision_recorder,
    BinBytesRecorder, BinFileRecorder, FullPrecisionSettings, HalfPrecisionSettings,
    Recorder, RecorderError, SerializeError,
};
