//! Flash Attention and memory-efficient attention implementations
//!
//! Flash Attention reduces memory usage from O(n²) to O(n) by computing
//! attention in tiles without materializing the full attention matrix.

use burn::prelude::*;

/// Attention implementation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionStrategy {
    /// Standard attention (materializes full attention matrix)
    #[default]
    Standard,
    /// Flash Attention style (tiled computation)
    Flash,
    /// Memory efficient (chunked queries)
    MemoryEfficient,
    /// Sliced attention (process in smaller batches)
    Sliced,
}

impl AttentionStrategy {
    /// Get a human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            AttentionStrategy::Standard => "standard",
            AttentionStrategy::Flash => "flash",
            AttentionStrategy::MemoryEfficient => "memory_efficient",
            AttentionStrategy::Sliced => "sliced",
        }
    }

    /// Estimated relative memory usage (1.0 = standard)
    pub fn memory_factor(&self) -> f32 {
        match self {
            AttentionStrategy::Standard => 1.0,
            AttentionStrategy::Flash => 0.1,
            AttentionStrategy::MemoryEfficient => 0.3,
            AttentionStrategy::Sliced => 0.5,
        }
    }
}

/// Configuration for attention computation
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Which strategy to use
    pub strategy: AttentionStrategy,
    /// Chunk size for memory-efficient attention
    pub chunk_size: usize,
    /// Slice size for sliced attention
    pub slice_size: usize,
    /// Whether to use fp16 for attention computation
    pub use_fp16: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            strategy: AttentionStrategy::Standard,
            chunk_size: 1024,
            slice_size: 4,
            use_fp16: false,
        }
    }
}

impl AttentionConfig {
    /// Configuration for Flash Attention
    pub fn flash() -> Self {
        Self {
            strategy: AttentionStrategy::Flash,
            chunk_size: 256,
            slice_size: 4,
            use_fp16: true,
        }
    }

    /// Configuration for memory-efficient attention
    pub fn memory_efficient() -> Self {
        Self {
            strategy: AttentionStrategy::MemoryEfficient,
            chunk_size: 512,
            slice_size: 4,
            use_fp16: false,
        }
    }

    /// Configuration for sliced attention (good for limited VRAM)
    pub fn sliced(slice_size: usize) -> Self {
        Self {
            strategy: AttentionStrategy::Sliced,
            chunk_size: 1024,
            slice_size,
            use_fp16: false,
        }
    }
}

/// Compute attention with the specified strategy
pub fn compute_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    config: &AttentionConfig,
) -> Tensor<B, 4> {
    match config.strategy {
        AttentionStrategy::Standard => standard_attention(q, k, v),
        AttentionStrategy::Flash => flash_attention(q, k, v, config.chunk_size),
        AttentionStrategy::MemoryEfficient => memory_efficient_attention(q, k, v, config.chunk_size),
        AttentionStrategy::Sliced => sliced_attention(q, k, v, config.slice_size),
    }
}

/// Standard attention implementation
///
/// Computes: softmax(Q @ K^T / sqrt(d)) @ V
fn standard_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [_, _, _, head_dim] = q.dims();
    let scale = (head_dim as f64).powf(-0.5);

    let attn = q.matmul(k.transpose()) * scale;
    let attn = burn::tensor::activation::softmax(attn, 3);
    attn.matmul(v)
}

/// Flash Attention style implementation
///
/// Processes attention in chunks to reduce peak memory usage.
/// This is a simplified version - true Flash Attention uses custom CUDA kernels.
fn flash_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    chunk_size: usize,
) -> Tensor<B, 4> {
    let [batch, heads, seq_len, head_dim] = q.dims();
    let [_, _, kv_len, _] = k.dims();
    let scale = (head_dim as f64).powf(-0.5);
    let device = q.device();

    // If sequence is short enough, use standard attention
    if seq_len <= chunk_size && kv_len <= chunk_size {
        return standard_attention(q, k, v);
    }

    // Process in chunks
    let num_q_chunks = (seq_len + chunk_size - 1) / chunk_size;
    let mut outputs = Vec::new();

    for q_chunk_idx in 0..num_q_chunks {
        let q_start = q_chunk_idx * chunk_size;
        let q_end = (q_start + chunk_size).min(seq_len);

        let q_chunk = q.clone().slice([0..batch, 0..heads, q_start..q_end, 0..head_dim]);

        // Compute attention for this query chunk against all keys
        let attn = q_chunk.matmul(k.clone().transpose()) * scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out_chunk = attn.matmul(v.clone());

        outputs.push(out_chunk);
    }

    // Concatenate chunks
    Tensor::cat(outputs, 2)
}

/// Memory-efficient attention
///
/// Processes queries in chunks to reduce memory usage.
fn memory_efficient_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    chunk_size: usize,
) -> Tensor<B, 4> {
    let [batch, heads, seq_len, head_dim] = q.dims();
    let scale = (head_dim as f64).powf(-0.5);

    if seq_len <= chunk_size {
        return standard_attention(q, k, v);
    }

    let num_chunks = (seq_len + chunk_size - 1) / chunk_size;
    let mut outputs = Vec::new();

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(seq_len);

        let q_chunk = q.clone().slice([0..batch, 0..heads, start..end, 0..head_dim]);

        // Standard attention for this chunk
        let attn = q_chunk.matmul(k.clone().transpose()) * scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out_chunk = attn.matmul(v.clone());

        outputs.push(out_chunk);
    }

    Tensor::cat(outputs, 2)
}

/// Sliced attention
///
/// Processes batch slices sequentially to reduce peak memory.
fn sliced_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    slice_size: usize,
) -> Tensor<B, 4> {
    let [batch, heads, seq_len, head_dim] = q.dims();
    let [_, _, kv_len, _] = k.dims();
    let scale = (head_dim as f64).powf(-0.5);

    if heads <= slice_size {
        return standard_attention(q, k, v);
    }

    let num_slices = (heads + slice_size - 1) / slice_size;
    let mut outputs = Vec::new();

    for slice_idx in 0..num_slices {
        let start = slice_idx * slice_size;
        let end = (start + slice_size).min(heads);

        let q_slice = q.clone().slice([0..batch, start..end, 0..seq_len, 0..head_dim]);
        let k_slice = k.clone().slice([0..batch, start..end, 0..kv_len, 0..head_dim]);
        let v_slice = v.clone().slice([0..batch, start..end, 0..kv_len, 0..head_dim]);

        let attn = q_slice.matmul(k_slice.transpose()) * scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out_slice = attn.matmul(v_slice);

        outputs.push(out_slice);
    }

    Tensor::cat(outputs, 1)
}

/// Compute attention weights only (for visualization)
pub fn compute_attention_weights<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [_, _, _, head_dim] = q.dims();
    let scale = (head_dim as f64).powf(-0.5);

    let attn = q.matmul(k.transpose()) * scale;
    burn::tensor::activation::softmax(attn, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_strategy_default() {
        assert_eq!(AttentionStrategy::default(), AttentionStrategy::Standard);
    }

    #[test]
    fn test_attention_config_presets() {
        let flash = AttentionConfig::flash();
        assert_eq!(flash.strategy, AttentionStrategy::Flash);
        assert!(flash.use_fp16);

        let efficient = AttentionConfig::memory_efficient();
        assert_eq!(efficient.strategy, AttentionStrategy::MemoryEfficient);

        let sliced = AttentionConfig::sliced(2);
        assert_eq!(sliced.strategy, AttentionStrategy::Sliced);
        assert_eq!(sliced.slice_size, 2);
    }

    #[test]
    fn test_memory_factor() {
        assert!(AttentionStrategy::Flash.memory_factor() < AttentionStrategy::Standard.memory_factor());
    }
}
