//! Shared attention utilities for CLIP encoders

use burn::prelude::*;

/// Create a causal attention mask
///
/// Creates a lower triangular mask with -inf for masked positions,
/// ensuring tokens can only attend to previous tokens.
pub fn create_causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    let data = TensorData::new(mask_data, [seq_len, seq_len]);
    Tensor::from_data(data, device)
}

/// Compute scaled dot-product attention
///
/// Shared attention computation used by both CLIP and OpenCLIP.
pub fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 2>>,
    head_dim: usize,
) -> Tensor<B, 4> {
    // Scaled dot-product attention
    let scale = (head_dim as f64).powf(-0.5);
    let attn = q.matmul(k.transpose()) * scale;

    // Apply causal mask
    let attn = match mask {
        Some(m) => attn + m.unsqueeze::<4>(),
        None => attn,
    };

    let attn = burn::tensor::activation::softmax(attn, 3);
    attn.matmul(v)
}
