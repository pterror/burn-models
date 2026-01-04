use burn::prelude::*;

/// Multi-head self-attention
pub fn qkv_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 2>>,
) -> Tensor<B, 4> {
    let [batch, heads, seq_len, head_dim] = q.dims();
    let scale = (head_dim as f64).powf(-0.25);

    let q = q * scale;
    let k = k * scale;

    // [batch, heads, seq_q, seq_k]
    let attn = q.matmul(k.transpose());

    let attn = match mask {
        Some(m) => attn + m.unsqueeze::<4>(),
        None => attn,
    };

    let attn = burn::tensor::activation::softmax(attn, 3);

    attn.matmul(v)
}

/// Causal attention mask for autoregressive decoding
pub fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    let mask = Tensor::<B, 2>::zeros([seq_len, seq_len], device);
    // TODO: implement lower triangular mask with -inf for future positions
    mask
}
