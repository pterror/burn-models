# Alternative Architectures

This document covers non-transformer and hybrid architectures that represent alternatives to the standard transformer-based models. These architectures aim to address transformer limitations like quadratic attention complexity, fixed context windows, and KV cache memory requirements.

---

## Alternative Architectures (Non-Transformer)

These architectures replace the transformer's attention mechanism entirely with alternatives that offer linear or subquadratic complexity.

### RWKV

**RWKV** (Receptance Weighted Key Value) is an RNN architecture that achieves transformer-level performance while maintaining linear time complexity and constant space (no KV cache).

| Version | Codename | Key Innovation |
|---------|----------|----------------|
| RWKV-5 | Eagle | Matrix-valued states |
| RWKV-6 | Finch | Dynamic recurrence via LoRA |
| RWKV-7 | Goose | Dynamic state evolution, generalized delta rule |
| RWKV-8 | Heron | DeepEmbed for sparse deployment (upcoming) |

**Key Features:**
- Linear time complexity O(n) vs transformer's O(n²)
- Constant memory (no KV cache needed)
- Infinite context length theoretically possible
- Parallelizable during training like transformers
- RWKV-7 is a "meta-in-context learner" - performs test-time training via in-context gradient descent

**Architecture Details:**
- Combines benefits of RNNs (efficient inference) with transformers (parallelizable training)
- Uses time-mixing and channel-mixing blocks instead of attention
- RWKV-7 introduces Dynamic State Evolution for improved context modeling

**Available Models:**
- RWKV7-G1 2.9B (recommended for ≤7B use cases)
- Trained on World v3.5 dataset (5.16T tokens)
- Strong reasoning, coding, and math capabilities

**Resources:**
- [RWKV GitHub](https://github.com/BlinkDL/RWKV-LM)
- [RWKV Wiki](https://wiki.rwkv.com/)
- [RWKV Paper (arXiv:2305.13048)](https://arxiv.org/abs/2305.13048)
- [RWKV-7 Survey (arXiv:2412.14847)](https://arxiv.org/abs/2412.14847)
- [Hugging Face Blog](https://huggingface.co/blog/rwkv)

---

### Mamba / Mamba-2

**Mamba** is a state space model (SSM) architecture that achieves linear scaling with sequence length while maintaining the ability to selectively focus on relevant parts of the input.

| Version | Key Innovation |
|---------|----------------|
| Mamba | Selective state spaces, hardware-aware parallel scan |
| Mamba-2 | State Space Duality (SSD), can operate in SSM or attention modes |

**Key Features:**
- 5x higher throughput than transformers during inference
- Linear scaling with sequence length
- Performance improves up to million-length sequences
- State-of-the-art on language, audio, and genomics tasks
- Mamba-3B outperforms transformers of same size, matches 2x larger

**Architecture Details:**
- **Selective State Spaces**: Ability to selectively focus on or ignore parts of input based on relevance
- **Hardware-aware Parallel Scan**: Optimizes GPU memory hierarchy for speed
- **Mamba-2 SSD**: Mathematical bridge between SSMs and transformers, enables transformer optimizations while keeping linear scalability

**Available Models:**
- mamba-130m through mamba-2.8b (300B tokens on The Pile)
- mamba2-130m through mamba2-2.7b
- mamba-2.8b-slimpj (600B tokens on SlimPajama)

**Variants:**
- **MoE Mamba**: Alternating Mamba and MoE layers, 2.2x fewer training steps
- **Vision Mamba (Vim)**: Bidirectional Mamba blocks for visual encoding

**Resources:**
- [Mamba GitHub](https://github.com/state-spaces/mamba)
- [Mamba Paper (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752)
- [Mamba-2 Blog](https://tridao.me/blog/2024/mamba2-part1-model/)
- [Visual Guide to Mamba](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)

---

### xLSTM (Extended LSTM)

**xLSTM** is a modernized LSTM architecture from Sepp Hochreiter (original LSTM inventor) that overcomes classical LSTM limitations through exponential gating and matrix memory.

**Key Innovations:**
- **Exponential Gating**: With appropriate normalization and stabilization
- **sLSTM**: Scalar memory with new memory mixing
- **mLSTM**: Fully parallelizable with matrix memory and covariance update rule

**Architecture:**
- xLSTM blocks combine sLSTM and mLSTM in residual block backbones
- Residually stacked into full xLSTM architectures
- Achieves favorable comparison with transformers and SSMs in performance and scaling

**Performance:**
- Large models (350M-1.3B) outperform transformer LLMs in perplexity
- Robust extrapolation to long contexts
- Vision-LSTM matches/surpasses ViT derivatives on ImageNet
- xLSTMTime achieves up to 18% improvement on time series benchmarks

**Available Models:**
- xLSTM Large (7B parameters, 2.3T tokens)
- Added to Hugging Face Transformers (July 2025)

**Resources:**
- [xLSTM GitHub](https://github.com/NX-AI/xlstm)
- [xLSTM Paper (arXiv:2405.04517)](https://arxiv.org/abs/2405.04517)
- [Hugging Face Docs](https://huggingface.co/docs/transformers/en/model_doc/xlstm)

---

### Griffin / Hawk (RecurrentGemma)

**Griffin** and **Hawk** are Google DeepMind models that combine gated linear recurrences with local attention for efficient language modeling.

| Model | Architecture |
|-------|--------------|
| Hawk | Pure RNN with gated linear recurrences (RG-LRU layer) |
| Griffin | Hybrid: RG-LRU + local attention |

**Key Features:**
- Hawk exceeds Mamba performance on downstream tasks
- Griffin matches Llama-2 performance with 6x fewer training tokens
- Hardware efficiency matches transformers during training
- Lower latency and higher throughput during inference
- Can extrapolate to 4x+ longer sequences than trained on
- Scaled up to 14B parameters

**Architecture Details:**
- **RG-LRU (Real-Gated Linear Recurrent Unit)**: Novel gated linear recurrent layer
- Griffin mixes RG-LRU with local attention
- Fixed-sized state reduces memory and enables efficient long sequence inference

**RecurrentGemma:**
- Open-weights models based on Griffin architecture
- Fast inference when generating long sequences

**Resources:**
- [RecurrentGemma GitHub](https://github.com/google-deepmind/recurrentgemma)
- [Griffin Paper (arXiv:2402.19427)](https://arxiv.org/abs/2402.19427)

---

### RetNet (Retentive Network)

**RetNet** is Microsoft's architecture that achieves training parallelism, low-cost inference, and good performance simultaneously through a novel retention mechanism.

**Three Computation Paradigms:**
1. **Parallel**: Enables training parallelism
2. **Recurrent**: O(1) inference cost, improves throughput/latency/memory
3. **Chunkwise Recurrent**: Linear complexity for long sequences

**Key Features:**
- 8.4x faster decoding than transformers with KV cache (7B model, 8K context)
- 70% less memory than transformers
- Better performance than transformer variants (RWKV, H3, Hyena, Linear Transformer)
- Multi-scale retention mechanism replaces multi-head attention

**Architecture Details:**
- Theoretically derives connection between recurrence and attention
- Uses xPos positional encoding
- No softmax required - enables parallel and recurrent processing

**Updates:**
- Gated RetNet (RetNet-3) released May 2024 as part of YOCO

**Resources:**
- [RetNet GitHub](https://github.com/microsoft/unilm/tree/master/retnet)
- [RetNet Paper (arXiv:2307.08621)](https://arxiv.org/abs/2307.08621)
- [Microsoft Research](https://www.microsoft.com/en-us/research/publication/retentive-network-a-successor-to-transformer-for-large-language-models/)

---

### Hyena

**Hyena** is a subquadratic attention replacement using long convolutions and data-controlled gating.

**Key Features:**
- 50+ point accuracy improvement over other implicit methods on long sequence tasks
- 20% training compute reduction vs transformers at 2K sequence length
- 2x faster than optimized attention at 8K context
- 100x faster at 64K context
- State-of-the-art for dense-attention-free architectures on WikiText103 and The Pile

**Architecture Details:**
- Interleaves implicitly parametrized long convolutions with multiplicative gating
- Hyena filters parameterized by feed-forward networks
- Builds on H3 method, extends to arbitrary number of projections
- Causal convolutions ensure autoregressive property
- Hierarchical convolutions scale linearly with input length

**How It Works:**
1. Take linear projections of input
2. Combine using long convolutions and element-wise multiplication
3. Apply convolutions in time/frequency domain alternately

**Resources:**
- [Hyena Paper (arXiv:2302.10866)](https://arxiv.org/abs/2302.10866)
- [Safari GitHub](https://github.com/HazyResearch/safari)
- [Hazy Research Blog](https://hazyresearch.stanford.edu/blog/2023-03-07-hyena)

---

### TTT (Test-Time Training)

**TTT** layers replace attention with a mechanism that compresses context through gradient descent, updating the hidden state during inference.

**Key Concept:**
- Hidden state is itself a machine learning model (linear model or MLP)
- Updated by self-supervised learning on input sequence, even at test time
- "Learning to learn at test time"

**Variants:**
- **TTT-Linear**: Hidden state is a linear model
- **TTT-MLP**: Hidden state is a two-layer MLP

**Performance:**
- Evaluated at 125M to 1.3B parameters
- Unlike Mamba, keeps reducing perplexity beyond 16K context
- TTT-Linear already faster than transformer at 8K context
- Matches Mamba in wall-clock time

**Recent Developments (2025):**
- **LaCT**: Hybrid with quadratic attention for local + linear TTT for non-local
- **TTT-E2E**: End-to-end test-time training via meta-learning
  - Scales with context length like full attention transformers
  - 2.7x faster than full attention at 128K context
  - Constant inference latency regardless of context length

**Resources:**
- [TTT Paper (arXiv:2407.04620)](https://arxiv.org/abs/2407.04620)
- [TTT GitHub](https://github.com/test-time-training/ttt-lm-jax)
- [Project Website](https://test-time-training.github.io/)

---

## Hybrid Architectures

These combine transformers with SSMs or other efficient architectures to get the best of both worlds.

### Jamba

**Jamba** is AI21's hybrid Transformer-Mamba-MoE architecture, the first production-grade Mamba-based model.

**Architecture:**
- Interleaves Transformer and Mamba layers at 1:7 ratio (1 attention per 7 Mamba)
- MoE layers every two blocks
- 16 experts, chooses 4 (vs Mixtral's 8 choose 2)
- Only 12B of 52B parameters active at inference

| Model | Total Params | Active Params | Context |
|-------|--------------|---------------|---------|
| Jamba 1.0 | 52B | 12B | 256K |
| Jamba 1.5 Mini | - | 12B | 256K |
| Jamba 1.5 Large | 398B | 94B | 256K |

**Key Features:**
- Fits in single 80GB GPU (original version)
- 3x throughput on long contexts vs Mixtral 8x7B
- Strong results up to 256K token context
- Apache 2.0 license

**Jamba 1.5 Architecture:**
- 72 layers interleaving Mamba and attention
- Grouped-query attention
- Low-rank adaptation
- 16 MoE experts for efficient routing

**Resources:**
- [Jamba Paper (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887)
- [Jamba 1.5 Paper (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570)
- [AI21 Blog](https://www.ai21.com/blog/announcing-jamba/)

---

### Zamba / Zamba2

**Zamba** is Zyphra's SSM-transformer hybrid with a Mamba backbone and shared attention layers.

**Architecture:**
- Mamba layers interspersed with shared attention layer(s)
- Zamba1: One shared attention block
- Zamba2: Two shared attention blocks
- Shared weights minimize parameter cost

| Model | Total Params | Active Params | Training Data |
|-------|--------------|---------------|---------------|
| Zamba 7B v1 | 7B | - | 1T tokens |
| Zamba2 1.2B | 1.2B | - | - |
| Zamba2 2.7B | 2.7B | - | - |
| Zamba2 7.4B | 7.4B | - | - |

**Key Features:**
- Significantly faster inference than comparable transformers
- Substantially less memory for long sequence generation
- Mamba2 blocks have ~4x throughput of equal-parameter transformer blocks
- Only need KV cache for shared attention invocations
- Outperforms LLaMA 1, LLaMA 2 7B, OLMo-7B with <50% training data

**Zamba2 Performance:**
- State-of-the-art for size class
- Outperforms Mistral, Gemma, Llama3 at 7B scale

**Resources:**
- [Zamba Paper (arXiv:2405.16712)](https://arxiv.org/abs/2405.16712)
- [Zamba HuggingFace](https://huggingface.co/Zyphra/Zamba-7B-v1)
- [Zyphra Blog](https://www.zyphra.com/post/zamba2-small)

---

### StripedHyena

**StripedHyena** is a hybrid architecture from Together AI and Nous Research, combining attention with gated Hyena convolutions.

**Architecture:**
- Multi-head grouped-query attention + gated convolutions in Hyena blocks
- "Striped Attention" - alternating attention and Hyena operators
- Builds on H3, Hyena, HyenaDNA, Monarch Mixer

**Models:**
- **StripedHyena-Hessian-7B (SH 7B)**: Base model
- **StripedHyena-Nous-7B (SH-N 7B)**: Chat model (with Nous Research)

**Key Features:**
- First alternative model competitive with best open-source transformers
- Up to 128K token context
- Training speed improvements: 30% at 32K, 50% at 64K, 100%+ at 128K
- 50%+ memory reduction during autoregressive generation
- Outperforms Llama-2 7B, Yi 7B, RWKV 14B on short-context tasks

**Collaborators:**
HazyResearch, hessian.AI, Nous Research, MILA, HuggingFace, DFKI

**Resources:**
- [StripedHyena GitHub](https://github.com/togethercomputer/stripedhyena)
- [StripedHyena HuggingFace](https://huggingface.co/togethercomputer/StripedHyena-Nous-7B)
- [Together AI Blog](https://www.together.ai/blog/stripedhyena-7b)

---

## Diffusion Language Models

These apply diffusion model principles to text generation, offering a fundamentally different approach from autoregressive models.

### LLaDA (Large Language Diffusion with mAsking)

**LLaDA** is a diffusion model for language trained from scratch with pre-training and SFT.

**Architecture:**
- Based on LLaMA3 but with bidirectional transformer (no causal mask)
- Forward process: data masking
- Reverse process: predict masked tokens
- Vanilla multi-head attention (not GQA, incompatible with KV caching)
- Reduced FFN dimension to compensate for attention parameter increase

**Key Differences from BERT:**
- Variable masking ratio (0 to 1) vs BERT's fixed ratio
- Training objective is upper bound on negative log-likelihood
- Makes LLaDA a true generative model

**Models:**
- LLaDA 1B
- LLaDA 8B (2.3T tokens, 0.13M H800 GPU hours)
- LLaDA-MoE-7B-A1B (~1B active params, surpasses LLaDA 1.5 8B dense)

**Key Features:**
- Competitive with LLaMA3 8B on in-context learning
- Strong instruction-following after SFT
- Solves the "reversal curse" - surpasses GPT-4o on reversal poem completion

**Resources:**
- [LLaDA GitHub](https://github.com/ML-GSAI/LLaDA)
- [LLaDA Paper (arXiv:2502.09992)](https://arxiv.org/abs/2502.09992)
- [LLaDA Demo](https://ml-gsai.github.io/LLaDA-demo/)

---

### TESS-2

**TESS-2** is a general instruction-following diffusion language model that matches/exceeds strong autoregressive models.

**Architecture:**
- Simplex diffusion model (builds on TESS-1)
- **Simplex-based Representation**: Words as k-logit simplex prior to token generation
- **Bidirectional Attention**: Full bidirectional during training
- **Self-conditioning**: Model predictions included as inputs for subsequent diffusion steps

**Training:**
1. Adapt strong AR model via continued pretraining with diffusion loss
2. Instruction tuning

**Key Innovations:**
- **Reward Guidance**: Novel inference-time guidance for alignment without training
- Performance improves with increased inference-time compute
- Fine-grained controllability over compute used at inference

**Performance:**
- Outperforms contemporary instruction-tuned diffusion models
- Matches/exceeds strong AR models

**Resources:**
- [TESS-2 GitHub](https://github.com/hamishivi/tess-2)
- [TESS-2 Paper (arXiv:2502.13917)](https://arxiv.org/abs/2502.13917)
- [ACL 2025 Paper](https://aclanthology.org/2025.acl-long.1029/)

---

## Fast/Distilled Image Generation

These models achieve high-quality image generation in very few steps through distillation techniques.

### SDXL-Lightning

**SDXL-Lightning** from ByteDance achieves state-of-the-art 1-4 step image generation.

**Method:**
- Progressive adversarial diffusion distillation
- Distilled from stable-diffusion-xl-base-1.0
- Uses SDXL's U-Net as discriminator (works in latent space)
- Enables training at 1024x1024 (vs SDXL Turbo's 512x512)

**Checkpoints:**
- 1-step, 2-step, 4-step, 8-step models
- Full UNet and LoRA versions
- LoRA can be applied to other base models

**Performance:**
- 4-step and 8-step often outperform original SDXL at 32 steps
- ~209ms for 1024x1024 image
- No negative prompt or CFG needed (baked into training)

**Resources:**
- [SDXL-Lightning HuggingFace](https://huggingface.co/ByteDance/SDXL-Lightning)
- [Paper (arXiv:2402.13929)](https://arxiv.org/abs/2402.13929)

---

### Hyper-SD / Hyper-SDXL

**Hyper-SD** models generate high-quality images in 1-8 steps.

**Performance:**
- Quantitatively better than SDXL Lightning
- Available for SD 1.5, SDXL, and other base models

**Resources:**
- [Stable Diffusion Art Guide](https://stable-diffusion-art.com/hyper-sdxl/)

---

### SANA

**SANA** from NVIDIA Research and MIT enables efficient high-resolution image synthesis with linear complexity.

**Key Innovations:**

1. **Deep Compression Autoencoder (DC-AE)**
   - 32x compression (vs traditional 8x)
   - 1024x1024 image → 32x32 latent
   - Drastically reduces sequence length for diffusion

2. **Linear DiT**
   - Replaces vanilla quadratic attention with linear attention
   - Complexity O(N) instead of O(N²)

3. **Decoder-only Text Encoder**
   - Uses Gemma LLM instead of CLIP/T5
   - Superior text comprehension and instruction-following

4. **Efficient Training/Sampling**
   - Flow-DPM-Solver reduces sampling steps
   - Efficient caption labeling accelerates convergence

**Performance:**
- SANA-0.6B competitive with Flux-12B
- 20x smaller, 100x+ faster throughput
- Deployable on 16GB laptop GPU
- <1 second for 1024x1024 image
- Up to 4096x4096 resolution

**SANA-Sprint (2025):**
- One/few-step generator
- 0.1s on H100, 0.3s on RTX 4090 for 1024px

**SANA-Video:**
- 5s linear DiT video model
- Real-time minute-length video with LongLive

**Resources:**
- [SANA GitHub](https://github.com/NVlabs/Sana)
- [SANA Project Page](https://nvlabs.github.io/Sana/)
- [Paper (arXiv:2410.10629)](https://arxiv.org/abs/2410.10629)

---

## Additional Video Generation

### Open-Sora

**Open-Sora** is an open-source implementation following OpenAI's Sora architecture report.

**Architecture:**
- 3D autoencoder for video compression
- Text encoder for prompts
- DiT-like transformer for video/text latents

**Note:** OpenAI's Sora itself is proprietary. Open-Sora provides an open alternative.

**Resources:**
- [Open-Sora GitHub](https://github.com/hpcaitech/Open-Sora)
- [Paper (arXiv:2412.20404)](https://arxiv.org/abs/2412.20404)

---

## Additional LLMs

### OLMo 3

**OLMo** (Open Language Model) from AI2 is designed for scientific transparency with full access to weights, data, code, and training recipes.

**Timeline:**
- OLMo 1 (Feb 2024): Initial release
- OLMoE (Sep 2024): MoE variant
- OLMo 2 (Nov 2024): 7B and 13B, up to 5T tokens
- OLMo 2 1B (May 2025): Outperforms Gemma 3 1B, Llama 3.2 1B
- OLMo 2 32B (2025): Largest in family
- OLMo 3 (Nov 2025): Production-ready, commercial use

**OLMo 3 Variants:**
- **OLMo 3 Base**: Core foundation model
- **OLMo 3 Instruct**: Tuned for instructions
- **OLMo 3 Think**: Explicit reasoning
- **OLMo 3 RL Zero**: Experimental RL training

**Key Features:**
- 65K token context (short book chapter length)
- 2.5x more efficient to train than Llama 3.1
- Outperforms Stanford Marin and Llama 3.1 open-weight models
- Full transparency: weights, data, code, checkpoints

**Training:**
- Two-stage curriculum pretraining
- OLMo-Mix-1124: ~3.9T tokens from DCLM, Dolma, Starcoder, Proof Pile II

**Resources:**
- [OLMo GitHub](https://github.com/allenai/OLMo)
- [AI2 OLMo Page](https://allenai.org/olmo)
- [OLMo 2 Blog](https://allenai.org/blog/olmo2)

---

### DBRX

**DBRX** from Databricks is a fine-grained MoE model that advances efficiency through more, smaller experts.

**Architecture:**
- Transformer decoder-only
- Fine-grained MoE: 16 experts, choose 4 (vs Mixtral's 8 choose 2)
- 65x more expert combinations than coarse-grained approaches
- Uses RoPE, GLU, GQA
- GPT-4 tokenizer (tiktoken)
- "Dropless" MoE routing (MegaBlocks library)

| Metric | Value |
|--------|-------|
| Total Parameters | 132B |
| Active Parameters | 36B |
| Training Data | 12T tokens |
| Training Hardware | 3072 NVIDIA H100s |

**Performance:**
- Outperforms all established open source models on standard benchmarks
- 2x faster inference than LLaMA2-70B
- ~40% size of Grok-1 (total and active params)

**Models:**
- DBRX Base
- DBRX Instruct

**Resources:**
- [DBRX HuggingFace](https://huggingface.co/databricks/dbrx-instruct)
- [Databricks Blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)

---

## Implementation Considerations

### Shared Components Needed

Many alternative architectures share building blocks:

| Component | Used By |
|-----------|---------|
| Linear attention | Mamba, SANA, RetNet |
| State space layers | Mamba, Griffin, Jamba, Zamba |
| Gated convolutions | Hyena, StripedHyena, RWKV |
| MoE routing | Jamba, DBRX, MoE-Mamba |
| RoPE | Most LLMs |

### Priority Recommendations

**High Priority (widely used, good documentation):**
1. Mamba/Mamba-2 - Foundation for many hybrids
2. RWKV-7 - Active community, production-ready
3. xLSTM - Now in HuggingFace Transformers

**Medium Priority (interesting but more specialized):**
4. Jamba - Production MoE hybrid
5. Griffin - Powers RecurrentGemma
6. SANA - Efficient image generation

**Experimental (novel approaches):**
7. TTT - Unique gradient-descent-as-inference approach
8. LLaDA/TESS-2 - Diffusion for language
9. RetNet - Microsoft's retention mechanism
