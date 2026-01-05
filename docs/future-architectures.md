# Future Architectures

This document tracks image generation architectures that could be implemented in burn-models.

## UNet-based

These use similar architecture to the current SD 1.x and SDXL implementations.

### Stable Diffusion 2.x

SD2 uses OpenCLIP instead of the original CLIP text encoder, and supports v-prediction in addition to epsilon-prediction.

| Variant | Resolution | Text Encoder |
|---------|------------|--------------|
| SD 2.0 | 768x768 | OpenCLIP ViT-H/14 |
| SD 2.1 | 768x768 | OpenCLIP ViT-H/14 |
| SD 2.0-base | 512x512 | OpenCLIP ViT-H/14 |

**Resources:**
- [stabilityai/stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2)
- [Stability AI Blog](https://stability.ai/blog/stable-diffusion-v2-release)

### Stable Cascade

Uses Würstchen architecture with a latent cascade approach - generates in a highly compressed latent space, then upscales.

**Resources:**
- [stabilityai/stable-cascade](https://huggingface.co/stabilityai/stable-cascade)
- [Würstchen Paper](https://arxiv.org/abs/2306.00637)

---

## DiT-based

These use Diffusion Transformer (DiT) architectures instead of UNet. Implementing these would require new backbone code.

### Flux

From Black Forest Labs (original Stable Diffusion team). Uses flow matching instead of diffusion, with a DiT backbone.

| Variant | Params | License |
|---------|--------|---------|
| Flux.1 [dev] | 12B | Non-commercial |
| Flux.1 [schnell] | 12B | Apache 2.0 |
| Flux.1 [pro] | 12B | API only |

**Resources:**
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Black Forest Labs](https://blackforestlabs.ai/)

### Stable Diffusion 3 / 3.5

Uses MMDiT (Multimodal Diffusion Transformer) with rectified flow. Triple text encoder (CLIP, OpenCLIP, T5).

| Variant | Params | License |
|---------|--------|---------|
| SD3 Medium | 2B | Community |
| SD3.5 Large | 8B | Community |
| SD3.5 Large Turbo | 8B | Community |

**Resources:**
- [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- [SD3 Paper](https://arxiv.org/abs/2403.03206)

### Z-Image

From Alibaba (Tongyi-MAI). Uses single-stream DiT (S3-DiT) architecture - all tokens flow through one transformer stack.

| Variant | Params | Notes |
|---------|--------|-------|
| Z-Image-Base | 6B | Foundation model |
| Z-Image-Turbo | 6B | Distilled, 8 NFE, sub-second |
| Z-Image-Edit | 6B | Image editing |

Key features:
- Sub-second inference on H800
- Fits in 16GB VRAM
- Bilingual text rendering (English/Chinese)
- #1 open-source on Artificial Analysis leaderboard (Dec 2025)

**Resources:**
- [GitHub](https://github.com/Tongyi-MAI/Z-Image)
- [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Paper](https://arxiv.org/abs/2511.22699)
- [Blog](https://tongyi-mai.github.io/Z-Image-blog/)

### Qwen-Image

From Alibaba (Qwen team). Uses MMDiT with Qwen2.5-VL as text encoder.

| Variant | Params | Notes |
|---------|--------|-------|
| Qwen-Image | 20B | Initial release |
| Qwen-Image-2512 | 20B | Dec 2025, improved realism |

Key features:
- Strong text rendering (English and Chinese)
- Supports T2I and image editing
- Object detection, segmentation, depth estimation
- Apache 2.0 license

**Resources:**
- [GitHub](https://github.com/QwenLM/Qwen-Image)
- [Hugging Face](https://huggingface.co/Qwen/Qwen-Image)
- [Paper](https://arxiv.org/abs/2508.02324)
- [Blog](https://qwenlm.github.io/blog/qwen-image/)

### PixArt-α / PixArt-Σ

Efficient DiT architecture with T5 text encoder. Notable for training efficiency.

| Variant | Params | Resolution |
|---------|--------|------------|
| PixArt-α | 600M | 1024x1024 |
| PixArt-Σ | 600M | 4K |

**Resources:**
- [PixArt-alpha/PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma)
- [PixArt-α Paper](https://arxiv.org/abs/2310.00426)
- [PixArt-Σ Paper](https://arxiv.org/abs/2403.04692)

### AuraFlow

Open source flow-based model. Fully open weights and training code.

**Resources:**
- [fal/AuraFlow](https://huggingface.co/fal/AuraFlow)
- [Blog](https://blog.fal.ai/auraflow/)

### Hunyuan-DiT

Bilingual (Chinese/English) DiT model from Tencent.

| Variant | Params |
|---------|--------|
| Hunyuan-DiT | 1.5B |

**Resources:**
- [Tencent/HunyuanDiT](https://huggingface.co/Tencent/HunyuanDiT)
- [Paper](https://arxiv.org/abs/2405.08748)
- [GitHub](https://github.com/Tencent/HunyuanDiT)

---

## Implementation Priority

Suggested order based on impact and implementation complexity:

1. **SD 2.x** - Minimal changes, reuses existing UNet
2. **Z-Image** - Efficient, fast, good open-source option
3. **Flux** - Very popular, but larger model
4. **SD3/3.5** - Natural evolution of SD line
5. **Qwen-Image** - Large but Apache 2.0
6. **PixArt** - Small and efficient
7. **Stable Cascade** - Different approach, interesting
8. **Others** - As demand arises
