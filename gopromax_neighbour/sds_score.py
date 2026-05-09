"""
SDS Realism Score
=================

Computes a **Score Distillation Sampling (SDS) score** for a single image
given a text prompt.  The score measures how consistent the image is with
the diffusion model's learned distribution p(image | prompt).

    Lower score  →  image is *more* consistent with the prompt / more realistic.
    Higher score →  image deviates from the diffusion prior.

Mathematically, for a single timestep t the contribution is:

    L_SDS(x; y) = E_{t, ε} [ w(t) · ‖ ε̂_φ(z_t; y, t) − ε ‖² ]

where
  z   = VAE(x)                            image latent
  z_t = √ᾱ_t · z + √(1−ᾱ_t) · ε          noised latent at step t
  ε̂_φ = ε_uncond + s·(ε_cond − ε_uncond)  CFG-guided noise prediction
  s   = guidance_scale

The final score is the mean over `num_samples` uniformly-spaced timesteps
in [t_min, t_max].  Using the mid-range (default 0.2–0.8) is most
discriminative; very small t (trivially clean) and very large t (pure noise)
carry less signal about image quality.

Dependencies
------------
    pip install diffusers transformers accelerate

Usage (Python)
--------------
    from sds_score import compute_sds_score
    from PIL import Image

    score = compute_sds_score(
        Image.open("render.png"),
        prompt="A street level image of an outdoor scene",
    )
    print(f"SDS score: {score:.6f}")   # lower = more realistic

Usage (CLI)
-----------
    python sds_score.py \\
        --image  render.png \\
        --prompt "A street level image of an outdoor scene"

    # Optional flags:
    #   --model_id   runwayml/stable-diffusion-v1-5  (default)
    #   --num_samples 16             (timesteps to average over)
    #   --guidance_scale 7.5
    #   --t_min 0.2  --t_max 0.8    (fractional, in [0,1])
    #   --device cuda
    #   --dtype float16              (float16 | float32; float16 saves VRAM)
"""

from __future__ import annotations

import argparse
import os
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy model cache  (avoid reloading on repeated calls within a session)
# ---------------------------------------------------------------------------
_SD_CACHE: dict = {}


def load_sd_components(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    token: str | None = None,
) -> dict:
    """Load (or return cached) VAE, UNet, text encoder, tokenizer, scheduler.

    The first call downloads weights from HuggingFace Hub (~5 GB) and may
    take a minute.  Subsequent calls within the same Python process return
    immediately from the in-memory cache.

    Parameters
    ----------
    model_id:
        Any Stable Diffusion 2.x model from the HuggingFace Hub.
        ``runwayml/stable-diffusion-v1-5`` is recommended for 512 px
        images (lower VRAM than SDXL, well-suited to outdoor photography).
    device:
        ``"cuda"`` or ``"cpu"``.  GPU is strongly recommended.
    dtype:
        ``torch.float16`` (default, saves ~2 GB VRAM) or ``torch.float32``.
    token:
        HuggingFace access token.  Falls back to the ``HF_TOKEN`` environment
        variable when ``None``.  Required for gated / private repositories.
    """
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache_key = (model_id, device, dtype)
    if cache_key in _SD_CACHE:
        return _SD_CACHE[cache_key]

    # Import here so the file can be imported without diffusers installed
    # (the ImportError will only surface when load_sd_components is actually
    # called, giving a clear error message).
    try:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:
        raise ImportError(
            "diffusers and transformers are required.\n"
            "Install with:  pip install diffusers transformers accelerate"
        ) from exc

    print(f"[sds_score] Loading '{model_id}' …  (first call may take a minute)")
    _kw = dict(token=token) if token else {}

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **_kw)
    text_encoder = (
        CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **_kw)
        .to(device=device, dtype=dtype)
        .eval()
    )
    vae = (
        AutoencoderKL.from_pretrained(model_id, subfolder="vae", **_kw)
        .to(device=device, dtype=dtype)
        .eval()
    )
    unet = (
        UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **_kw)
        .to(device=device, dtype=dtype)
        .eval()
    )
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", **_kw)

    # Freeze all parameters — we only use these models for inference.
    for model in (text_encoder, vae, unet):
        for p in model.parameters():
            p.requires_grad_(False)

    components = dict(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
    )
    _SD_CACHE[cache_key] = components
    print("[sds_score] Model loaded and cached.")
    return components


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preprocess_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    device: str,
    dtype: torch.dtype,
    size: int = 512,
) -> torch.Tensor:
    """Convert any image input to a (1, 3, H, W) tensor in [-1, 1]."""
    if isinstance(image, torch.Tensor):
        # Accept (C, H, W) or (1, C, H, W); assume [0, 1] range
        img = image.float()
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
        img = img * 2.0 - 1.0
    else:
        if isinstance(image, np.ndarray):
            # HWC uint8 or float
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(image)
        # PIL.Image path
        image = image.convert("RGB").resize((size, size), Image.LANCZOS)
        arr = np.array(image, dtype=np.float32) / 255.0          # [0, 1]
        img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

    return img.to(device=device, dtype=dtype)


def _encode_prompt(
    prompt: str,
    tokenizer,
    text_encoder,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a (2, seq_len, dim) tensor: [unconditional, conditional]."""
    def _tokenize(text: str):
        return tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

    with torch.no_grad():
        cond_emb   = text_encoder(_tokenize(prompt))[0].to(dtype)
        uncond_emb = text_encoder(_tokenize(""))[0].to(dtype)

    # shape: (2, seq_len, hidden_dim)
    return torch.cat([uncond_emb, cond_emb], dim=0)


def _get_timestep_indices(
    scheduler,
    t_min: float,
    t_max: float,
    num_samples: int,
) -> torch.Tensor:
    """Map fractional [t_min, t_max] to integer scheduler step indices.

    The DDPM scheduler has `num_train_timesteps` steps (typically 1000).
    We return `num_samples` evenly-spaced indices in the requested range.
    """
    T = scheduler.config.num_train_timesteps  # typically 1000
    lo = int(t_min * T)
    hi = int(t_max * T)
    indices = torch.linspace(lo, hi, num_samples).long().clamp(0, T - 1)
    return indices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_sds_score(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    prompt: str,
    *,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    num_samples: int = 32,
    guidance_scale: float = 15.0,
    t_min: float = 0.2,
    t_max: float = 0.8,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    image_size: int = 512,
    sd_components: dict | None = None,
) -> float:
    """Compute the SDS realism score for a single image.

    Lower values indicate the image is more consistent with the diffusion
    model's learned distribution for the given prompt (i.e. more realistic).

    Parameters
    ----------
    image:
        The image to evaluate.  Accepts:
          - ``PIL.Image``
          - ``numpy.ndarray``  shape (H, W, 3), uint8 or float32 in [0, 1]
          - ``torch.Tensor``   shape (C, H, W) or (1, C, H, W), float in [0, 1]
    prompt:
        Text describing the expected scene content, e.g.
        ``"A street level image of an outdoor scene"``.
    model_id:
        HuggingFace model ID for the Stable Diffusion model.
        Default: ``runwayml/stable-diffusion-v1-5`` (512 px native).
    num_samples:
        Number of uniformly-spaced timesteps to average the SDS loss over.
        More samples → more stable score but slower.  32 is a good default.
    guidance_scale:
        Classifier-free guidance scale.  Only affects the UNet call; the
        score is the guidance *delta* magnitude, so this parameter is unused
        in the loss but kept for API compatibility.
    t_min, t_max:
        Fractional range of the diffusion timestep axis to sample from,
        in [0.0, 1.0].  The mid-range (0.2, 0.8) avoids degenerate extremes
        (trivially clean vs. pure noise) and is the most discriminative.
    device:
        ``"cuda"`` or ``"cpu"``.
    dtype:
        ``torch.float16`` (default) or ``torch.float32``.
    image_size:
        Resize the image to this square resolution before encoding.
        Should match the model's native resolution (512 for SD 2.1-base).
    sd_components:
        Pre-loaded components dict from ``load_sd_components()``.  Pass this
        when calling ``compute_sds_score`` in a loop to avoid reloading the
        model on every call.

    Returns
    -------
    float
        Mean guidance delta magnitude `‖ε_cond − ε_uncond‖²` averaged over
        timesteps.  Lower = image is more consistent with the prompt;
        higher = image deviates from the prompt.
    """
    # --- Load (or reuse) model components -----------------------------------
    if sd_components is None:
        sd_components = load_sd_components(model_id, device, dtype)

    tokenizer    = sd_components["tokenizer"]
    text_encoder = sd_components["text_encoder"]
    vae          = sd_components["vae"]
    unet         = sd_components["unet"]
    scheduler    = sd_components["scheduler"]

    # --- Preprocess image → latent ------------------------------------------
    img_tensor = _preprocess_image(image, device, dtype, size=image_size)

    # Encode with frozen VAE; scale by VAE scaling factor (0.18215 for SD)
    latent_dist = vae.encode(img_tensor).latent_dist
    z = latent_dist.mean * vae.config.scaling_factor        # (1, 4, H/8, W/8)

    # --- Encode text prompt -------------------------------------------------
    # text_embeds: (2, seq_len, dim)  [uncond, cond]
    text_embeds = _encode_prompt(prompt, tokenizer, text_encoder, device, dtype)
    # Expand to match batch: (2, seq_len, dim)  (already correct for batch=1)

    # --- Gather scheduler alphas --------------------------------------------
    # alphas_cumprod: shape (T,), float32
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

    # --- Timestep indices ---------------------------------------------------
    t_indices = _get_timestep_indices(scheduler, t_min, t_max, num_samples)

    # --- Accumulate SDS loss ------------------------------------------------
    total_loss = 0.0

    for t_idx in t_indices:
        t_scalar = t_idx.item()

        # Sample noise
        noise = torch.randn_like(z)

        # Form noisy latent: z_t = sqrt(ᾱ_t) * z + sqrt(1 - ᾱ_t) * ε
        alpha_bar = alphas_cumprod[t_idx]                    # scalar
        z_t = (alpha_bar ** 0.5) * z + ((1.0 - alpha_bar) ** 0.5) * noise

        # Timestep tensor for UNet (batch of 2: [uncond, cond])
        t_tensor = torch.tensor(
            [t_scalar, t_scalar], device=device, dtype=torch.long
        )

        # Latent batch: repeat z_t for uncond + cond
        z_t_batch = z_t.repeat(2, 1, 1, 1)                  # (2, 4, H/8, W/8)

        # UNet forward pass (single batched call for both cond and uncond)
        noise_pred = unet(z_t_batch, t_tensor, encoder_hidden_states=text_embeds).sample

        # Classifier-free guidance components
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

        # Prompt-sensitive score: guidance delta magnitude
        # ‖ε_cond − ε_uncond‖²  measures how hard the model corrects the
        # noised image toward the prompt.  This is the only prompt-dependent
        # term; the alternative ‖ε̂_CFG − ε‖² is dominated by the
        # image-quality / unconditional component and is not discriminative.
        #   Low  → image already consistent with prompt (little correction needed)
        #   High → image deviates from prompt (heavy correction applied)
        guidance_delta = noise_pred_cond - noise_pred_uncond   # (1, 4, H/8, W/8)
        step_loss = (guidance_delta ** 2).mean().item()
        total_loss += step_loss

    return total_loss / num_samples


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the SDS realism score for an image.\n"
            "Lower score = image is more consistent with the diffusion prior\n"
            "              (i.e. more photorealistic for the given prompt)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the image file (PNG / JPG / any PIL-readable format).",
    )
    parser.add_argument(
        "--prompt", required=True,
        help='Text prompt, e.g. "A street level image of an outdoor scene".',
    )
    parser.add_argument(
        "--model_id", default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model ID for the Stable Diffusion model.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=32,
        help="Number of timesteps to average over (default: 32).",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=15.0,
        help="Classifier-free guidance scale (default: 15.0).",
    )
    parser.add_argument(
        "--t_min", type=float, default=0.2,
        help="Lower bound of the fractional timestep range (default: 0.2).",
    )
    parser.add_argument(
        "--t_max", type=float, default=0.8,
        help="Upper bound of the fractional timestep range (default: 0.8).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device: "cuda" or "cpu" (default: cuda if available).',
    )
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "float32"],
        help="Model precision (default: float16, saves VRAM).",
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Resize image to this square resolution (default: 512).",
    )
    parser.add_argument(
        "--hf_token", default=None,
        help="HuggingFace access token for gated models. "
             "Falls back to HF_TOKEN env var when not set.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    image = Image.open(args.image).convert("RGB")

    print(f"[sds_score] Image : {os.path.abspath(args.image)}")
    print(f"[sds_score] Prompt: {args.prompt!r}")
    print(f"[sds_score] Model : {args.model_id}")
    print(
        f"[sds_score] Config: num_samples={args.num_samples}, "
        f"guidance_scale={args.guidance_scale}, "
        f"t_range=({args.t_min}, {args.t_max}), "
        f"device={args.device}, dtype={args.dtype}"
    )

    components = load_sd_components(args.model_id, args.device, dtype,
                                    token=args.hf_token)

    score = compute_sds_score(
        image,
        args.prompt,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        t_min=args.t_min,
        t_max=args.t_max,
        device=args.device,
        dtype=dtype,
        image_size=args.image_size,
        sd_components=components,
    )

    print(f"\nSDS score: {score:.6f}")
    print("(lower = image more consistent with prompt; higher = image deviates from prompt)")


if __name__ == "__main__":
    main()
