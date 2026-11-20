"""Generate one offline GPU inpainting candidate inside the owned job process group."""

import json
import sys
import time
from pathlib import Path


def main(directory):
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import StableDiffusionXLInpaintPipeline

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tralens.rendering.reconstruction_worker import inputs
    from tralens.rendering.reconstruction import pixels_hash

    config, req, rgb, mask = inputs(directory)
    h, w = mask.shape
    y, x = (1024 - h) // 2, (1024 - w) // 2
    pad = ((y, 1024 - h - y), (x, 1024 - w - x))
    image = Image.fromarray(np.pad(rgb, pad + ((0, 0),), mode="symmetric"))
    removal = Image.fromarray(np.pad(mask.astype("uint8") * 255, pad, mode="symmetric"))
    start = time.monotonic()
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        config["model"],
        local_files_only=True,
        use_safetensors=True,
        variant="fp16",
        torch_dtype=torch.float16,
        add_watermarker=False,
    ).to("cuda")
    prompt = (
        "continuous watercolor background, preserve surrounding colors and texture"
        if np.mean(np.ptp(rgb, axis=2) > 12) > 0.05
        else "black and white manga, continuous pen lines, matching screentone"
    )
    params = req["parameters"]
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt="words, lettering, watermark, new objects",
            image=image,
            mask_image=removal,
            height=1024,
            width=1024,
            strength=params["strength"],
            guidance_scale=params["guidance"],
            num_inference_steps=params["steps"],
            generator=torch.Generator(device="cuda").manual_seed(req["seed"]),
        ).images[0]
    result = np.asarray(out.convert("RGB"))[y : y + h, x : x + w].copy()
    Image.fromarray(result).save(directory / "output.png")
    (directory / "result.json").write_text(
        json.dumps(
            dict(
                backend=config["backend"],
                model_sha256=config["sha256"],
                request_revision=req["revision"],
                output_pixels_sha256=pixels_hash(result),
                gpu=torch.cuda.get_device_name(),
                inference_seconds=time.monotonic() - start,
            )
        )
    )
