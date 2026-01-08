import torch
from diffusers import StableDiffusionPipeline

class ImageGenerator:
    def __init__(self, device: str | None = None, model_id: str = "runwayml/stable-diffusion-v1-5"):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

        if device == "cuda":
            self.pipe.enable_attention_slicing()

        self.pipe = self.pipe.to(device)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 25,
        guidance: float = 7.5,
        on_progress=None,   # <- функция вида on_progress(step_idx, total_steps)
    ):
        total_steps = int(steps)

        # diffusers обычно передаёт step index с нуля
        def _callback(step: int, timestep: int, latents):
            if on_progress is not None:
                on_progress(step + 1, total_steps)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=total_steps,
            guidance_scale=float(guidance),
            callback=_callback,
            callback_steps=1,
        )
        return result.images[0]
