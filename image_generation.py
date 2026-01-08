import torch
from diffusers import StableDiffusionPipeline

class ImageGenerator:
    def __init__(self, device: str | None = None, model_id: str = "runwayml/stable-diffusion-v1-5"):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

        # небольшой win по памяти/скорости
        if device == "cuda":
            self.pipe.enable_attention_slicing()

        self.pipe = self.pipe.to(device)

    @torch.inference_mode()
    def generate(self, prompt: str, negative_prompt: str = "", steps: int = 25, guidance: float = 7.5):
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
        )
        return result.images[0]
