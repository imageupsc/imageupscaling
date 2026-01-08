import torch
from diffusers import StableDiffusionPipeline

class ImageGenerator:
    def __init__(self, device="cuda"):
        model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(device)

    def generate(self, prompt: str):
        image = self.pipe(prompt).images[0]
        return image
