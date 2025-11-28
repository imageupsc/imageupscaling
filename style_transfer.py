import torch
from PIL import Image
from torchvision import transforms
from style_model import TransformerNet


def load_style_model(style_name: str):
    model = TransformerNet()
    state_dict = torch.load(f"saved_models/{style_name}.pth", map_location="cpu")

    for k in list(state_dict.keys()):
        if "running_mean" in k or "running_var" in k:
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def apply_style(model, image: Image.Image):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))]
    )

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor).cpu()

    output = output.squeeze().clamp(0, 255).permute(1, 2, 0).numpy().astype("uint8")
    return Image.fromarray(output)
