import torch
from PIL.Image import Image
from pipelines.models import TextToImageRequest
from torch import Generator
from diffusers import StableDiffusionXLPipeline


def load_pipeline() -> StableDiffusionXLPipeline:
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "./models/newdream-sdxl-21",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda", torch.float16)
    #pipeline(prompt="")

    # Warmup
    with torch.inference_mode():
        pipeline("")


    return pipeline


def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=30,
        generator=generator,
    ).images[0]
