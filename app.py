import argparse
import os
import pdb

import cv2
import gradio as gr
import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image, ImageOps
from transformers import CLIPTextModel, DPTFeatureExtractor, DPTForDepthEstimation,CLIPVisionModelWithProjection
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from models import BrushNetModel, UNet2DConditionModel, ImageProjModel
from diffusion_pipeline import StableDiffusionITGOPipeline
from lavis.models.blip2_models.blip2_opt_mutivison import Blip2OPT_Mutivison

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)

def pad_image_and_create_mask(image, up, down, left, right):
    """
    Pads an image and creates a corresponding mask where the original image area is 255 and the pad area is 0.

    :param image: PIL.Image object, the original image to pad
    :param up: int, number of pixels to pad on the top
    :param down: int, number of pixels to pad on the bottom
    :param left: int, number of pixels to pad on the left
    :param right: int, number of pixels to pad on the right
    :return: tuple of two PIL.Image objects, the padded image and the corresponding mask
    """
    # Padding the image
    padded_image = ImageOps.expand(image, border=(left, up, right, down), fill=(0,0,0)).convert('RGB')

    # Creating the mask for the original image area
    # The size of the mask is the same as the padded image
    mask = Image.new("L", padded_image.size, 255)
    mask.paste(0, (left, up, left + image.width, up + image.height))

    return padded_image, mask

class AutoPaintController:
    def __init__(
        self, pretrained_model_path,  base_model_path=None, weight_dtype=torch.float16,
    ) -> None:
        self.weight_dtype=weight_dtype
        self.pretrained_model_path = pretrained_model_path
        self.base_model_path = base_model_path
        self.vlm = Blip2OPT_Mutivison(
        vit_model="eva_clip_g",
        img_size= 224,
        drop_path_rate=0,
        fuse_feature_dim=1408,
        use_grad_checkpoint=False,
        mlp_layers= 2,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="/home/iv/Intern_new/ChenBin/outpainting/LAVIS/huggingface_model/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    )
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224,224), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )
        checkpoint = torch.load(os.path.join(pretrained_model_path, 'VLM.pth'),map_location='cpu')
        self.vlm.load_state_dict(checkpoint)
        self.vlm.to('cuda')
        torch.set_grad_enabled(False)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained('/home/iv/Intern_new/ChenBin/outpainting/PowerPaint-dev/models/image_encoder')
        # brushnet = BrushNetModel.from_unet(unet)
        ckpt = torch.load(os.path.join(pretrained_model_path, 'mlp.pth'), map_location='cpu')
        # brushnet.load_state_dict(ckpt['brushnet'])
        mlp_ca = ImageProjModel(768, 1024, num_hidden_layers=2)
        mlp_ca.load_state_dict(ckpt["mlp_ca"])
        mlp_ca = mlp_ca.to('cuda')

        self.pipeline = StableDiffusionITGOPipeline.from_pretrained(
            base_model_path,
            unet=UNet2DConditionModel.from_pretrained(
                base_model_path,
                subfolder="unet",
                local_files_only=True,
                torch_dtype=weight_dtype,
            ).to("cuda"),
            brushnet=BrushNetModel.from_pretrained(
                pretrained_model_path,
                torch_dtype=weight_dtype,
            ),
            mlp_ca=mlp_ca,
            image_encoder=image_encoder,
            torch_dtype=weight_dtype,
        )
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline = self.pipeline.to('cuda')
        self.pipeline.set_progress_bar_config(disable=False)

    def predict(
        self,
        negative_prompt,
        use_manual_prompt,
        prompt,
        input_image,
        up_expand_pixel=1,
        down_expand_pixel=1,
        left_expand_pixel=1,
        right_expand_pixel=1,
        ddim_steps=45,
        scale=7.5,
        seed=24,
    ):
        # pdb.set_trace()
        # image, mask = input_image["image"].convert("RGB"), input_image["mask"].convert("RGB")
        if use_manual_prompt=='Use manual prompt':
            vlm_image = self.transform(input_image)
            vlm_image = vlm_image.unsqueeze(0).to('cuda',dtype=self.weight_dtype)
            prompt = self.vlm.generate({"image":vlm_image})[0]
        else:
            prompt = prompt
        print(prompt)
        # pad image for outpainting
        masked_image, mask = pad_image_and_create_mask(input_image,up_expand_pixel,down_expand_pixel,left_expand_pixel,right_expand_pixel)
        #resize into multiples of 8
        w, h = masked_image.size
        w, h = w // 8 * 8, h // 8 * 8
        masked_image = masked_image.resize((w, h))
        mask = mask.resize((w, h))
        result = self.pipeline(
            prompt,
            masked_image,
            input_image,
            mask,
            num_inference_steps=ddim_steps,
            generator= torch.Generator(device='cuda').manual_seed(seed),
            guidance_scale=scale,
            negative_prompt=negative_prompt,
        ).images[0]
        # paste the inpainting results into original images
        # dict_res = [result, mask]
        return  prompt , [result], [mask]


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--pretrained_model_path", type=str, required=True)
    args.add_argument("--base_model_path", type=str, default=None)
    args.add_argument("--weight_dtype", type=str, default="float16")
    args.add_argument("--share", action="store_true")
    args.add_argument(
        "--local_files_only", action="store_true", help="enable it to use cached files without requesting from the hub"
    )
    args.add_argument("--port", type=int, default=7860)
    args = args.parse_args()
    if args.base_model_path is None:
        args.base_model_path = "runwayml/stable-diffusion-v1-5"
    return args


if __name__ == "__main__":
    args = parse_args()

    # initialize the pipeline controller
    weight_dtype = torch.float16 if args.weight_dtype == "float16" else torch.float32
    controller = AutoPaintController(
        pretrained_model_path=args.pretrained_model_path,
        base_model_path=args.base_model_path,
        weight_dtype=weight_dtype,
    )

    # ui
    with gr.Blocks(css="style.css") as demo:
        with gr.Row():
            gr.Markdown(
                "<div align='center'><font size='18'>ITGO: A General Framework for Text-Guided Image Outpainting</font></div>"  # noqa
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input image")
                input_image = gr.Image(sources="upload", type="pil")
                use_manual_prompt = gr.Radio(["Use manual prompt", "Do not use manual prompt"], label="Prompt choice", info="use manual prompt?")
                negative_prompt = gr.Textbox(label="negative_prompt")
                prompt = gr.Textbox(label="prompt")
                up_expand_pixel = gr.Slider(
                    label="up_expand_pixel",
                    minimum=0,
                    maximum=400,
                    step=1,
                    randomize=False,
                )
                down_expand_pixel = gr.Slider(
                    label="down_expand_pixel",
                    minimum=0,
                    maximum=400,
                    step=1,
                    randomize=False,
                )
                left_expand_pixel = gr.Slider(
                    label="left_expand_pixel",
                    minimum=0,
                    maximum=400,
                    step=1,
                    randomize=False,
                )
                right_expand_pixel = gr.Slider(
                    label="right_expand_pixel",
                    minimum=0,
                    maximum=400,
                    step=1,
                    randomize=False,
                )
                run_button = gr.Button(value="Run")
                with gr.Accordion("Advanced options", open=False):
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=45, step=1)
                    scale = gr.Slider(
                        info="For object removal and image outpainting, it is recommended to set the value at 10 or above.",
                        label="Guidance Scale",
                        minimum=0.1,
                        maximum=30.0,
                        value=7.5,
                        step=0.1,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=2147483647,
                        step=1,
                        randomize=True,
                    )
            with gr.Column():
                gr.Markdown("### ouput prompt")
                ouput_prompt=gr.Textbox(label="ouput prompt")
                gr.Markdown("### Outpainting result")
                result = gr.Gallery(label="Generated images", show_label=False, columns=2)
                gr.Markdown("### Mask")
                mask = gr.Gallery(label="Generated masks", show_label=False, columns=2)

        # =========================================
        # passing parameters into function call
        # =========================================
        prefix_args = [
            negative_prompt,
            use_manual_prompt,
            prompt,
            input_image,
            up_expand_pixel,
            down_expand_pixel,
            left_expand_pixel,
            right_expand_pixel,
            ddim_steps,
            scale,
            seed,
        ]

        def update_click(
                negative_prompt,
                use_manual_prompt,
                prompt,
                input_image,
                up_expand_pixel,
                down_expand_pixel,
                left_expand_pixel,
                right_expand_pixel,
                ddim_steps,
                scale,
                seed,
        ):
            return controller.predict(
                negative_prompt,
                use_manual_prompt,
                prompt,
                input_image,
                up_expand_pixel,
                down_expand_pixel,
                left_expand_pixel,
                right_expand_pixel,
                ddim_steps,
                scale,
                seed,
            )

        # set the buttons
        run_button.click(
            fn=update_click,
            inputs=prefix_args,
            outputs=[ouput_prompt,result,mask],
        )

    demo.queue()
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
