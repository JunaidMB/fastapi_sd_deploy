from diffusers import StableDiffusionImg2ImgPipeline
import json
from itsdangerous import base64_encode
import io
import base64
import torch
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, List, Union, Dict


# Load model
img2imgpipe = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/mo-di-diffusion", torch_dtype=torch.float16)
img2imgpipe.to("cuda")


def multiple_rounds_img2img(
  init_image: Image,
  prompt: str,  
  negative_prompt: str,
  strength_array: List[float],
  guidance_array: Union[List[float], List[int]],
  final_images_to_return: Optional[int] = 5,
  num_rounds: Optional[int] = 4,
  seed: Optional[int] = 123) -> List:

  # Parameter checking
  ## init_image
  assert isinstance(init_image, Image.Image), "init_image must be an Image"

  ## prompt & negative_prompt
  assert isinstance(prompt, str) and len(prompt) > 0, "Prompt provided must be a comma separated string and cannot be an empty string" 
  assert isinstance(negative_prompt, str), "Negative Prompt provided must be a comma separated string"

  ## num rounds
  assert num_rounds > 0, "num_rounds must be greater than 0"

  ## strength_array & guidance array
  assert len(strength_array) == num_rounds, 'strength_array length must be identical to num_rounds'
  assert len(guidance_array) == num_rounds, 'guidance_array length must be identical to num_rounds'

  ## final_images_to_return
  assert final_images_to_return > 0, "final_images_to_return must be greater than 0"

  ## seed
  assert isinstance(seed, int), "seed must be an integer"
  
  # Main Body
  torch.manual_seed(seed)
  output_image_array = [init_image]

  for idx in list(range(0, num_rounds - 1)):
    
    img2imgpipeline = img2imgpipe(prompt = prompt,
                          image=output_image_array[idx],
                          strength=strength_array[idx],
                          guidance_scale=guidance_array[idx],
                          num_inference_steps=400,
                          num_images_per_prompt = 1,
                          negative_prompt = negative_prompt)

    output_image_array.append( img2imgpipeline.images[0] )

    # For final round of inference
    torch.manual_seed(seed)
    img2imgpipeline_final = img2imgpipe(prompt = prompt,
                            image=output_image_array[-1],
                            strength=strength_array[-1],
                            guidance_scale=guidance_array[-1],
                            num_inference_steps=400,
                            num_images_per_prompt = final_images_to_return,
                            negative_prompt = negative_prompt)

    return img2imgpipeline_final.images