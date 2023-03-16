import json
from itsdangerous import base64_encode
import io
import base64
from diffusers import StableDiffusionImg2ImgPipeline
import torch
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import glob 
from typing import Optional, List, Union, Dict
from fastapi import Body, FastAPI
from pydantic import BaseModel
from img2img import multiple_rounds_img2img

app = FastAPI()

class Post(BaseModel):
  initial_image: str
  prompt: str
  negative_prompt: str
  final_images_to_return: int
  seed: int

# Convert Images to Byestring
def img_to_bytestring(img: Image.Image) -> List[str]:
  # Convert generated image to bytestring
  buffered = io.BytesIO()
  img.save(buffered, format = "PNG")
  imgs_data = buffered.getvalue()
  imgs_b64 = base64.b64encode(imgs_data).decode('utf-8')

  return imgs_b64

  
@app.post("/")
def generate_image_direct(payload: Post) -> Dict[str, List[str]]:

  # Load the image data into a PIL Image object
  image_data = base64.b64decode(payload.initial_image)
  img = Image.open(io.BytesIO(image_data))

  returned_imgs = multiple_rounds_img2img(
  init_image = img,
  prompt = payload.prompt,
  negative_prompt = payload.negative_prompt,
  strength_array = [0.7, 0.6, 0.5, 0.4],
  guidance_array = [20.0, 18.0, 16.0, 14.0],
  final_images_to_return = payload.final_images_to_return,
  num_rounds = 4,
  seed = payload.seed)

  # Convert generated image to bytestring
  generated_img_bytestring = [img_to_bytestring(img) for img in returned_imgs]

  return {"generated_images": generated_img_bytestring}