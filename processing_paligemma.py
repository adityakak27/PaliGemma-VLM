from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
import torch
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]



#for this processor, the image size is 224x224, and the model generates 128 tokens for the text description

#this class loads image, rezises and whatever, and adds placeholders for image tokens
class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens : int, image_size : int):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        #more info about the following token-based code and stuff at: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        #the tokenizer needs some special tokens; we add them here
        tokens_to_add = {"additional_special_tokens" : [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [
            f"‹loc{i:04d}>" for i in range (1024) #location tokens
        ]
        #these tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}›" for i in range (128)
        ]
        #these are used for object segmentation

        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_eos_tokens = False
        tokenizer.add_bos_tokens = False #will add eos and bos tokens separately

        self.tokenizer = tokenizer

    
    def __call__(self, text : List[str], images : List[Image.Image], padding : str = "longest", truncation : bool = True) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Recieved {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(
            images,
            size = (self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1 / 255.0,
            image_mean = IMAGENET_STANDARD_MEAN, #normalising
            image_std = IMAGENET_STANDARD_STD
        )
        #convert list of arrays into single numpy array with shape [batch_size, channel, height, width]
        pixel_values = np.stack(pixel_values, axis=0)
        #now we convert the np array into a tensor
        pixel_values = torch.tensor(pixel_values)

        #create the 'input' to the model
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_length,
                image_token = self.IMAGE_TOKEN
            )
            for prompt in text
        ]

        #tokenize all this using the placeholder tokens
        inputs = self.tokenizer(
            input_strings,
            return_tensors = "pt",
            padding = padding,
            truncation = truncation
        )

        return_data = {"pixel_values" : pixel_values , **inputs}

        return return_data

