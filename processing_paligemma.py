from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
import torch
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def rescale(
        image: np.ndarray, scale : float, dtype : np.dtype = np.float32
    ) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)

    return rescaled_image

def resize(
        image : Image,
        size : Tuple[int, int],
        resample : Image.Resampling = None,
        reducing_gap : Optional[int] = None,
) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample = resample, reducing_gap = reducing_gap
    )
    return resized_image

def normalize(
        image : np.ndarray,
        mean : Union[float, Iterable[float]],
        std : Union[float, Iterable[float]]
) -> np.ndarray:
    mean = np.array(mean, dtype= image.dtype)
    std = np.array(std, dtype= image.dtype)

    image = (image - mean) / std
    return image

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    #input text is tokenzied normally, bos token and newline character is added, as newline is a essential part of the training for the model, apparently.
    #tokenized text is also prefixed with a fixed number of <image> tokens (^ ofcourse, \n and whitespaces and everything is tokenized separately)

    #THIS IS THE HUGGINGFACE IMPLEMENTATION (thank you hf <3)
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def process_images(
        image : List[Image.Image],
        size : Dict[str, int] = None,
        resample : Image.Resampling = None,
        rescale_factor : float = None,
        image_mean : Optional[Union[float, List[float]]] = None,
        image_std : Optional[Union[float, List[float]]] = None,
    ) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [resize(image = image, size = (height, width), resample = resample) for image in images]

    #convert each to an np array
    images = [np.array(image) for image in images]
    #rescale pixel values to be between 0 and 1
    images = [rescale(image, scale = rescale_factor) for image in images]
    #normalize to have mean 0 and std dev 1
    images = [normalize(image, mean = image_mean, std = image_std) for image in images]
    #change the dimensions as per the model expectations ([])
    images = [image.transpose(2, 0, 1) for image in images]

    return images

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
