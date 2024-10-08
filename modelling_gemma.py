import torch
from torch import nn
from typing import Tuple, Optional, List
from torch.nn import CrossEntropyLoss
import math
from modelling_siglip import SigLipVisionConfig, SigLipVisionModel


class GemmaConfig():

    def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim = 256,
            max_position_embeddings = 8192,
            rms_norm_eps = 1e-6,
            rope_theta = 10000.0,
            attention_bias = False,
            attention_dropout = 0.0,
            pad_token_id = None,
            **kwargs
    ):
        super().__init()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_layers = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id



class PaliGemmaConfig():
    
    def __init__(
            self,
            vision_config = None,
            text_config = None,
            ignore_index = -100,
            image_token_index = 256000,
            vocab_size = 257152,
            projection_dim = 2048,
            hidden_size = 2048,
            pad_token_id = None,
            **kwargs
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SigLipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id = pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config : PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias= True)

    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states

class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hiddem_size, config.vocab_size, bias=False)

class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config : PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SigLipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def merge_input_ids_with_image_features(
            self, image_features : torch.Tensor, inputs_embeds : torch.Tensor, input_ids : torch.Tensor, attention_mask : torch.Tensor, kv_cache : Optional[KVCache] = None
    ):
        _, _, embed_dim = image_features.shape
        batch_size, seq_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        #scale the image features to shape = [batch_size, seq_length, hidden_size]
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)
        #combine image and text embedding tokens and mask out the placehodler or paddings tokens
        final_embedding = torch.zeros(batch_size, seq_length, embed_dim, dtype= inputs_embeds.dtype, device=inputs_embeds.device)

        #[batch_size,seq_length] true for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        #[batch_size,seq_length] true for image tokens
        image_mask = input_ids == self.config.image_token_index
        #[batch_size,seq_length] true for padding tokens
        pad_mask = input_ids == self.pad_token_id 

        #we neeeeed to expand these, otherwise we cant use these in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)

        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)

        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)
        
        #CREATING THE ATTENTION MASK

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )

        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len

            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )

        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]

            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids       


    def forward(
            self,
            input_ids : torch.LongTensor = None,
            pixel_values : torch.LongTensor = None,
            attention_mask : Optional[torch.tensor] = None,
            kv_cache : Optional[KVCache] = None
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        #extracting the input embeddings, size [batch_size, seq_length, hidden_size]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        #merging the text and images, size [batch_size, channel, width, height] to [batch_size, num_patches, embed_dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        
        #convert from [batch_size, num_patches, embed_dim] to [batch_size, num_patches, hidden_size]
        image_features = self.multi_modal_projector(selected_image_feature)

        #merge the text and image token embeddings
        inputs_embeds, attention_mask, position_ids = self.merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            kv_cache = kv_cache
        )

        return outputs
