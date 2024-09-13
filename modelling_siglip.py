from typing import Optional, Tuple
import torch
import torch.nn as nn

class SigLipVisionConfig:
    def __init__(
            self,
            hidden_size = 768,
            intermediate_size = 3072,
            num_hidden_layers = 12,
            num_attn_heads = 12,
            num_channels = 3, #number of channels for each image (R, G, B)
            image_size = 224,
            patch_size = 16, #each image is divided into 16-sized patches
            layer_norm_eps = 1e-6,
            attn_dropout = 0.0,
            num_image_tokens : int = None,
            **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_heads = num_attn_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attn_dropout = attn_dropout
        self.num_image_tokens = num_image_tokens


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config : SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embd_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embd_dim,
            kernel_size = self.patch_size,
            padding = 'valid', #no padding needed
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2 
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embd_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((-1, 1)),
            persistent = False,
        )

    def forward(self, pixel_values = torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape #batch_size, channels , height, width

        patch_embds = self.patch_embedding(pixel_values)

        embeddings = patch_embds.flatten(2)

        embeddings = embeddings.transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class SigLipAttention(nn.Module):
    #multi head attention mech from 'attention is all you need' paper;

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embd_dim = config.hidden_size
        self.num_heads = config.num_attn_heads
        self.head_dim = self.embd_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 #equals to 1/sqrt(head_dim)
        self.dropout = config.attn_dropout

        self.k_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.v_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim)

    def forward(self, hidden_states : torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # xcurrent shape is [batch_size, num_patches, embd_dim]
        batch_size, seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states) #all of these have the same dimensions as hidden states
        #currently, linear layers are present, so no contextualisation; 
        #each token has no idea about the others; q,k,v are made because the self attention mech has to see each sequence as query, key and value all 3
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) #(4, 1024) x (1024, 8, 128) = (4, 8, 128) basic matrix multiplication
        #IMPORTANT: understand, that at this step, query states is SPLIT into smaller dimension;
        #dimension now, after splitting is [batch_size, num_patches, num_heads, head_dim];
        #embd_dim dimension is SPLIT into (num_heads) number of (head_dim) sizes
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)
        #here, we calculated attention using the formula Q * K ^T / sqrt(d_k)

        if attn_weights.size() != (batch_size, self.num_heads, seq_length, seq_length):
            raise ValueError(
                f"Attention should be of size {(batch_size, self.num_heads, seq_length, seq_length)}, but is {attn_weights.size()}"
            )
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) #'causal attention masking' nahin samajh aaya :(

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        #we do not apply dropout while training; (learnt from andrej karpathy video)

        attn_output = torch.matmul(attn_weights, value_states) #QxK matrix is a lower triangular matrix; each line adds a new token, and the output is a set of contextualised embeddings

        if attn_output.size() != (batch_size, self.num_heads, seq_length, seq_length):
            raise ValueError(
                f"Attention should be of size {(batch_size, self.num_heads, seq_length, self.head_dim)}, but is {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.embd_dim) #something called 'stride', which changes dimension without changing memory allocation and hence overhead computation
        #now, we concatenate all the heads: (4, 8, 128) -> (4, 1024) to get a contextualised token group, as we return to the original shape
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights




class SigLipEncoderLayer(nn.Module):
    def __init__(self, config : SigLipVisionConfig):
        super().__init__()
        self.embd_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embd_dim, eps = config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embd_dim, eps = config.layer_norm_eps)
         

    def forward(self, hidden_states : torch.Tensor) -> torch.Tensor :
        #residual  : [batch_size, num_patches, embd_dim]
        residual = hidden_states
        #[batch_size, num_patches, embd_dim] => [batch_size, num_patches, embd_dim] (does not change dimensions of the vector)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states = hidden_states)
        hidden_states = residual + hidden_states
        #dimensions remain the same throughout
        residual = hidden_states
        #dimensions still remain the same
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SigLipEncoder(nn.Module):
    def __init__(self, config : SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds : torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states) #output of one layer goes into input of the next one

        return hidden_states #return the output of the last layer
    

class SigLipMLP(nn.Module):
    def __init__(self, config : SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states : torch.Tensor) -> torch.Tensor:
        #changes dimension from [batch_size , num_patches , embd_dim] => [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        #dimension is now [batch_size, num_patches, embd_dim]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh") #introduces non linearity
        #dimension is changed from [batch_size, num_patches, intermediate_size] BACK TO [batch_size, num_patches, embd_dim] (recompressed)
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config : SigLipVisionConfig):
        super().__init__()
        self.config = config
        embd_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embd_dim, eps = config.layer_norm_eps)

    def forward(self, pixel_values : torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds = hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state



class SigLipVisionModel(nn.Module):

    def __init__(self, config : SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SigLipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        #[batch size, channels, height, width] => [batch size, patches, embd dim]
        return self.vision_model(pixel_values = pixel_values)
