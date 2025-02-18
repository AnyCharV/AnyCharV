import torch.nn as nn
from .xf import LayerNorm, Transformer
from transformers.models.clip.modeling_clip import CLIPVisionModelOutput
from transformers import CLIPVisionModelWithProjection


# clip model with MLP
class CLIPVisionModelMLP(nn.Module):
    def __init__(self, clip_model, mlp_num=5):
        super().__init__()

        self.clip_model = clip_model
        
        self.dtype = self.clip_model.dtype
        self.device = self.clip_model.device
        
        self.mapper = Transformer(n_ctx=1, width=1024, layers=mlp_num, heads=1)
        self.final_ln = LayerNorm(1024)

    def forward(self, x):
        x = self.clip_model(x).image_embeds
        # print(x.shape)

        # print('before', x.dtype)
        x = x.unsqueeze(1).to(self.mapper.dtype)
        # print('after', x.dtype)

        x = self.mapper(x)
        x = self.final_ln(x)
        # print(x.shape)

        x = x.squeeze(1)

        return CLIPVisionModelOutput(image_embeds=x)
