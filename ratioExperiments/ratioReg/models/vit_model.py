'''
@Author: WANG Maonan
@Date: 2021-06-07 21:21:50
@Description: 关于 Vision Transformer 的代码
LastEditTime: 2021-12-11 09:06:49
'''
from vit_pytorch import ViT

def ViT_Regression(patch_size=5):
    ViT_model = ViT(
        image_size = 100,
        patch_size = patch_size,
        num_classes = 1, # nn.Linear(dim, num_classes)
        dim = 32, # Last dimension of output tensor after linear transformation nn.Linear(..., dim)
        depth = 6,
        heads = 8,
        channels = 1,
        mlp_dim = 64,
        dropout = 0, # dropout rate
        emb_dropout = 0 # embedding dropout rate
    )
    return ViT_model