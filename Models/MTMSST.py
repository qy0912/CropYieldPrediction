import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Attention_a(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# cross attention transformer

class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, Attention_a(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, Attention_a(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim = 1)
        return sm_tokens, lg_tokens

# multi-scale encoder

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params),
                Transformer(dim = lg_dim, dropout = dropout, **lg_enc_params),
                CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens
        
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 144, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches_small = (image_size // patch_size) ** 2
        num_patches_large = 21        

        patch_dim_small = in_channels * patch_size ** 2
        patch_dim_large = in_channels * 49

        patch_dim_small_long = patch_dim_small*2
        patch_dim_large_long = patch_dim_large*2

        self.to_patch_embedding_small = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim_small, dim),
        )

        self.to_patch_embedding_large = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = 7, p2 = 7),
            nn.Linear(patch_dim_large, dim),
        )
        
        ###
        self.to_patch_embedding_small_long = nn.Sequential(
            Rearrange('b (t n) c (h p1) (w p2) -> b t (h w) (p1 p2 c n)', p1 = patch_size, p2 = patch_size, n=2),
            nn.Linear(patch_dim_small_long, dim),
        )

        self.to_patch_embedding_large_long = nn.Sequential(
            Rearrange('b (t n) c (h p1) (w p2) -> b t (h w) (p1 p2 c n)', p1 = 7, p2 = 7,n=2),
            nn.Linear(patch_dim_large_long, dim),
        )
        ###

        self.pos_embedding_small = nn.Parameter(torch.randn(1, num_frames, num_patches_small + 1, dim))
        self.space_token_small = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer_small = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.pos_embedding_large = nn.Parameter(torch.randn(1, num_frames, num_patches_large + 1, dim))
        self.space_token_large = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer_large = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        ###------
        self.pos_embedding_small_long = nn.Parameter(torch.randn(1, num_frames//2, num_patches_small + 1, dim))
        self.space_token_small_long = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer_small_long = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.pos_embedding_large_long = nn.Parameter(torch.randn(1, num_frames//2, num_patches_large + 1, dim))
        self.space_token_large_long = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer_large_long = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)
        ###------


        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        ###-----
        self.temporal_token_long = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer_long = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)
        ###-----


        self.dropout_s = nn.Dropout(emb_dropout)
        self.dropout_l = nn.Dropout(emb_dropout)
        ###
        self.dropout_s_long = nn.Dropout(emb_dropout)
        self.dropout_l_long = nn.Dropout(emb_dropout)
        ###

        # sm_enc_params = dict(
        #     depth = sm_enc_depth,
        #     heads = sm_enc_heads,
        #     mlp_dim = sm_enc_mlp_dim,
        #     dim_head = sm_enc_dim_head
        # )

        self.short_spatial_cross = CrossTransformer(sm_dim=144, lg_dim=144, depth=3, heads=3, dim_head=64, dropout=0.)
        self.long_spatial_cross = CrossTransformer(sm_dim=144, lg_dim=144, depth=3, heads=3, dim_head=64, dropout=0.)
        self.pool = pool

        self.temporal_cross = CrossTransformer(sm_dim=144, lg_dim=144, depth=3, heads=3, dim_head=64, dropout=0.)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x_s = self.to_patch_embedding_small(x)
        x_l = self.to_patch_embedding_large(x)

        x_s_long = self.to_patch_embedding_small_long(x)
        x_l_long = self.to_patch_embedding_large_long(x)

        #3*3 patched 
        b, t, n, _ = x_s.shape
        cls_space_tokens_s = repeat(self.space_token_small, '() n d -> b t n d', b = b, t=t)
        x_s = torch.cat((cls_space_tokens_s, x_s), dim=2)
        x_s += self.pos_embedding_small[:, :, :(n + 1)]
        x_s = self.dropout_s(x_s)
        x_s = rearrange(x_s, 'b t n d -> (b t) n d')
        x_s = self.space_transformer_small(x_s)
        x_s = rearrange(x_s[:, 0], '(b t) ... -> b t ...', b=b)
        
        # print(x_s.shape)

        #7*7 patched
        b, t, n, _ = x_l.shape
        cls_space_tokens_l = repeat(self.space_token_large, '() n d -> b t n d', b = b, t=t)
        x_l = torch.cat((cls_space_tokens_l, x_l), dim=2)
        x_l += self.pos_embedding_large[:, :, :(n + 1)]
        x_l = self.dropout_l(x_l)
        x_l = rearrange(x_l, 'b t n d -> (b t) n d')
        x_l = self.space_transformer_large(x_l)
        x_l = rearrange(x_l[:, 0], '(b t) ... -> b t ...', b=b)

        # print(x_l.shape)
        
        #3*3 patched long seq
        b, t, n, _ = x_s_long.shape
        cls_space_tokens_s_long = repeat(self.space_token_small_long, '() n d -> b t n d', b = b, t=t)
        x_s_long = torch.cat((cls_space_tokens_s_long, x_s_long), dim=2)
        x_s_long += self.pos_embedding_small_long[:, :, :(n + 1)]
        x_s_long = self.dropout_s_long(x_s_long)
        x_s_long = rearrange(x_s_long, 'b t n d -> (b t) n d')
        x_s_long = self.space_transformer_small_long(x_s_long)
        x_s_long = rearrange(x_s_long[:, 0], '(b t) ... -> b t ...', b=b)
        
        # print(x_s.shape)

        #7*7 patched long seq
        b, t, n, _ = x_l_long.shape
        cls_space_tokens_l_long = repeat(self.space_token_large, '() n d -> b t n d', b = b, t=t)
        x_l_long = torch.cat((cls_space_tokens_l_long, x_l_long), dim=2)
        x_l_long += self.pos_embedding_large_long[:, :, :(n + 1)]
        x_l_long = self.dropout_l_long(x_l_long)
        x_l_long = rearrange(x_l_long, 'b t n d -> (b t) n d')
        x_l_long = self.space_transformer_large(x_l_long)
        x_l_long = rearrange(x_l_long[:, 0], '(b t) ... -> b t ...', b=b)

        x_s,x_l = self.short_spatial_cross(x_s,x_l)
        x_s_long,x_l_long = self.long_spatial_cross(x_s,x_l)

        x_short = x_s + x_l
        x_long = x_s_long + x_s_long
        x_short,x_long = self.temporal_cross(x_short,x_long)


        cat_x = torch.cat((x_short,x_long),dim=1)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, cat_x), dim=1)
        x = self.temporal_transformer(x)


        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)