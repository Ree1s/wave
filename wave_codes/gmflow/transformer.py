import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import split_feature, merge_splits


def single_head_full_attention(q, k, v):
    
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** .5)  
    attn = torch.softmax(scores, dim=2)  
    out = torch.matmul(attn, v)  

    return out


def generate_shift_window_attn_mask(input_resolution, window_size_h, window_size_w,
                                    shift_size_h, shift_size_w, device=torch.device('cuda')):
    
    
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)  
    h_slices = (slice(0, -window_size_h),
                slice(-window_size_h, -shift_size_h),
                slice(-shift_size_h, None))
    w_slices = (slice(0, -window_size_w),
                slice(-window_size_w, -shift_size_w),
                slice(-shift_size_w, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = split_feature(img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True)

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def single_head_split_window_attention(q, k, v,
                                       num_splits=1,
                                       with_shift=False,
                                       h=None,
                                       w=None,
                                       attn_mask=None,
                                       ):
    
    
    assert q.dim() == k.dim() == v.dim() == 3

    assert h is not None and w is not None
    assert q.size(1) == h * w

    b, _, c = q.size()

    b_new = b * num_splits * num_splits

    window_size_h = h // num_splits
    window_size_w = w // num_splits

    q = q.view(b, h, w, c)  
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)

    scale_factor = c ** 0.5

    if with_shift:
        assert attn_mask is not None  
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2

        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

    q = split_feature(q, num_splits=num_splits, channel_last=True)  
    k = split_feature(k, num_splits=num_splits, channel_last=True)
    v = split_feature(v, num_splits=num_splits, channel_last=True)

    scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)
                          ) / scale_factor  

    if with_shift:
        scores += attn_mask.repeat(b, 1, 1)

    attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(attn, v.view(b_new, -1, c))  

    out = merge_splits(out.view(b_new, h // num_splits, w // num_splits, c),
                       num_splits=num_splits, channel_last=True)  

    
    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

    out = out.view(b, -1, c)

    return out


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=1,
                 attention_type='swin',
                 no_ffn=False,
                 ffn_dim_expansion=4,
                 with_shift=False,
                 **kwargs,
                 ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.attention_type = attention_type
        self.no_ffn = no_ffn

        self.with_shift = with_shift

        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                attn_num_splits=None,
                **kwargs,
                ):
        
        query, key, value = source, target, target

        
        query = self.q_proj(query)  
        key = self.k_proj(key)  
        value = self.v_proj(value)  

        if self.attention_type == 'swin' and attn_num_splits > 1:
            if self.nhead > 1:
                
                
                raise NotImplementedError
            else:
                message = single_head_split_window_attention(query, key, value,
                                                             num_splits=attn_num_splits,
                                                             with_shift=self.with_shift,
                                                             h=height,
                                                             w=width,
                                                             attn_mask=shifted_window_attn_mask,
                                                             )
        else:
            message = single_head_full_attention(query, key, value)  

        message = self.merge(message)  
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(self,
                 d_model=256,
                 nhead=1,
                 attention_type='swin',
                 ffn_dim_expansion=4,
                 with_shift=False,
                 **kwargs,
                 ):
        super(TransformerBlock, self).__init__()

        self.self_attn = TransformerLayer(d_model=d_model,
                                          nhead=nhead,
                                          attention_type=attention_type,
                                          no_ffn=True,
                                          ffn_dim_expansion=ffn_dim_expansion,
                                          with_shift=with_shift,
                                          )

        self.cross_attn_ffn = TransformerLayer(d_model=d_model,
                                               nhead=nhead,
                                               attention_type=attention_type,
                                               ffn_dim_expansion=ffn_dim_expansion,
                                               with_shift=with_shift,
                                               )

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                attn_num_splits=None,
                **kwargs,
                ):
        

        
        source = self.self_attn(source, source,
                                height=height,
                                width=width,
                                shifted_window_attn_mask=shifted_window_attn_mask,
                                attn_num_splits=attn_num_splits,
                                )

        
        source = self.cross_attn_ffn(source, target,
                                     height=height,
                                     width=width,
                                     shifted_window_attn_mask=shifted_window_attn_mask,
                                     attn_num_splits=attn_num_splits,
                                     )

        return source


class FeatureTransformer(nn.Module):
    def __init__(self,
                 num_layers=6,
                 d_model=128,
                 nhead=1,
                 attention_type='swin',
                 ffn_dim_expansion=4,
                 **kwargs,
                 ):
        super(FeatureTransformer, self).__init__()

        self.attention_type = attention_type

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model,
                             nhead=nhead,
                             attention_type=attention_type,
                             ffn_dim_expansion=ffn_dim_expansion,
                             with_shift=True if attention_type == 'swin' and i % 2 == 1 else False,
                             )
            for i in range(num_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, feature1,
                attn_num_splits=None,
                **kwargs,
                ):

        b, c, h, w = feature0.shape
        assert self.d_model == c

        feature0 = feature0.flatten(-2).permute(0, 2, 1)  
        feature1 = feature1.flatten(-2).permute(0, 2, 1)  

        if self.attention_type == 'swin' and attn_num_splits > 1:
            
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=feature0.device,
            )  
        else:
            shifted_window_attn_mask = None

        
        concat0 = torch.cat((feature0, feature1), dim=0)  
        concat1 = torch.cat((feature1, feature0), dim=0)  

        for layer in self.layers:
            concat0 = layer(concat0, concat1,
                            height=h,
                            width=w,
                            shifted_window_attn_mask=shifted_window_attn_mask,
                            attn_num_splits=attn_num_splits,
                            )

            
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

        feature0, feature1 = concat0.chunk(chunks=2, dim=0)  

        
        feature0 = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  
        feature1 = feature1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  

        return feature0, feature1


class FeatureFlowAttention(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels,
                 **kwargs,
                 ):
        super(FeatureFlowAttention, self).__init__()

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow,
                local_window_attn=False,
                local_window_radius=1,
                **kwargs,
                ):
        
        if local_window_attn:
            return self.forward_local_window_attn(feature0, flow,
                                                  local_window_radius=local_window_radius)

        b, c, h, w = feature0.size()

        query = feature0.view(b, c, h * w).permute(0, 2, 1)  

        
        
        
        
        
        
        query = self.q_proj(query)  
        key = self.k_proj(query)  

        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)  

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  

        return out

    def forward_local_window_attn(self, feature0, flow,
                                  local_window_radius=1,
                                  ):
        assert flow.size(1) == 2
        assert local_window_radius > 0

        b, c, h, w = feature0.size()

        feature0_reshape = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)
                                       ).reshape(b * h * w, 1, c)  

        kernel_size = 2 * local_window_radius + 1

        feature0_proj = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)

        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size,
                                   padding=local_window_radius)  

        feature0_window = feature0_window.view(b, c, kernel_size ** 2, h, w).permute(
            0, 3, 4, 1, 2).reshape(b * h * w, c, kernel_size ** 2)  

        flow_window = F.unfold(flow, kernel_size=kernel_size,
                               padding=local_window_radius)  

        flow_window = flow_window.view(b, 2, kernel_size ** 2, h, w).permute(
            0, 3, 4, 2, 1).reshape(b * h * w, kernel_size ** 2, 2)  

        scores = torch.matmul(feature0_reshape, feature0_window) / (c ** 0.5)  

        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, flow_window).view(b, h, w, 2).permute(0, 3, 1, 2).contiguous()  

        return out
