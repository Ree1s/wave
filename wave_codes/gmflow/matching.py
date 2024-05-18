import torch
import torch.nn.functional as F

from .geometry import coords_grid, generate_window_grid, normalize_coords


def global_correlation_softmax(feature0, feature1,
                               pred_bidir_flow=False,
                               ):
    
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  
    feature1 = feature1.view(b, c, -1)  

    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)  

    
    init_grid = coords_grid(b, h, w).to(correlation.device)  
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  

    correlation = correlation.view(b, h * w, h * w)  

    if pred_bidir_flow:
        correlation = torch.cat((correlation, correlation.permute(0, 2, 1)), dim=0)  
        init_grid = init_grid.repeat(2, 1, 1, 1)  
        grid = grid.repeat(2, 1, 1)  
        b = b * 2

    prob = F.softmax(correlation, dim=-1)  

    correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)  

    
    flow = correspondence - init_grid

    return flow, prob


def local_correlation_softmax(feature0, feature1, local_radius,
                              padding_mode='zeros',
                              ):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  
    sample_coords = coords.unsqueeze(-2) + window_grid  

    sample_coords_softmax = sample_coords

    
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  

    valid = valid_x & valid_y  

    
    sample_coords_norm = normalize_coords(sample_coords, h, w)  
    window_feature = F.grid_sample(feature1, sample_coords_norm,
                                   padding_mode=padding_mode, align_corners=True
                                   ).permute(0, 2, 1, 3)  
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  

    
    corr[~valid] = -1e9

    prob = F.softmax(corr, -1)  

    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(
        b, h, w, 2).permute(0, 3, 1, 2)  

    flow = correspondence - coords_init
    match_prob = prob

    return flow, match_prob
