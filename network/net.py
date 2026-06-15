import torch
import torch.nn as nn
        

class NeRF_MLP(nn.Module):
    
    def __init__(self, input_ch=63, input_ch_views=27, hidden=256, layers=8, skips=[4]):
        super().__init__() 
        
        self.skips = skips
        self.layers = layers
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, hidden)] + 
            [nn.Linear(hidden, hidden) if i not in self.skips 
             else nn.Linear(hidden + input_ch, hidden) for i in range(layers-1)]
        )
        
        self.sigma_out = nn.Linear(hidden, 1)
        
        self.feature_linear = nn.Linear(hidden, hidden)
        
        self.views_linears = nn.Sequential(
            nn.Linear(hidden + input_ch_views, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3) 
        )


    def forward(self, x, d):
        """
        x: positional encoding of coordinates [N_rays, N_samples, 63]
        d: positional encoding of view direction [N_rays, N_samples, 27]
        """
        h = x
        
        #  Passes 8 layers
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = nn.functional.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], dim=-1) 
                
        sigma = self.sigma_out(h)
        
        feature = self.feature_linear(h)
        h = torch.cat([feature, d], dim=-1) 
        rgb = self.views_linears(h)
        
        return torch.cat([rgb, sigma], dim=-1)