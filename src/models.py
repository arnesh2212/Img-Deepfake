import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class BC_PCA(nn.Module):
    """
    Band-Conditioned Phased Cross-Attention (batch-first).
    Q: (B, QL, d)
    mag_tokens: list of tensors [(B, TL, d), ...]  (one per band)
    phase_tokens: list of tensors [(B, TL, d), ...]
    returns fused tokens (B, QL, d)
    """
    def __init__(self, d_model, n_bands=3, hidden_dim=None):
        super().__init__()
        self.d = d_model
        self.n_bands = n_bands
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.band_embed = nn.Parameter(torch.randn(n_bands, d_model) * 0.02)
        self.band_logit = nn.Parameter(torch.zeros(n_bands))
        hid = hidden_dim or max(32, d_model // 8)
        self.conf_mlp = nn.Sequential(
            nn.Linear(d_model * 2, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, 1)
        )
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, mag_tokens, phase_tokens):
        B, QL, d = Q.shape
        assert d == self.d
        assert len(mag_tokens) == self.n_bands and len(phase_tokens) == self.n_bands

        Qp = self.q_proj(Q) 
        band_outs = []
        for b in range(self.n_bands):
            M = mag_tokens[b]      # (B, TL, d)
            Phi = phase_tokens[b]  # (B, TL, d)
            Kb = self.k_proj(Phi + self.band_embed[b].unsqueeze(0).unsqueeze(0))  # (B,TL,d)
            Vb = self.v_proj(M)  # (B,TL,d)

            att = torch.einsum('bqd,bkd->bqk', Qp, Kb) / math.sqrt(d)   # (B,QL,TL)
            A = torch.softmax(att, dim=-1)
            Ob = torch.einsum('bqk,bkd->bqd', A, Vb)  # (B,QL,d)
            band_outs.append(Ob)

        s = torch.softmax(self.band_logit, dim=0) 
        O = sum(s[b] * band_outs[b] for b in range(self.n_bands)) 

        pooled_mag = torch.stack([M.mean(dim=1) for M in mag_tokens], dim=1).mean(dim=1)  
        pooled_patch = Q.mean(dim=1)  # (B,d)
        gate_in = torch.cat([pooled_mag, pooled_patch], dim=1)  
        alpha = torch.sigmoid(self.conf_mlp(gate_in)).unsqueeze(1)  

        fused = self.norm(Q + alpha * self.out(O))
        return fused  



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        out = self.norm(query + attn_output)
        return out

class SEBlock(nn.Module):
    def __init__(self, c1, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c1, max(1, c1 // r), 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(max(1, c1 // r), c1, 1, 1, 0)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return x * self.sig(self.fc2(self.relu(self.fc1(self.pool(x)))))

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.point = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ConvResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        mid = max(16, out_ch // 2)
        self.conv1 = DepthwiseSeparableConv(in_ch, mid, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = DepthwiseSeparableConv(mid, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        s = self.skip(x)
        return F.relu(self.bn(y + s))

class EnhancedFrequencyBank(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.base_ch = base_ch
        self.low = ConvResidualBlock(in_ch, base_ch, dilation=1)
        self.mid = ConvResidualBlock(in_ch, base_ch, dilation=2)
        self.high = ConvResidualBlock(in_ch, base_ch, dilation=4)
        self.se = SEBlock(base_ch * 3, r=8)
        self.out = nn.Conv2d(base_ch * 3, base_ch, 1)
    def forward(self, x):
        l = self.low(F.avg_pool2d(x, kernel_size=1))
        m = self.mid(x)
        hp = x - F.interpolate(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1), size=x.shape[-2:])
        h = self.high(hp)
        cat = torch.cat([l, m, h], dim=1)
        cat = self.se(cat)
        return F.relu(self.out(cat))



class FFTBank(nn.Module):
    def __init__(self, out_dim=1024, grid=14, multi_scale=True):
        super().__init__()
        self.multi_scale = multi_scale
        self.grid = grid
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_dim//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim//2, out_dim, 1),
            nn.ReLU()
        )
        self.resblock = ConvResidualBlock(out_dim, out_dim//2)
        self.out_channels = out_dim // 2
    def forward(self, img):
        B, C, H, W = img.shape
        fft = torch.fft.fft2(img, dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft))
        mag = mag.mean(dim=1, keepdim=True)
        feat = self.conv(mag)
        feat = F.adaptive_avg_pool2d(feat, (self.grid, self.grid))
        feat = self.resblock(feat)
        P = feat.shape[2] * feat.shape[3]
        out = feat.reshape(B, feat.shape[1], P).permute(0, 2, 1)
        return out, mag




class FFTBankv2(nn.Module):

    def __init__(self, out_dim=1024, grid=14, n_bands=3, band_definitions=None):
        super().__init__()
        self.grid = grid
        self.out_dim = out_dim
        self.n_bands = n_bands
        self.out_channels = out_dim // 2
        if band_definitions is None:
            self.band_definitions = [(0, 0.1), (0.1, 0.4), (0.4, 1.0)]
        else:
            self.band_definitions = band_definitions
            
        assert self.n_bands == len(self.band_definitions)

        self.conv_mag_bands = nn.ModuleList()
        self.conv_phase_bands = nn.ModuleList()
        self.resblock_mag_bands = nn.ModuleList()
        self.resblock_phase_bands = nn.ModuleList()

        for _ in range(self.n_bands):
            self.conv_mag_bands.append(nn.Sequential(
                nn.Conv2d(1, out_dim // 2, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_dim // 2, out_dim, 1), nn.ReLU()
            ))
            self.conv_phase_bands.append(nn.Sequential(
                nn.Conv2d(1, out_dim // 2, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_dim // 2, out_dim, 1), nn.ReLU()
            ))
            self.resblock_mag_bands.append(ConvResidualBlock(out_dim, self.out_channels))
            self.resblock_phase_bands.append(ConvResidualBlock(out_dim, self.out_channels))
            
    def _create_radial_masks(self, H, W, device):
        """Helper to create radial masks."""
        

        y, x = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device), 
            indexing='ij'
        )

        center_y, center_x = H // 2, W // 2
        
        # Use fftshift-style coordinates (center is 0)
        r = torch.sqrt(
            (x - center_x).float()**2 + (y - center_y).float()**2
        )
        max_radius = min(center_y, center_x)
        # Add a small epsilon to avoid division by zero if max_radius is 0
        r_normalized = r / (max_radius + 1e-6) # Normalized radius [0, ~1.414]

        masks = []
        for (low, high) in self.band_definitions:
            mask = (r_normalized >= low) & (r_normalized < high)
            masks.append(mask.float().unsqueeze(0).unsqueeze(0))
        return masks # List of [1, 1, H, W] masks

    def forward(self, img):
        B, C, H, W = img.shape
        device = img.device

        gray_imgs = []
        for i in range(B):
            gray_imgs.append(TF.rgb_to_grayscale(img[i], num_output_channels=1))
        img_gray = torch.stack(gray_imgs) # (B, 1, H, W)

        fft = torch.fft.fft2(img_gray, dim=(-2, -1)) # (B, 1, H, W)
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

        real = fft_shifted.real
        imag = fft_shifted.imag
        
        mag = torch.log1p(torch.sqrt(real*real + imag*imag))
        phase = torch.atan2(imag, real) / math.pi # Normalized [-1, 1]

        masks = self._create_radial_masks(H, W, device) # List of n_bands masks
        
        all_mag_tokens = []
        all_phase_tokens = []
        

        for b in range(self.n_bands):
            mask = masks[b]
            
            # Apply mask
            mag_map = mag * mask
            phase_map = phase * mask

            mag_map_pooled = F.adaptive_avg_pool2d(mag_map, (self.grid, self.grid))
            phase_map_pooled = F.adaptive_avg_pool2d(phase_map, (self.grid, self.grid))

            # 2. Convolve the SMALL 14x14 maps
            feat_mag = self.conv_mag_bands[b](mag_map_pooled)
            # feat_mag = F.adaptive_avg_pool2d(feat_mag, (self.grid, self.grid)) # No longer needed
            feat_mag = self.resblock_mag_bands[b](feat_mag)
            
            feat_phase = self.conv_phase_bands[b](phase_map_pooled)
            feat_phase = self.resblock_phase_bands[b](feat_phase)

            # Reshape into tokens
            P = self.grid * self.grid
            mag_tokens = feat_mag.reshape(B, self.out_channels, P).permute(0, 2, 1)
            phase_tokens = feat_phase.reshape(B, self.out_channels, P).permute(0, 2, 1)
            
            all_mag_tokens.append(mag_tokens)
            all_phase_tokens.append(phase_tokens)

        full_mag_map_unpooled = torch.log1p(torch.sqrt(fft.real**2 + fft.imag**2))
        
        return all_mag_tokens, all_phase_tokens, full_mag_map_unpooled
    
    
    
class ClipLayerWeighting(nn.Module):
    def __init__(self, n_layers=12, embed_dim=768, kernel_size=3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.per_channel = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, padding=pad, groups=embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, clip_layers):
        B, L, D = clip_layers.shape
        x = clip_layers.permute(0, 2, 1)
        x = self.per_channel(x)
        x = self.pool(x).squeeze(-1)
        return x

class ResidualUpsampleV2(nn.Module):
    def __init__(self, in_ch, out_ch, use_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvResidualBlock(out_ch, out_ch)
        self.attention = SpatialAttention(out_ch) if use_attention else None
    
    def forward(self, x, skip=None):
        x = F.relu(self.up(x))
        if skip is not None:
            x = x + skip  # residual connection
        x = self.block(x)
        if self.attention is not None:
            x = self.attention(x)
        return x



class SpatialAttention(nn.Module):
    # Spatial attention for seg decoder
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention

class ResidualUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvResidualBlock(out_ch, out_ch)
    def forward(self, x):
        x = F.relu(self.up(x))
        return self.block(x)

class DinoPatchDecoder(nn.Module):
    def __init__(self, patch_dim=1024, freq_dim=512, patch_count=196, out_size=224, mid_ch=512):
        super().__init__()
        self.patch_count = patch_count
        self.grid = int(math.sqrt(patch_count))
        in_ch = patch_dim + freq_dim
        self.start = nn.Conv2d(in_ch, mid_ch, 1)
        self.res1 = ConvResidualBlock(mid_ch, mid_ch//2)
        self.up1 = ResidualUpsample(mid_ch//2, mid_ch//4)
        self.up2 = ResidualUpsample(mid_ch//4, mid_ch//8)
        self.up3 = ResidualUpsample(mid_ch//8, mid_ch//16)
        self.up4 = ResidualUpsample(mid_ch//16, mid_ch//32)
        self.final = nn.Conv2d(mid_ch//32, 1, 1)
        self.out_size = out_size
    def forward(self, patch_tokens, freq_patch_tokens=None, fft_mag=None):
        B, P, C = patch_tokens.shape
        g = self.grid
        x = patch_tokens.permute(0,2,1).reshape(B, C, g, g)
        if freq_patch_tokens is not None:
            f = freq_patch_tokens.permute(0,2,1).reshape(B, freq_patch_tokens.shape[2], g, g)
            x = torch.cat([x, f], dim=1)
        x = F.relu(self.start(x))
        x = self.res1(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        if fft_mag is not None:
            mag_small = F.adaptive_avg_pool2d(fft_mag, (self.out_size, self.out_size))
            x = x + 0.5 * mag_small
        return x

class SRMConv2D(nn.Module):
    # Steganalysis Residual Module with fixed high-pass filters
    # Extracts manipulation-sensitive noise patterns
    # These filters are NOT learned - proven effective in forensics literature
    # Input: (B, 3, H, W) RGB image
    # Output: (B, 9, H, W) noise residuals (3 filters × 3 color channels)
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Define 3 classic SRM kernels from steganalysis literature
        # These suppress content, amplify noise/artifacts
        
        # Kernel 1: Basic edge detector (3x3)
        # Detects local inconsistencies at pixel level
        kernel1 = torch.tensor([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ], dtype=torch.float32)
        
        # Kernel 2: Horizontal edge (3x3)
        # Sensitive to horizontal splicing boundaries
        kernel2 = torch.tensor([
            [ 0,  0,  0],
            [ 1, -2,  1],
            [ 0,  0,  0]
        ], dtype=torch.float32)
        
        # Kernel 3: Square 5x5 edge detector
        # Captures larger-scale noise patterns (resampling, compression)
        kernel3 = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=torch.float32) / 4.0
        
        # Create depthwise convolutions (one filter per channel)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False, groups=in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 5, padding=2, bias=False, groups=in_channels)
        
        # Initialize with SRM kernels and freeze
        with torch.no_grad():
            for c in range(in_channels):
                self.conv1.weight[c, 0] = kernel1
                self.conv2.weight[c, 0] = kernel2
                self.conv3.weight[c, 0] = kernel3
        
        # Freeze - these are hand-crafted, not learned
        self.conv1.weight.requires_grad = False
        self.conv2.weight.requires_grad = False
        self.conv3.weight.requires_grad = False
    
    def forward(self, x):
        # x: (B, 3, H, W)
        # Apply three different SRM filters
        noise1 = self.conv1(x)  # (B, 3, H, W)
        noise2 = self.conv2(x)  # (B, 3, H, W)
        noise3 = self.conv3(x)  # (B, 3, H, W)
        
        # Stack all noise maps
        noise = torch.cat([noise1, noise2, noise3], dim=1)  # (B, 9, H, W)
        return noise

class NoiseResidualBank(nn.Module):
    # Processes SRM noise residuals to detect manipulation artifacts
    # Theory: Real images have consistent camera sensor noise
    #         Fake images have learned/inconsistent noise patterns
    # GAN fingerprints appear in noise domain even when content looks real
    # Input: (B, 3, H, W) RGB image
    # Output: (B, P, out_channels) noise patch tokens for cross-attention
    #         (B, 1, H, W) anomaly emphasis map for visualization
    
    def __init__(self, in_ch=3, base_ch=64, grid=14):
        super().__init__()
        self.grid = grid
        
        # SRM extracts noise (suppresses content)
        self.srm = SRMConv2D(in_ch)
        
        # Process 9-channel noise maps (3 SRM filters × 3 color channels)
        # Learn what "suspicious noise" looks like
        self.noise_proc = nn.Sequential(
            nn.Conv2d(9, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            ConvResidualBlock(base_ch, base_ch),  # Deepen representation
            ConvResidualBlock(base_ch, base_ch)
        )
        # Anomaly emphasis module (NOT "attention" - just emphasis weighting)
        # Computes spatial emphasis: where is noise most suspicious?
        # Uses gating mechanism to highlight anomalous regions
        self.anomaly_emphasis = nn.Sequential(
            nn.Conv2d(base_ch, base_ch//4, 1),
            nn.ReLU(),
            nn.Conv2d(base_ch//4, 1, 1),
            nn.Sigmoid()  # (B, 1, H, W) emphasis weights in [0,1]
        )
        
        self.out_channels = base_ch
    def forward(self, img):
        # img: (B, 3, H, W)
        B = img.shape[0]
        
        # Extract noise residuals using fixed SRM filters
        # This removes semantic content, keeps manipulation traces
        noise = self.srm(img)  # (B, 9, H, W)
        
        # Process noise to learn manipulation-specific patterns
        feat = self.noise_proc(noise)  # (B, base_ch, H, W)
        
        # Compute spatial emphasis: where is noise anomalous?
        # High values → suspicious regions (boundaries, artifacts)
        # Low values → normal regions (uniform texture)
        emphasis = self.anomaly_emphasis(feat)  # (B, 1, H, W)
        
        # Apply emphasis weighting to features
        # Amplifies features in suspicious regions
        feat = feat * emphasis  # (B, base_ch, H, W)
        
        # Downsample to patch grid for cross-attention
        feat = F.adaptive_avg_pool2d(feat, (self.grid, self.grid))  # (B, base_ch, grid, grid)
        
        # Reshape to patch tokens: (B, P, base_ch) where P = grid²
        P = self.grid * self.grid
        noise_tokens = feat.reshape(B, self.out_channels, P).permute(0, 2, 1)  # (B, P, base_ch)
        
        return noise_tokens, emphasis
    


class FusedSegmentationDecoder(nn.Module):
    #segmentation decoder with skip connections and spatial attention maybe better localization.
    def __init__(self, patch_dim=1024, freq_dim=512, fused_dim=768, 
                 patch_count=196, out_size=224, mid_ch=512):
        super().__init__()
        self.patch_count = patch_count
        self.grid = int(math.sqrt(patch_count))
        self.out_size = out_size
        
        # Project fused features to spatial grid
        self.fused_proj = nn.Linear(fused_dim, mid_ch // 4)
        
        # Combine patch, freq, and fused features
        in_ch = patch_dim + freq_dim + (mid_ch // 4)
        
        self.start = nn.Conv2d(in_ch, mid_ch, 1)
        self.res1 = ConvResidualBlock(mid_ch, mid_ch // 2)
        
        # Upsampling path with attention
        self.up1 = ResidualUpsampleV2(mid_ch // 2, mid_ch // 4, use_attention=True)
        self.up2 = ResidualUpsampleV2(mid_ch // 4, mid_ch // 8 , use_attention=True)
        self.up3 = ResidualUpsampleV2(mid_ch // 8, mid_ch // 16 , use_attention=True)
        self.up4 = ResidualUpsampleV2(mid_ch // 16, mid_ch // 32 , use_attention=True)
        
        self.final = nn.Conv2d(mid_ch // 32, 1, 1)
    
    def forward(self, patch_tokens, freq_patch_tokens, fused_patch_tokens):
            # patch_tokens: (B, P, patch_dim) - DINO patches
            # freq_patch_tokens: (B, P, freq_dim) - FFT patches
            # fused_patch_tokens: (B, P, fused_dim) - Cross attended fused patches
        B, P, _ = patch_tokens.shape
        g = self.grid
        
        # Reshape to spatial grid
        patch_spatial = patch_tokens.permute(0, 2, 1).reshape(B, -1, g, g)
        freq_spatial = freq_patch_tokens.permute(0, 2, 1).reshape(B, -1, g, g)
        
        # Project and reshape fused features
        fused_proj = self.fused_proj(fused_patch_tokens)
        fused_spatial = fused_proj.permute(0, 2, 1).reshape(B, -1, g, g)
        
        # Concatenate all features
        x = torch.cat([patch_spatial, freq_spatial, fused_spatial], dim=1)
        
        # Decoder pathway
        x = F.relu(self.start(x))
        x = self.res1(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
    
        # Interpolate to target size
        x = F.interpolate(x, size=(self.out_size, self.out_size), 
                         mode='bilinear', align_corners=False)
        
        return x





def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return 1 - (2 * inter + eps) / (union + eps)


class FreqDINOv1(nn.Module):
    def __init__(self, n_fusion_layers=3, n_heads=8, d_model=768,
                 dino_embed_size=1024, n_patches=196, clip_layers=12, num_classes=3):
        super().__init__()
        self.dino_embed_size = dino_embed_size
        self.n_patches = n_patches
        self.grid = int(math.sqrt(n_patches))
        
        self.freq = EnhancedFrequencyBank(in_ch=3, base_ch=64)
        self.fft_bank = FFTBank(out_dim=dino_embed_size, grid=self.grid)
        fft_out_ch = self.fft_bank.out_channels  # fixed channel count from FFTBank
        
        self.clip_weight = ClipLayerWeighting(n_layers=clip_layers, embed_dim=768)
        self.clip_proj = nn.Linear(768, d_model)
        self.dino_proj = nn.Linear(dino_embed_size, d_model)
        self.patch_proj = nn.Linear(dino_embed_size, d_model)
        self.fft_to_dmodel = nn.Linear(fft_out_ch, d_model)
        
        self.pos = PositionalEncoding(d_model, max_len=n_patches + 4)
        self.cross_layers = nn.ModuleList([CrossAttentionLayer(d_model, n_heads) for _ in range(n_fusion_layers)])
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model * 3 + 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        # self.seg_decoder = DinoPatchDecoder(patch_dim=dino_embed_size, freq_dim=fft_out_ch, patch_count=n_patches, out_size=224, mid_ch=512)
        self.seg_decoder = FusedSegmentationDecoder(
            patch_dim=dino_embed_size,
            freq_dim=fft_out_ch,
            fused_dim=d_model,
            patch_count=n_patches,
            out_size=224,
            mid_ch=512
        )
        self.aux_seg_proj = nn.Conv2d(64, 128, 1)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, image, clip_embed, dino_cls, dino_reg, dino_patch, has_mask=None, mask=None):
        B = image.shape[0]
        
        f = self.freq(image)
        f_pool = F.adaptive_avg_pool2d(f, 1).view(B, -1)
        
        f_aux = self.aux_seg_proj(f)
        
        clip_w = self.clip_weight(clip_embed)
        clip_proj = self.clip_proj(clip_w)
        clip_tok = clip_proj.unsqueeze(0)
        
        patch_tokens = dino_patch
        patch_proj = self.patch_proj(patch_tokens)
        patch_seq = patch_proj.permute(1,0,2)
        
        dino_cls_proj = self.dino_proj(dino_cls) if dino_cls.ndim==2 else self.dino_proj(dino_cls.view(B,-1))
        dino_cls_tok = dino_cls_proj.unsqueeze(0)
        
        freq_patch_tokens, fft_mag = self.fft_bank(image)
        freq_proj_dmodel = self.fft_to_dmodel(freq_patch_tokens)
        freq_seq = freq_proj_dmodel.permute(1,0,2)
        
        q = clip_tok
        k = torch.cat([patch_seq, freq_seq], dim=0)
        v = k
        for layer in self.cross_layers:
            q = layer(q, k, v)
        fused_clip = q.squeeze(0)
        
        q2 = dino_cls_tok
        k2 = torch.cat([patch_seq, freq_seq, fused_clip.unsqueeze(0)], dim=0)
        v2 = k2
        for layer in self.cross_layers:
            q2 = layer(q2, k2, v2)
        fused_dino_cls = q2.squeeze(0)
        
        q3 = patch_seq
        k3 = torch.cat([patch_seq, freq_seq, fused_clip.unsqueeze(0), fused_dino_cls.unsqueeze(0)], dim=0)
        v3 = k3
        for layer in self.cross_layers:
            q3 = layer(q3, k3, v3)
        fused_patches = q3.permute(1, 0, 2)  # (B, P, d_model)
        
        cls_input = torch.cat([f_pool, fused_clip, fused_dino_cls, clip_proj], dim=1)
        logits = self.classifier_head(cls_input)
        
        seg_logits = self.seg_decoder(dino_patch, freq_patch_tokens, fused_patches)
        
        seg_loss = None
        if (mask is not None) and (has_mask is not None):
            return {
                "logits": logits,
                "seg_logits": seg_logits,
                "seg_loss": seg_loss,  
                "has_mask": has_mask,
                "mask": mask
            }
        return {
            "logits": logits,
            "seg_logits": seg_logits,
            "seg_loss": seg_loss
        }




class FreqDINOv2(nn.Module):
    def __init__(self, n_fusion_layers=3, n_heads=8, d_model=768,
                 dino_embed_size=1024, n_patches=196, clip_layers=12, num_classes=3,
                 n_bands=3):
        super().__init__()
        self.dino_embed_size = dino_embed_size
        self.n_patches = n_patches
        self.grid = int(math.sqrt(n_patches))
        self.n_bands = n_bands

        # image-level spatial freq and FFT banks
        self.freq = EnhancedFrequencyBank(in_ch=3, base_ch=64)
        
        # This FFTBank is the new one, which outputs lists of tokens
        self.fft_bank = FFTBankv2(out_dim=dino_embed_size, grid=self.grid, n_bands=self.n_bands)
        
        fft_out_ch = self.fft_bank.out_channels  # out_dim//2

        # clip/dino linear projections
        self.clip_weight = ClipLayerWeighting(n_layers=clip_layers, embed_dim=768)
        self.clip_proj = nn.Linear(768, d_model)
        self.dino_proj = nn.Linear(dino_embed_size, d_model)
        self.patch_proj = nn.Linear(dino_embed_size, d_model)


        self.fft_to_dmodel = nn.Linear(fft_out_ch, d_model)
        self.phase_to_dmodel = nn.Linear(fft_out_ch, d_model)



        self.pos = PositionalEncoding(d_model, max_len=n_patches + 4)
        self.cross_layers = nn.ModuleList([CrossAttentionLayer(d_model, n_heads) for _ in range(n_fusion_layers)])
        
        self.noise_bank = NoiseResidualBank(in_ch=3, base_ch=64, grid=self.grid)
        noise_out_ch = self.noise_bank.out_channels
        self.noise_to_dmodel = nn.Linear(noise_out_ch, d_model)
        
        # #if using contrastive ->
        # self.noise_proj_contrast = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 2, 256)  # Contrastive embedding dimension
        # )
        # self.dino_proj_contrast = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 2, 256)
        # )

        # classifier and segmentation
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model * 3 + 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        self.seg_decoder = FusedSegmentationDecoder(
            patch_dim=dino_embed_size,
            freq_dim=fft_out_ch, # The seg_decoder will use the raw full-spectrum tokens
            fused_dim=d_model,
            patch_count=n_patches,
            out_size=224,
            mid_ch=512
        )
        self.aux_seg_proj = nn.Conv2d(64, 128, 1)
        self.norm = nn.LayerNorm(d_model)

        self.bc_pca_clip = BC_PCA(d_model=d_model, n_bands=self.n_bands)

    def forward(self, image, clip_embed, dino_cls, dino_reg, dino_patch, has_mask=None, mask=None):
        B = image.shape[0]

        f = self.freq(image)
        f_pool = F.adaptive_avg_pool2d(f, 1).view(B, -1)
        # f_aux = self.aux_seg_proj(f)

        clip_w = self.clip_weight(clip_embed)
        clip_proj = self.clip_proj(clip_w)  # (B, d_model)

        patch_tokens = dino_patch  # (B, P, dino_embed_size)
        patch_proj = self.patch_proj(patch_tokens)  # (B, P, d_model)
        patch_seq = patch_proj.permute(1,0,2)  # (P, B, d_model)

        dino_cls_proj = self.dino_proj(dino_cls) if dino_cls.ndim==2 else self.dino_proj(dino_cls.view(B,-1))
        dino_cls_tok = dino_cls_proj.unsqueeze(0)


        mag_bands_raw, phase_bands_raw, fft_mag = self.fft_bank(image)
        
        mag_bands = [self.fft_to_dmodel(band) for band in mag_bands_raw]
        phase_bands = [self.phase_to_dmodel(band) for band in phase_bands_raw]
        

        
        clip_tok_batch = clip_proj.unsqueeze(1)  # (B, 1, d_model)
        
        noise_tokens, noise_emphasis = self.noise_bank(image) # token: (B, P, 64) ; noise_emphasis: (B, 1, H, W)
        noise_proj_dmodel = self.noise_to_dmodel(noise_tokens)  # (B, P, d_model)
        noise_seq = noise_proj_dmodel.permute(1,0,2)  # (P, B, d_model)
        
        fused_clip_batch = self.bc_pca_clip(clip_tok_batch, mag_bands, phase_bands)  # (B,1,d)
        fused_clip = fused_clip_batch.squeeze(1) 


        freq_proj_dmodel = torch.stack(mag_bands, dim=0).mean(dim=0) # (B, P, d_model)
        freq_seq = freq_proj_dmodel.permute(1,0,2) # (P, B, d_model)


        
        q2 = dino_cls_tok
        k2 = torch.cat([patch_seq, noise_seq, fused_clip.unsqueeze(0)], dim=0)
        v2 = k2
        for layer in self.cross_layers:
            q2 = layer(q2, k2, v2)
        fused_dino_cls = q2.squeeze(0)

        q3 = patch_seq
        k3 = torch.cat([patch_seq, fused_clip.unsqueeze(0), fused_dino_cls.unsqueeze(0)], dim=0)
        v3 = k3
        for layer in self.cross_layers:
            q3 = layer(q3, k3, v3)
        fused_patches = q3.permute(1, 0, 2)  # (B, P, d_model)

        cls_input = torch.cat([f_pool, fused_clip, fused_dino_cls, clip_proj], dim=1)
        logits = self.classifier_head(cls_input)


        seg_decoder_freq_tokens = mag_bands_raw[1] #
        
        seg_logits = self.seg_decoder(dino_patch, seg_decoder_freq_tokens, fused_patches)

        seg_loss = None
        
        # noise_pooled = noise_proj_dmodel.mean(dim=1)  # (B, d_model) - average over patches
        # noise_contrast = self.noise_proj_contrast(noise_pooled)  # (B, 256)
        # noise_contrast = F.normalize(noise_contrast, p=2, dim=1) #l2 normalize
        
        # dino_contrast = self.dino_proj_contrast(fused_dino_cls)  # (B, 256)
        # dino_contrast = F.normalize(dino_contrast, p=2, dim=1) #l2 normalize
        
        
        if (mask is not None) and (has_mask is not None):
             return {
                 "logits": logits,
                 "seg_logits": seg_logits,
                 "seg_loss": seg_loss,  
                 "has_mask": has_mask,
                 "mask": mask
                #  "noise_contrast": noise_contrast, 
                #  "dino_contrast": dino_contrast,
                #  "noise_emphasis": noise_emphasis
                 
             }
        return {
             "logits": logits,
             "seg_logits": seg_logits, 
             "seg_loss": seg_loss
            #  "noise_contrast": noise_contrast,
            # "dino_contrast": dino_contrast,
            # "noise_emphasis": noise_emphasis
        }



class FreqDINO(nn.Module):
    def __init__(self, n_fusion_layers=3, n_heads=8, d_model=768,
                 dino_embed_size=1024, n_patches=196, clip_layers=12, num_classes=3,
                 n_bands=3):
        super().__init__()
        self.dino_embed_size = dino_embed_size
        self.n_patches = n_patches
        self.grid = int(math.sqrt(n_patches))
        self.n_bands = n_bands

        # image-level spatial freq and FFT banks
        self.freq = EnhancedFrequencyBank(in_ch=3, base_ch=64)
        
        # This FFTBank is the new one, which outputs lists of tokens
        self.fft_bank = FFTBankv2(out_dim=dino_embed_size, grid=self.grid, n_bands=self.n_bands)
        
        fft_out_ch = self.fft_bank.out_channels  # out_dim//2

        # clip/dino linear projections
        self.clip_weight = ClipLayerWeighting(n_layers=clip_layers, embed_dim=768)
        self.clip_proj = nn.Linear(768, d_model)
        self.dino_proj = nn.Linear(dino_embed_size, d_model)
        self.patch_proj = nn.Linear(dino_embed_size, d_model)


        self.fft_to_dmodel = nn.Linear(fft_out_ch, d_model)
        self.phase_to_dmodel = nn.Linear(fft_out_ch, d_model)



        self.pos = PositionalEncoding(d_model, max_len=n_patches + 4)
        self.cross_layers = nn.ModuleList([CrossAttentionLayer(d_model, n_heads) for _ in range(n_fusion_layers)])
        
        self.noise_bank = NoiseResidualBank(in_ch=3, base_ch=64, grid=self.grid)
        noise_out_ch = self.noise_bank.out_channels
        self.noise_to_dmodel = nn.Linear(noise_out_ch, d_model)
        

        # classifier and segmentation
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model * 3 + 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        self.seg_decoder = FusedSegmentationDecoder(
            patch_dim=dino_embed_size,
            freq_dim=fft_out_ch, # The seg_decoder will use the raw full-spectrum tokens
            fused_dim=d_model,
            patch_count=n_patches,
            out_size=224,
            mid_ch=512
        )
        self.aux_seg_proj = nn.Conv2d(64, 128, 1)
        self.norm = nn.LayerNorm(d_model)

        self.bc_pca_clip = BC_PCA(d_model=d_model, n_bands=self.n_bands)

    def forward(self, image, clip_embed, dino_cls, dino_reg, dino_patch, has_mask=None, mask=None):
        B = image.shape[0]

        f = self.freq(image)
        f_pool = F.adaptive_avg_pool2d(f, 1).view(B, -1)
        # f_aux = self.aux_seg_proj(f)

        clip_w = self.clip_weight(clip_embed)
        clip_proj = self.clip_proj(clip_w)  # (B, d_model)

        patch_tokens = dino_patch  # (B, P, dino_embed_size)
        patch_proj = self.patch_proj(patch_tokens)  # (B, P, d_model)
        patch_seq = patch_proj.permute(1,0,2)  # (P, B, d_model)

        dino_cls_proj = self.dino_proj(dino_cls) if dino_cls.ndim==2 else self.dino_proj(dino_cls.view(B,-1))
        dino_cls_tok = dino_cls_proj.unsqueeze(0)


        mag_bands_raw, phase_bands_raw, fft_mag = self.fft_bank(image)
        
        mag_bands = [self.fft_to_dmodel(band) for band in mag_bands_raw]
        phase_bands = [self.phase_to_dmodel(band) for band in phase_bands_raw]
        

        
        clip_tok_batch = clip_proj.unsqueeze(1)  # (B, 1, d_model)
        
        noise_tokens, noise_emphasis = self.noise_bank(image) # token: (B, P, 64) ; noise_emphasis: (B, 1, H, W)
        noise_proj_dmodel = self.noise_to_dmodel(noise_tokens)  # (B, P, d_model)
        noise_seq = noise_proj_dmodel.permute(1,0,2)  # (P, B, d_model)
        
        fused_clip_batch = self.bc_pca_clip(clip_tok_batch, mag_bands, phase_bands)  # (B,1,d)
        fused_clip = fused_clip_batch.squeeze(1) 


        freq_proj_dmodel = torch.stack(mag_bands, dim=0).mean(dim=0) # (B, P, d_model)
        freq_seq = freq_proj_dmodel.permute(1,0,2) # (P, B, d_model)


        
        q2 = dino_cls_tok
        k2 = torch.cat([patch_seq, noise_seq, fused_clip.unsqueeze(0)], dim=0)
        v2 = k2
        for layer in self.cross_layers:
            q2 = layer(q2, k2, v2)
        fused_dino_cls = q2.squeeze(0)

        q3 = patch_seq
        k3 = torch.cat([patch_seq, fused_clip.unsqueeze(0), fused_dino_cls.unsqueeze(0)], dim=0)
        v3 = k3
        for layer in self.cross_layers:
            q3 = layer(q3, k3, v3)
        fused_patches = q3.permute(1, 0, 2)  # (B, P, d_model)

        cls_input = torch.cat([f_pool, fused_clip, fused_dino_cls, clip_proj], dim=1)
        logits = self.classifier_head(cls_input)


        seg_decoder_freq_tokens = mag_bands_raw[1] #
        
        seg_logits = self.seg_decoder(dino_patch, seg_decoder_freq_tokens, fused_patches)

        seg_loss = None

        if (mask is not None) and (has_mask is not None):
             return {
                 "logits": logits,
                 "seg_logits": seg_logits,
                 "seg_loss": seg_loss,  
                 "has_mask": has_mask,
                 "mask": mask
                 
             }
        return {
             "logits": logits,
             "seg_logits": seg_logits,
             "seg_loss": seg_loss

        }
        
        

#TODO Abilations 
# - Remove Noise remmoved from concat
# - w/o BC-PCA removed fused_clip
# - w/o BC-PCA and ClIP [remove fused_clip and clip_tok]
# - w/o dino patches (seg decoder only uses freq tokens
# - ONLY Dino feats (remove all other inputs)





if __name__ == "__main__":
    model = FreqDINOv2()
    B = 2
    image = torch.randn(B, 3, 224, 224)
    clip_embed = torch.randn(B, 12, 768)
    dino_cls = torch.randn(B, 1024)
    dino_reg = torch.randn(B, 4, 1024)
    dino_patch = torch.randn(B, 196, 1024)
    has_mask = torch.tensor([1, 0])
    mask = torch.randn(B, 224, 224).gt(0).float()
    
    out = model(image, clip_embed, dino_cls, dino_reg, dino_patch, 
                has_mask=has_mask, mask=mask)
    print("Logits shape:", out["logits"].shape)
    print("Seg logits shape:", out["seg_logits"].shape)
