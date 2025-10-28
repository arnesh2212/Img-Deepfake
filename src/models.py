import torch
import torch.nn as nn
import torch.nn.functional as F
import math



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
        self.seg_decoder = DinoPatchDecoder(patch_dim=dino_embed_size, freq_dim=fft_out_ch, patch_count=n_patches, out_size=224, mid_ch=512)
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
        cls_input = torch.cat([f_pool, fused_clip, fused_dino_cls, clip_proj], dim=1)
        logits = self.classifier_head(cls_input)
        seg_logits = self.seg_decoder(dino_patch, freq_patch_tokens, fft_mag)
        seg_loss = None
        if (mask is not None) and (has_mask is not None):
            has_mask = has_mask.view(-1).bool()
            if has_mask.any():
                seg_pred = seg_logits[has_mask]
                seg_gt = mask[has_mask].unsqueeze(1).float()
                bce = F.binary_cross_entropy_with_logits(seg_pred, seg_gt)
                dloss = dice_loss(seg_pred, seg_gt).mean()
                seg_loss = bce + dloss
        return {
            "logits": logits,
            "seg_logits": seg_logits,
            "seg_loss": seg_loss
        }



class FreqDINOv2(nn.Module):
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
