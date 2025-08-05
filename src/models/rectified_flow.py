import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """
    A 3D residual block that takes a volume and a time embedding.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(in_channels)
        self.activation1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.activation2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_conv = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.activation1(h)
        h = self.conv1(h)
        # Project and add the time embedding (broadcast over D, H, W)
        t_emb_proj = self.time_proj(t_emb)
        h = h + t_emb_proj[:, :, None, None, None]
        h = self.norm2(h)
        h = self.activation2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.residual_conv(x)


class RectifiedFlowModel3D(nn.Module):
    """
    A 3D Rectified Flow model that predicts a velocity field to transform
    a noised posterior-mean volume toward the true contrast-enhanced volume.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32, time_emb_dim=128, dropout=0.1):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # Encoder (downsampling path)
        self.enc1 = ResidualBlock3D(in_channels, base_channels, time_emb_dim, dropout)
        self.enc2 = ResidualBlock3D(base_channels, base_channels * 2, time_emb_dim, dropout)
        self.enc3 = ResidualBlock3D(base_channels * 2, base_channels * 4, time_emb_dim, dropout)
        self.enc4 = ResidualBlock3D(base_channels * 4, base_channels * 8, time_emb_dim, dropout)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(base_channels * 8, base_channels * 8, time_emb_dim, dropout)

        # Decoder (upsampling path)
        self.dec4 = ResidualBlock3D(base_channels * 16, base_channels * 4, time_emb_dim, dropout)
        self.dec3 = ResidualBlock3D(base_channels * 8, base_channels * 2, time_emb_dim, dropout)
        self.dec2 = ResidualBlock3D(base_channels * 4, base_channels, time_emb_dim, dropout)
        self.dec1 = ResidualBlock3D(base_channels * 2, base_channels, time_emb_dim, dropout)

        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        self.downsample = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

    def forward(self, x, t):
        t_emb = self.time_embedding(t.unsqueeze(1))
        enc1 = self.enc1(x, t_emb)
        enc2 = self.enc2(self.downsample(enc1), t_emb)
        enc3 = self.enc3(self.downsample(enc2), t_emb)
        enc4 = self.enc4(self.downsample(enc3), t_emb)
        bottleneck = self.bottleneck(self.downsample(enc4), t_emb)
        dec4 = self.dec4(
            torch.cat([F.interpolate(bottleneck, size=enc4.shape[2:], mode="trilinear", align_corners=False), enc4],
                      dim=1),
            t_emb
        )
        dec3 = self.dec3(
            torch.cat([F.interpolate(dec4, size=enc3.shape[2:], mode="trilinear", align_corners=False), enc3], dim=1),
            t_emb
        )
        dec2 = self.dec2(
            torch.cat([F.interpolate(dec3, size=enc2.shape[2:], mode="trilinear", align_corners=False), enc2], dim=1),
            t_emb
        )
        dec1 = self.dec1(
            torch.cat([F.interpolate(dec2, size=enc1.shape[2:], mode="trilinear", align_corners=False), enc1], dim=1),
            t_emb
        )
        return self.final_conv(dec1)
