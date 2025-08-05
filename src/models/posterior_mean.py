import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D_BN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.res_conv(x)
        out = self.bn1(x)
        out = self.act(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return out + identity


class PosteriorMeanModel3D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 base_features=(16, 32, 64, 128),
                 dropout=0.1):
        super().__init__()
        # Encoder path
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        prev_ch = in_channels
        for f in base_features:
            self.enc_blocks.append(ResidualBlock3D_BN(prev_ch, f, dropout=dropout))
            prev_ch = f

        # Bottleneck
        self.bottleneck = ResidualBlock3D_BN(prev_ch, prev_ch * 2, dropout=dropout)

        # Decoder path
        rev_features = list(reversed(base_features))
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        in_ch = prev_ch * 2
        for f in rev_features:
            self.up_convs.append(nn.ConvTranspose3d(in_ch, f, kernel_size=2, stride=2))
            self.dec_blocks.append(ResidualBlock3D_BN(f * 2, f, dropout=dropout))
            in_ch = f

        # Final conv
        self.final_conv = nn.Conv3d(base_features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        out = x
        # Encoder
        for enc in self.enc_blocks:
            out = enc(out)
            skips.append(out)
            out = self.pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder
        for up, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            out = up(out)
            # align spatial dims
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode='trilinear', align_corners=False)
            out = torch.cat([skip, out], dim=1)
            out = dec(out)

        return self.final_conv(out)
