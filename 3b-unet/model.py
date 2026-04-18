import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

#in_channels = 1 for grayscale, 3 for RGB
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2),
                    DoubleConv(feature * 2, feature)
                )
            )
        #self.final_conv = nn.Sequential(nn.Conv2d(features[0], out_channels, kernel_size=1),nn.Sigmoid())
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) #replace due to gray images giving white images as result

    def forward(self, x):
        skip_connections = []
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        # Decoder
        for idx, up in enumerate(self.ups):
            x = up[0](x)  # Upsample
            skip = skip_connections[idx]

            # Handle size mismatch incase image dimensions are not a power of 2.
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = up[1](x)
        return self.final_conv(x)