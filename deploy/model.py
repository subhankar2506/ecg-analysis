import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel recalibration
    """
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SeparableConv1d(nn.Module):
    """
    Depthwise separable 1D convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class XceptionBlock(nn.Module):
    """
    Xception block with residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1, dilation=1, dropout=0.2, use_se=True):
        super(XceptionBlock, self).__init__()

        self.sepconv1 = SeparableConv1d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=(kernel_size - 1) // 2 * dilation, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.sepconv2 = SeparableConv1d(
            out_channels, out_channels, kernel_size, stride=1,
            padding=(kernel_size - 1) // 2 * dilation, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # SE block
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels or stride != 1 else nn.Identity()
        self.bn_res = nn.BatchNorm1d(out_channels) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        residual = self.bn_res(self.residual(x))

        out = self.sepconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.sepconv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        out = out + residual
        out = self.relu(out)

        return out

class GlobalContextAttention(nn.Module):
    """
    Global context attention module
    """
    def __init__(self, in_channels, reduction=16):
        super(GlobalContextAttention, self).__init__()
        self.conv_squeeze = nn.Conv1d(in_channels, 1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Spatial attention
        spatial_attn = self.sigmoid(self.bn(self.conv_squeeze(x)))
        x_spatial = x * spatial_attn

        # Channel attention
        b, c, _ = x.size()
        avg_pool = F.adaptive_avg_pool1d(x_spatial, 1).view(b, c)
        channel_attn = self.fc(avg_pool).view(b, c, 1)

        out = x * channel_attn
        return out

class XceptionTime(nn.Module):
    """
    XceptionTime: Adaptation of Xception architecture for time series classification
    """
    def __init__(self, input_channels=2, num_classes=5, initial_filters=64, depth=6, kernel_size=15, dropout=0.2):
        super(XceptionTime, self).__init__()

        # Input layer
        self.conv1 = nn.Conv1d(input_channels, initial_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Xception blocks with different dilation rates for multi-scale feature extraction
        self.blocks = nn.ModuleList()

        filters = initial_filters
        for i in range(depth):
            if i % 2 == 0 and i > 0:
                stride = 2
                filters *= 2
            else:
                stride = 1

            # Alternate between different dilation rates
            dilation = 2 ** (i % 3)

            self.blocks.append(
                XceptionBlock(
                    filters // 2 if i > 0 and stride == 2 else initial_filters if i == 0 else filters,
                    filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    dropout=dropout,
                    use_se=(i % 2 == 0)  # Add SE blocks to alternate layers
                )
            )

        # Global context attention
        self.global_context = GlobalContextAttention(filters)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters, num_classes)

    def forward(self, x):
        # If x is in shape [batch_size, seq_len, channels], permute to [batch_size, channels, seq_len]
        if x.size(1) > x.size(2):
            x = x.permute(0, 2, 1)

        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Xception blocks
        for block in self.blocks:
            x = block(x)

        # Apply global context attention
        x = self.global_context(x)

        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x