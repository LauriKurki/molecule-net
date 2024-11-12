from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride)
        self.res_conv = nn.Conv3d(in_channels, out_channels, 1, stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if x.shape != res.shape:
            res = self.res_conv(res)

        x = self.relu(x + res)
        return x
    
class AttentionBlock3D(nn.Module):
    def __init__(self, in_channels, query_channels, attention_channels, kernel_size, stride):
        super().__init__()
        self.x_conv = nn.Conv3d(in_channels, attention_channels, kernel_size, stride)
        self.q_conv = nn.Conv3d(query_channels, attention_channels, kernel_size, stride)
        self.a_conv = nn.Conv3d(attention_channels, 1, kernel_size, stride)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, q):
        q = F.interpolate(q, size=x.size()[2:], mode="bilinear", align_corners=False)
        
        # convoluve query
        q = self.relu(self.q_conv(q))

        # convolve x and sum
        a = self.relu(self.x_conv(x))
        a = self.relu(x+q)

        a = self.sigmoid(self.a_conv(a))
        x = x * a

        return x
