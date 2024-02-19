import torch.nn as nn

class convBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(convBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self,x):
        return self.conv(x)

class vggnet(nn.Module):
    def __init__(self):
        super(vggnet, self).__init__()
        self.vggNet_backbone = nn.Sequential(
            convBlock(in_channel=3, out_channel=64, kernel_size=3, stride=1, padding=0),
            convBlock(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            convBlock(64,128,3,1,1),
            convBlock(128,128,3,1,1),
            convBlock(128,256,3,1,1),
            convBlock(256, 256, 3, 1, 1),
            convBlock(256,256,1,1,0),
            nn.MaxPool2d(2,2,0),
            convBlock(256,512,3,1,1),
            convBlock(512, 512, 3, 1, 1),
            convBlock(512, 512, 1, 1, 0)
        )

        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(256, 8)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.vggNet_backbone(x)
        out = self.linear(out)
        return out

    def predict(self, output):
        sig_out = self.sigmoid(output)
        sig_out[sig_out > 0.5] = 1
        sig_out[sig_out < 0.5] = 0
        return sig_out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_noraml_(m.weight, nonlinearity='leack_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)