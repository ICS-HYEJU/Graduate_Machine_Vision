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
    def forward(self, x):
        return self.conv(x)

class residual_block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(residual_block, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * residual_block.expansion, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * residual_block.expansion)
        )
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        # projection mapping using 1x1 conv
        if stride != 1 or (in_channels != out_channels * residual_block.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * residual_block.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * residual_block.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()

        block = residual_block
        num_block = [2, 2, 2, 2]

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 1)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, 8)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    def predict(self, output):
        sig_out = self.sigmoid(output)
        sig_out[sig_out > 0.5] = 1
        sig_out[sig_out <= 0.5] = 0
        return sig_out

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

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