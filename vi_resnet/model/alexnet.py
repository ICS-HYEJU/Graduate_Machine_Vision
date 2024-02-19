import torch.nn as nn

class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.alexnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.LocalResponseNorm(96),
            #
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(256),
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            #
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            #
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            #
            nn.AdaptiveAvgPool2d(6),
            nn.Flatten(),
            #
            nn.Dropout(0.5),
            #
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU,
            #
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            #
            nn.Linear(4096,8)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        return self.alexnet(x)

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
                    nn.init.constant_(m.bias,0)