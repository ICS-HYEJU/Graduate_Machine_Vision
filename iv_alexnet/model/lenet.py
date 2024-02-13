import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.lenet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Tanh(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Tanh(),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.Tanh(),
            nn.Flatten(),  # (1, 4320) --> input (52, 52)
            nn.Linear(4320, 2160),
            nn.Tanh(),
            nn.Linear(2160, 8)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.lenet(x)

    def predict(self, output):
        sig_out = self.sigmoid(output)
        sig_out[sig_out > 0.5] = 1
        sig_out[sig_out <= 0.5] = 0
        return sig_out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    import torch

    model = LeNet()
    model.init_weights()
    x = torch.rand(1, 3, 52, 52)
    output = model.forward(x)
    preds = model.predict(output)
    print(output.shape)