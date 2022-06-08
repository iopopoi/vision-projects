import torch.nn as nn

class ModelCNN(nn.Module):
    def __init__(self, in_channel = 1):
        super(ModelCNN, self).__init__()

        self.in_dim = 28 * 28 * 3
        self.out_dim = 10
        self.keep_prob = 0.5

        c1 = 6
        c2 = 16
        c3 = 64
 
        self.layer1 = 0
        if in_channel == 3:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels= c1, kernel_size=3, stride=1, padding=1), # 28 * 28 * 6
                nn.BatchNorm2d(c1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2) # 14 * 14 * 6
                )
        else:            
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels= c1, kernel_size=3, stride=1, padding=1), # 28 * 28 * 6
                nn.BatchNorm2d(c1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2) # 14 * 14 * 6
                )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels= c2, kernel_size=3, stride=1), # 12 * 12 * 16 
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 6 * 6 * 16 
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=c2, out_channels= c3, kernel_size=3, stride=1), # 4 * 4 * 64
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 2 * 2 * 64
            )
        
        self.fc4 = nn.Linear(2 * 2 * 64, 120, bias=True) 
        nn.init.xavier_uniform_(self.fc4.weight)
        self.layer4 = nn.Sequential(
            self.fc4,
            nn.ReLU()
            ,nn.Dropout(p = 1 - self.keep_prob)
        )

        self.fc5 = nn.Linear(120, 80, bias=True)
        nn.init.xavier_uniform_(self.fc5.weight)
        self.layer5 = nn.Sequential( 
            self.fc5,
            nn.ReLU()
            ,nn.Dropout(p = 1 - self.keep_prob)
        )

        
        self.fc6 = nn.Linear(80, self.out_dim, bias=True)
        nn.init.xavier_uniform_(self.fc6.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)

        out = self.layer4(out)
        out = self.layer5(out)

        out = self.fc6(out)
        return out

