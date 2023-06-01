# PyTorch Imports
import torch
import torch.nn as nn
import torchvision 
 


# Class: PredictionHead
class PredictionHead(nn.Module):
    def __init__(self,in_features,out_features):
        super(PredictionHead,self).__init__()
        self.fc1 = nn.Linear(in_features,out_features)
        self.fc2 = nn.Linear(in_features,out_features)
        self.fc3 = nn.Linear(out_features,1)
    def forward(self,x):
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        id1 = torch.nn.functional.sigmoid(self.fc3(y1).reshape(-1))
        id2 = torch.nn.functional.sigmoid(self.fc3(y2).reshape(-1))
        y3 = 1 - torch.minimum(id1, id2)
        return y1, y2, y3,id1,id2



# Class: Resnet18
class Resnet18_Min(nn.Module):
    def __init__(self,out_features):
        super(Resnet18_Min, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = PredictionHead(self.model.fc.in_features,out_features)

    def forward(self,x):
        return self.model(x)