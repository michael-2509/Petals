import torch
from torch import nn
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    model = models.densenet121(pretrained=True)
   
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024, 512)),
                           ('relu', nn.ReLU()),
                           ('dropout', nn.Dropout(0.2)),
                           ('fc2', nn.Linear(512, 102)),
                           ('output', nn.LogSoftmax(dim=1))
                            ]))
        
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    return model



