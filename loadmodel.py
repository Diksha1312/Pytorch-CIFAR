import torch
import torch.nn
import torch.backends.cudnn as cudnn
from torchvision.models import resnet18

from torchsummary import summary

import os

model = resnet18()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

if os.path.exists('model/resnet.pth'):
    print("===> Loading model")
    checkpoint = torch.load('model/resnet.pth', map_location=device)
    new_state_dict = {key.replace("module.", ""): value for key, value in checkpoint['model'].items()}
    model.load_state_dict(checkpoint['model'])
    #best_acc = checkpoint['acc']
    #start_epoch = checkpoint['epoch']
    save_model_summary(model, (3,32,32), save_path)

# for name, param in model.named_parameters():
#     print(f"Parameter name: {name}, Size: {param.size()}")


