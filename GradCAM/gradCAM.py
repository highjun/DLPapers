import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class GradCAM():
    def __init__(self, model,layer,device):
        self.model = model
        self.layer = layer
        self.device = device
        layer.register_forward_hook(self.forward_hook)
        layer.register_backward_hook(self.backward_hook)
    def forward_hook(self,_,input,output):
        self.output = output[0]
    def backward_hook(self,_,grad_input,grad_output):
        self.grad_output = grad_output[0]
    def output(self,input_tensor ,target_class = -1):
        self.model.zero_grad()
        self.model.eval()
        self.target_class = target_class
        input_batch =input_tensor.unsqueeze(0).to(self.device)
        pred = self.model(input_batch)
        if self.target_class == -1:
            self.target_class = torch.argmax(pred)
        pred = pred.squeeze(0)
        pred[self.target_class].backward()
        alpha_k = torch.sum(self.grad_output.squeeze(0),dim=(1,2),keepdim=True)
        cam = torch.sum(alpha_k*self.output,dim=0,keepdim= True).cpu().unsqueeze(0)
        cam = torch.nn.functional.relu(cam)
        upsample = torch.nn.Upsample(size = (input_tensor.shape[1],input_tensor.shape[2]), mode='bilinear')
        cam = upsample(cam).squeeze()
        cam = (cam - torch.min(cam))/(torch.max(cam)-torch.min(cam))
        cam = cam.detach().numpy()
        cmap = plt.get_cmap('jet')
        cam = cmap(cam)

        T = transforms.ToPILImage()

        min_tmp = torch.min(input_tensor,dim = 2,keepdim= True)[0]
        max_tmp = torch.max(input_tensor,dim=2,keepdim= True)[0]
        backgd  = (input_tensor- min_tmp)/(max_tmp-min_tmp)

        cam = T((cam*255).astype(np.uint8))
        backgd = T(backgd).convert("RGBA")
        blended = Image.blend(backgd, cam, 0.5)
        
        return blended