import copy
import numpy as np
from PIL import Image
import torch
class GuidedBP():
    def __init__(self,model):
        self.model = copy.deepcopy(model)
        for module in self.model.modules():
            if module.__class__.__name__ =="ReLU":
                module.register_backward_hook(self.backward_hook)
    def backward_hook(self,_, grad_input, grad_output):
        #grad_input: tuple, grad_output: tuple
        return ((grad_output[0] > 0)* grad_input[0],)
    def output(self,input_tensor, target_class = -1):
        self.model.eval()
        self.model.zero_grad()
        self.target_class = target_class    
        input_batch = input_tensor.unsqueeze(0)
        input_batch.requires_grad_()
        pred = self.model(input_batch)
        if self.target_class == -1:
            self.target_class = torch.argmax(pred)
        pred.squeeze(0)[target_class].backward()
        output = input_batch.grad.squeeze(0).cpu()
        output = output-output.min()
        output = output/output.max()
        return Image.fromarray((output.permute(1,2,0).detach().numpy()*255).astype(np.uint8))