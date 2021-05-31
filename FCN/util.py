import numpy as np

def deNormImg(img):
    out = (img + 1) / 2
    return out.clip(0, 1)

def countParameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def calReceptiveField(model):
    layers = []
    receptive_field = 1
    stride_product = 1
    for _, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            layers.append([layer.stride[0], layer.kernel_size[0]])
        if isinstance(layer, nn.MaxPool2d):
            layers.append([2 , 2])
    for stride, kernel in layers[-1::-1]:
        stride_product *= stride
        receptive_field += (kernel -1) * stride_product
    return receptive_field

def calConfusionMatrix(pred_label, gt_label,class_num, additional = None):
    mat = np.zeros((class_num,class_num))
    mask = gt_label <class_num
    if additional is not None:
        mat += additional
    pred_flat, gt_flat= pred_label[mask].view(-1), gt_label[mask].view(-1)
    
    bin = pred_flat*class_num + gt_flat
    mat += np.bincount(np.array(bin),minlength=class_num*class_num).reshape(class_num,class_num)
    # for idx in range(pred_flat.size()[0]):
    #     mat[gt_flat[idx],pred_flat[idx]] +=1
    return mat

def calMetric(mat:np.ndarray):
    t_i = np.sum(mat,axis=1) #sum of row
    c_i = np.sum(mat,axis=0)#sum of column
    n_ii = np.diagonal(mat)
    n_cl = mat.shape[0]
    
    pixel_acc = np.sum(n_ii)/ np.sum(t_i)
    mean_acc = np.sum(n_ii/t_i)/n_cl
    mean_iu = np.sum(n_ii/(t_i + c_i - n_ii))/n_cl
    freq_weighted_iu = np.sum(t_i*n_ii/(t_i+c_i-n_ii))/np.sum(t_i)

    return pixel_acc, mean_acc, mean_iu, freq_weighted_iu
