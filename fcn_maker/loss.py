from __future__ import (print_function,
                        division)
import torch
import numpy as np


class dice_loss(torch.nn.Module):
    '''
    Dice loss.
    
    Expects integer or one-hot class labeling in y_true.
    Expects outputs in range [0, 1] in y_pred.
    
    Computes the soft dice loss considering all classes in target_class as one
    aggregate target class and ignoring all elements with ground truth classes
    in mask_class.
    
    target_class : integer or list
    mask_class : integer or list
    '''
    def __init__(self, target_class=1, mask_class=None):
        super(dice_loss, self).__init__()
        if not hasattr(target_class, '__len__'):
            target_class = [target_class]
        if mask_class is not None and not hasattr(mask_class, '__len__'):
            mask_class = [mask_class]
        self.target_class = target_class
        self.mask_class = mask_class
        self.smooth = 1
            
    def forward(self, y_pred, y_true):
        # If needed, change ground truth from categorical to integer format.
        if y_true.ndimension() > y_pred.ndimension():
            y_true = torch.max(y_true, axis=1)[1]   # argmax
            
        # Flatten all inputs.
        y_true_f = y_true.view(-1).int()
        y_pred_f = y_pred.view(-1)
        
        # Aggregate target classes, mask out classes in mask_class.
        y_target = torch.sum([torch.equal(y_true_f, t) for t in target_class],
                              axis=0)
        if mask_class is not None:
            mask_out = torch.sum([K.equal(y_true_f, t) for t in mask_class],
                                  axis=0)
            idxs = torch.equal(mask_out, 0).nonzero()
            y_target = y_target[idxs]
            y_pred_f = y_pred_f[idxs]
        
        # Compute dice value.
        intersection = torch.sum(y_target * y_pred_f)
        dice_val = -(2.*intersection+self.smooth) / \
                    (torch.sum(y_target)+torch.sum(y_pred_f)+self.smooth)
                    
        return dice_val
