from __future__ import (print_function,
                        division)
from keras import backend as K
import numpy as np


def dice_loss(target_class=1, mask_class=None):
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
    if not hasattr(target_class, '__len__'):
        target_class = [target_class]
    if mask_class is not None and not hasattr(mask_class, '__len__'):
        mask_class = [mask_class]
    
    # Define the keras expression.
    def dice(y_true, y_pred):
        smooth = 1
        
        # If needed, change ground truth from categorical to integer format.
        if K.ndim(y_true) > K.ndim(y_pred):
            data_format = K.image_data_format()
            if data_format=='channels_first':
                class_axis = 1
            elif data_format=='channels_last':
                class_axis = K.ndim(y_true)-1
            else:
                raise ValueError("Unknown data_format {}".format(data_format))
            y_true = K.argmax(y_true, axis=class_axis)
        y_true_f = K.flatten(y_true)
        y_true_f = K.cast(y_true_f, 'int32')
        y_pred_f = K.flatten(y_pred)
        y_target = K.sum([K.equal(y_true_f, t) for t in target_class],
                         axis=0)
        if mask_class is not None:
            mask_out = K.sum([K.equal(y_true_f, t) for t in mask_class],
                             axis=0)
            idxs = K.equal(mask_out, 0).nonzero()
            y_target = y_target[idxs]
            y_pred_f = y_pred_f[idxs]
        intersection = K.sum(y_target * y_pred_f)
        return -(2.*intersection+smooth) / \
                (K.sum(y_target)+K.sum(y_pred_f)+smooth)
    
    # Set a custom function name
    tag = "_"+"_".join(str(i) for i in target_class)
    if mask_class is not None:
        tag += "_"+"_".join("m"+str(i) for i in mask_class)
    dice.__name__ = "dice_loss"+tag
    
    return dice

