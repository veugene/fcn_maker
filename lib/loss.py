from keras import backend as K
from theano import tensor as T
import numpy as np


def categorical_crossentropy(weighted=False, masked_class=None):
    '''
    Categorical crossentropy for N-dimensional inputs.
    
    weighted : if True, loss is automatically reweighted with respecte to
        classes so that each class is given equal importance (simulates class
        balancing).
    masked_class : the (integer) class(es) that is/are masked out of the loss.
    '''
    if masked_class is not None and not hasattr(masked_class, '__len__'):
        masked_class = [masked_class]
    
    def categorical_crossentropy(y_true, y_pred):
        shape_y_pred_f = (K.prod(K.shape(y_pred)[:-1]), K.shape(y_pred)[-1])
        y_pred_f = K.reshape(y_pred, shape_y_pred_f)
        n_classes = y_pred.shape[1]
        if y_true.ndim==y_pred.ndim-1:
            y_true = K.one_hot(y_true, n_classes)
            dim_order = [0, y_true.ndim-1]+list(range(1, y_true.ndim-1))
            y_true = K.permute_dimensions(y_true, dim_order)
        y_true_f = K.flatten(y_true)
        y_true_f = K.cast(y_true_f, 'int32')
        cce = K.categorical_crossentropy(y_pred_f, y_true_f)
        if weighted:
            # inverse proportion
            non_class_axes = [i for i in range(y_true.ndim) if i!=1]
            class_weights = K.sum(y_true) / K.sum(y_true,
                                                  axis=non_class_axes,
                                                  keepdims=True)
            # weights sum to 1
            class_weights = class_weights.flatten() / T.sum(class_weights)
            weighted_y_true = y_true*class_weights
            sample_weights = T.max(weighted_y_true, axis=1)
            wcce = cce*sample_weights
        if masked_class is not None:
            mask_out = K.sum([K.equal(y_true_f, t) for t in masked_class],
                             axis=0)
            idxs = K.not_equal(mask_out, 1).nonzero()
            wcce = cce[idxs]
        return K.mean(wcce)
    
    # Set a custom function name
    tag = ""
    if masked_class is not None:
        tag += "_"+"_".join("m"+str(i) for i in masked_class)
    dice.__name__ = "categorical_crossentropy"+tag
    
    return categorical_crossentropy


def dice_loss(target_class=1, masked_class=None):
    '''
    Dice loss.
    
    Expects integer class labeling in y_true.
    Expects outputs in range [0, 1] in y_pred.
    
    Computes the soft dice loss considering all classes in target_class as one
    aggregate target class and ignoring all elements with ground truth classes
    in masked_class.
    
    target_class : integer or list
    masked_class : integer or list
    '''
    if not hasattr(target_class, '__len__'):
        target_class = [target_class]
    if masked_class is not None and not hasattr(masked_class, '__len__'):
        masked_class = [masked_class]
    
    # Define the keras expression.
    def dice(y_true, y_pred):
        smooth = 1
        y_true_f = K.flatten(y_true)
        y_true_f = K.cast(y_true_f, 'int32')
        y_pred_f = K.flatten(y_pred)
        y_target = K.sum([K.equal(y_true_f, t) for t in target_class], axis=0)
        if masked_class is not None:
            mask_out = K.sum([K.equal(y_true_f, t) for t in masked_class], 
                             axis=0)
            idxs = K.not_equal(mask_out, 1).nonzero()
            y_target = y_target[idxs]
            y_pred_f = y_pred_f[idxs]
        intersection = K.sum(y_target * y_pred_f)
        return -(2.*intersection+smooth) / \
                (K.sum(y_target)+K.sum(y_pred_f)+smooth)
    
    # Set a custom function name
    tag = "_"+"_".join(str(i) for i in target_class)
    if masked_class is not None:
        tag += "_"+"_".join("m"+str(i) for i in masked_class)
    dice.__name__ = "dice_loss"+tag
    
    return dice

