from keras import backend as K
from theano import tensor as T
import numpy as np

def categorical_crossentropy_ND(y_true, y_pred):
    '''
    y_true must use an integer class representation
    y_pred must use a one-hot class representation
    '''
    shp_y_pred = K.shape(y_pred)
    y_pred_flat = K.reshape(y_pred, (K.prod(shp_y_pred[:-1]), shp_y_pred[-1]))
    y_true_flat = K.flatten(y_true)
    y_true_flat = K.cast(y_true_flat, 'int32')
    out = K.categorical_crossentropy(y_pred_flat, y_true_flat)
    return K.mean(out)


def dice_loss(y_true, y_pred):
    '''
    Dice loss -- works for only binary classes.
    Expects integer 0/1 class labeling in y_true, y_pred.
    '''
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_true_f = K.cast(y_true_f, 'int32')
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -(2.*intersection+smooth) / (K.sum(y_true_f)+K.sum(y_pred_f)+smooth)


def masked_dice_loss(y_true, y_pred):
    '''
    Dice loss -- works for only binary classes.
    Expects integer 1/2 class labeling in y_true, y_pred.
    
    Class 0 is masked out.
    
    NOTE: THEANO only
    '''
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_true_f = K.cast(y_true_f, 'int32')
    y_pred_f = K.flatten(y_pred)
    idxs = K.gt(y_true_f, 0).nonzero()
    y_true_f = y_true_f[idxs] - 1
    y_pred_f = y_pred_f[idxs]
    intersection = K.sum(y_true_f * y_pred_f)
    return -(2.*intersection+smooth) / (K.sum(y_true_f)+K.sum(y_pred_f)+smooth)


def cce_with_regional_penalty(weight, power, nb_row, nb_col):
    '''
    y_true has shape [batch_size, 1, w, h]
    y_pred has shape [batch_size, num_classes, w, h]
    
    NOTE: THEANO only
    '''
    def f(y_true, y_pred):
        loss = categorical_crossentropy_ND(y_true, y_pred)
        
        y_true_flat = y_true.flatten()
        y_true_flat = K.cast(y_true_flat, 'int32')
        y_true_onehot_flat = T.extra_ops.to_one_hot(y_true_flat,
                                                    nb_class=y_pred.shape[-1])
        y_true_onehot = K.reshape(y_true_onehot_flat, y_pred.shape)
        
        abs_err = K.abs(y_true_onehot-y_pred)
        abs_err = K.permute_dimensions(abs_err, [0,3,1,2])
        kernel = K.ones((2, 2, nb_row, nb_col)) / np.float32(nb_row*nb_col)
        conv = K.conv2d(abs_err, kernel, strides=(1, 1), border_mode='same')
        penalty = K.pow(conv, power)
        return (1-weight)*loss + weight*K.mean(penalty)
    return f 
