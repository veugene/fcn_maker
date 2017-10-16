from keras import backend as K
import numpy as np


# Define axes according to keras.
batch_axis = 0
class_axis = 1 if K.image_data_format()=='channels_first' else -1


def categorical_crossentropy(weighted=False, mask_class=None):
    '''
    Categorical crossentropy for N-dimensional inputs.
    
    Expects integer or one-hot class labeling in y_true.
    
    weighted : if True, loss is automatically reweighted with respect to
        classes so that each class is given equal importance (simulates class
        balancing).
    mask_class : the (integer) class(es) that is/are masked out of the loss.
    '''
    if mask_class is not None and not hasattr(mask_class, '__len__'):
        mask_class = [mask_class]
    
    def categorical_crossentropy(y_true, y_pred):
        shape_y_pred_f = (K.prod(K.shape(y_pred)[:-1]), K.shape(y_pred)[-1])
        y_pred_f = K.reshape(y_pred, shape_y_pred_f)
        n_classes = y_pred.shape[1]
        if K.ndim(y_true)==K.ndim(y_pred)-1:
            y_true = K.one_hot(y_true, n_classes)
            dim_order = [0, K.ndim(y_true)-1]+list(range(1, K.ndim(y_true)-1))
            y_true = K.permute_dimensions(y_true, dim_order)
        y_true_f = K.flatten(y_true)
        y_true_f = K.cast(y_true_f, 'int32')
        cce = K.categorical_crossentropy(y_pred_f, y_true_f)
        if weighted:
            # inverse proportion
            non_class_axes = [i for i in range(K.ndim(y_true)) if i!=class_axis]
            class_weights = K.sum(y_true) / K.sum(y_true,
                                                  axis=non_class_axes,
                                                  keepdims=True)
            # weights sum to 1
            class_weights = class_weights.flatten() / K.sum(class_weights)
            weighted_y_true = y_true*class_weights
            sample_weights = K.max(weighted_y_true, axis=class_axis)
            wcce = cce*sample_weights
        if mask_class is not None:
            mask_out = K.sum([K.equal(y_true_f, t) for t in mask_class],
                             axis=batch_axis)
            idxs = K.not_equal(mask_out, 1).nonzero()
            wcce = cce[idxs]
        return K.mean(wcce)
    
    # Set a custom function name
    tag = ""
    if mask_class is not None:
        tag += "_"+"_".join("m"+str(i) for i in mask_class)
    dice.__name__ = "categorical_crossentropy"+tag
    
    return categorical_crossentropy


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
        if K.ndim(y_true)==K.ndim(y_pred):
            # Change ground truth from categorical to integer format.
            y_true = K.argmax(y_true, axis=class_axis)
        y_true_f = K.flatten(y_true)
        y_true_f = K.cast(y_true_f, 'int32')
        y_pred_f = K.flatten(y_pred)
        y_target = K.sum([K.cast(K.equal(y_true_f, t), K.floatx()) \
                                       for t in target_class], axis=batch_axis)
        if mask_class is not None:
            mask_out = K.sum([K.cast(K.equal(y_true_f, t), K.floatx()) \
                                       for t in mask_class], axis=batch_axis)
            idxs = K.not_equal(mask_out, 1).nonzero()
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

