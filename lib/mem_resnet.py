from keras.layers import (Convolution2D,
                          merge,
                          UpSampling2D,
                          Permute,
                          Activation,
                          Lambda,
                          Dropout,
                          BatchNormalization)
from keras.regularizers import l2




# Subclassing convolution to allow bias initialization
class bConvolution2D(Convolution2D):
    def __init__(self, *args, init_bias=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_bias = None
        if init_bias is not None:
            self.init_bias = [init_bias]*self.nb_filter
        
    def build(self, input_shape):
        super().build(input_shape)
        if self.bias and self.init_bias is not None:
            self.b.set_value(self.init_bias)
            
            
# Return a new instance of l2 regularizer, or return None
def _l2(decay):
    if decay is not None:
        return l2(decay)
    else:
        return None


# (Partially) forget cell contents and writes new contents
def _update(x, cell, init_bias=None, weight_decay=None):
    nb_filter = x._keras_shape[1]
    update_gate = bConvolution2D(nb_filter, 1, 1, init='he_normal',
                                 init_bias=init_bias,
                                 activation='sigmoid', border_mode='valid',
                                 W_regularizer=_l2(weight_decay))(x)
    cell_after_write = merge([cell, x, update_gate],
                            mode=lambda ins: ins[2]*ins[0] + (1-ins[2])*ins[1],
                            output_shape=lambda x:x[0])
    return cell_after_write

# Read cell contents (gated)
def _read(x, cell, init_bias=None, weight_decay=None):
    nb_filter = x._keras_shape[1]
    read_gate = bConvolution2D(nb_filter, 1, 1, init='he_normal',
                               init_bias=init_bias,
                               activation='sigmoid', border_mode='valid',
                               W_regularizer=_l2(weight_decay))(x)
    read_from_cell = merge([cell, read_gate], mode='mul')
    return read_from_cell
        

# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual, weight_decay=None):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[2] // residual._keras_shape[2]
    stride_height = input._keras_shape[3] // residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1],
                                 nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init='he_normal', border_mode='valid', bias=False,
                                 W_regularizer=_l2(weight_decay))(input)
        #shortcut = BatchNormalization(mode=0, axis=1, momentum=0.9)(shortcut)

    return merge([shortcut, residual], mode='sum')


def bn_relu_conv(nb_filter, nb_row, nb_col, subsample=False, upsample=False,
                 batch_norm=True, bias=False, weight_decay=None, momentum=0.9):
    def f(input):
        processed = input
        if batch_norm:
            processed = BatchNormalization(mode=0, axis=1,
                                           momentum=momentum)(processed)
        processed = Activation('relu')(processed)
        stride = (1, 1)
        if subsample:
            stride = (2, 2)
        if upsample:
            processed = UpSampling2D(size=(2, 2))(processed)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col,
                             subsample=stride, init='he_normal',
                             border_mode='same', bias=bias,
                             W_regularizer=_l2(weight_decay))(processed)
    return f


# Build an (optionally residual) block with a memory cell
def block(nb_filter, residual=True, subsample=False, memory=True,
          bottleneck=False, dropout=0., weight_decay=None, momentum=0.9,
          update_bias=None, read_bias=None):
    def f(input_x, input_cell):
        # Batch norm on input
        normalized_x = BatchNormalization(mode=0, axis=1,
                                          momentum=momentum)(input_x)
        
        # Write to memory cell
        if memory:
            updated_cell = _update(normalized_x, input_cell,
                                   init_bias=update_bias,
                                   weight_decay=weight_decay)
        # Process x
        if bottleneck:
            processed_x = bn_relu_conv(nb_filter, 1, 1,
                                     subsample=subsample, batch_norm=False,
                                     weight_decay=weight_decay,
                                     momentum=momentum)(normalized_x)
            processed_x = bn_relu_conv(nb_filter, 3, 3,
                                     subsample=False, batch_norm=True,
                                     weight_decay=weight_decay,
                                     momentum=momentum)(processed_x)
            if dropout:
                processed_x = Dropout(dropout)(processed_x)
            processed_x = bn_relu_conv(4*nb_filter, 1, 1,
                                     subsample=False, batch_norm=True,
                                     weight_decay=weight_decay,
                                     momentum=momentum)(processed_x)
        else:
            processed_x = bn_relu_conv(nb_filter, 3, 3,
                                     subsample=subsample, batch_norm=False,
                                     weight_decay=weight_decay,
                                     momentum=momentum)(normalized_x)
            if dropout:
                processed_x = Dropout(dropout)(processed_x)
            processed_x = bn_relu_conv(nb_filter, 3, 3,
                                     subsample=False, batch_norm=True,
                                     weight_decay=weight_decay,
                                     momentum=momentum)(processed_x)
        
        output = processed_x
        if residual:
            output = _shortcut(input_x, output, weight_decay=weight_decay)
        if memory:
            # Adjust memory size if needed
            if subsample or \
               updated_cell._keras_shape[1] != output._keras_shape[1]:
                subsample_tuple=(1, 1)
                if subsample:
                    subsample_tuple=(2, 2)
                updated_cell = Convolution2D(output._keras_shape[1], 1, 1,
                                     init='he_normal', activation='linear',
                                     border_mode='valid',
                                     subsample=subsample_tuple,
                                     W_regularizer=_l2(weight_decay))(updated_cell)
            
            # Read from memory cell
            memory_read = _read(output, updated_cell, init_bias=read_bias,
                                weight_decay=weight_decay)
            
            output = merge([output, memory_read], mode='sum')
            return output, updated_cell
        else:
            return output, None
    
    return f


# Build a stack of blocks.
def stackenblochen(nb_filter, repetitions, residual=True, subsample=False,
                   memory=True, bottleneck=False, dropout=0.,
                   weight_decay=None, update_bias=None, read_bias=None,
                   momentum=0.9):
    def f(x, cell):
        for i in range(repetitions):
            kwargs = {'nb_filter' : nb_filter,
                      'residual' : residual,
                      'subsample': (i==0 and subsample), 
                      'memory' : memory,
                      'bottleneck' : bottleneck,
                      'dropout' : dropout,
                      'weight_decay' : weight_decay,
                      'update_bias' : update_bias,
                      'read_bias' : read_bias,
                      'momentum' : momentum}
            x, cell = block(**kwargs)(x, cell)
        return x, cell
    return f
