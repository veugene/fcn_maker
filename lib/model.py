from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          Dense,
                          Permute,
                          Lambda,
                          add,
                          concatenate)
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.regularizers import l2
from keras.initializers import VarianceScaling
import numpy as np
from .blocks import (Convolution,
                     get_nonlinearity,
                     get_channel_axis,
                     bottleneck,
                     basic_block,
                     basic_block_mp,
                     residual_block,
                     unet_block,
                     vnet_block,
                     dense_block)


def _l2(decay):
    """
    Return a new instance of l2 regularizer, or return None
    """
    if decay is not None:
        return l2(decay)
    else:
        return None
    

def _softmax(x):
    """
    Softmax that works on ND inputs.
    """
    data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError("Unknown data_format " + str(data_format))
    if data_format=='channels_first':
        axis = -1
    else:
        axis = -2
    e = K.exp(x - K.max(x, axis=axis, keepdims=True))
    s = K.sum(e, axis=axis, keepdims=True)
    return e / s

def _unique(name):
    """
    Return a unique name string.
    """
    return name + '_' + str(K.get_uid(name))
    
    
def assemble_model(input_shape, num_classes, blocks,
                   preprocessor=None, postprocessor=None,
                   long_skip=True, long_skip_merge_mode='concat',
                   top_block_keep_resolution=False, init='he_normal',
                   weight_decay=0.0001, ndim=2, verbose=True):
    """
    input_shape : A tuple specifiying the image input shape.
    num_classes : The number of classes in the segmentation output. If None,
        no classifier will be assembled.
    blocks : A list of tuples, each containing a block function and a
        dictionary of keyword arguments to pass to it. The length must be
        odd-valued. The first set of blocks before the middle one is assumed
        to be on the downsampling (encoder) path; the second set after the
        middle one is assumed to be on the upsampling (decoder) path.
        If instead of a tuple, a None is passed, the corresponding block will
        simply preserve its input, passing it onto the next block.
    preprocessor : A block/layer/model. The model input is run through the 
        preprocessor before being passed to the first block.
    postprocessor : A block/layer/model. The output of the last block is passed
        through the postprocessor before being passed to the classifier.
    long_skip : A boolean specifying whether to use long skip connections
        from the downward path to the upward path. These can either concatenate
        or sum features across.
    long_skip_merge_mode : Either or 'sum', 'concat' features across skip.
    top_block_keep_resolution : If True, the top blocks in the decoder and 
        in the encoder keep the full resolution of the input image. Else, half
        resolution.
    init : A string specifying (or a function defining) the initializer for
        the layers that adapt features along long skip connections.
    weight_decay : The weight decay (L2 penalty) used in layers that adapt long
        skip connections.
    ndim : The spatial dimensionality of the input and output (either 2 or 3).
    verbose : A boolean specifying whether to print messages about model   
        structure during construction (if True).
    """
    
    '''
    Determine channel axis.
    '''
    channel_axis = get_channel_axis()
        
    '''
    Block list must be of odd length.
    '''
    if len(blocks)%2 != 1:
        raise ValueError('blocks list must be of odd length')
    
    '''
    `None` blocks are identity_block.
    '''
    blocks = list(blocks)
    for i, block in enumerate(blocks):
        if block is None:
            blocks[i] = (identity_block, {})
        
    '''
    ndim must be only 2 or 3.
    '''
    if ndim not in [2, 3]:
        raise ValueError("ndim must be either 2 or 3")
                
    '''
    Function to print if verbose==True
    '''
    def v_print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
        else:
            return None
    
    '''
    Helper function to create a long skip connection with concatenation.
    '''
    def make_long_skip(prev_x, concat_x, name=None):
        if long_skip_merge_mode == 'sum':
            num_target_filters = concat_x._keras_shape[channel_axis]
            if prev_x._keras_shape[channel_axis] != num_target_filters:
                prev_x = Convolution(filters=num_target_filters,
                                     kernel_size=1,
                                     ndim=ndim,
                                     kernel_initializer=init,
                                     padding='valid',
                                     kernel_regularizer=_l2(weight_decay),
                                     name=_unique(name+'_prev'))(prev_x)
        
        def _crop_to_fit(inputs):
            """
            Spatially crop a tensor's feature maps to the shape of the target
            tensor. Target is thus expected to be smaller.
            """
            x, target = inputs
            
            # Compute slices for cropping.
            indices = [slice(None, None)]*(ndim+2)
            spatial_dims = set(range(ndim+2)).difference([0, channel_axis])
            for dim in spatial_dims:
                indices[dim] = slice(0, target.shape[dim])
            
            # Crop.
            x = x[indices]
            return x
        
        # Crop upward path to match long skip resolution, if needed.
        cropped_shape = list(concat_x._keras_shape)
        cropped_shape[channel_axis] = prev_x._keras_shape[channel_axis]
        cropped_shape = cropped_shape[1:]
        crop = Lambda(_crop_to_fit, output_shape=cropped_shape)
        prev_x = crop([prev_x, concat_x])
        
        # Merge.
        if long_skip_merge_mode=='sum':
            merged = add([prev_x, concat_x])
        elif long_skip_merge_mode == 'concat':
            merged = concatenate([prev_x, concat_x], axis=channel_axis)
        else:
            raise ValueError("Unrecognized merge mode: {}"
                             "".format(long_skip_merge_mode))
        return merged
    
    '''
    Build all the blocks on the contracting and expanding paths.
    '''
    tensors = {}
    preprocessor_tensor = None
    depth = len(blocks)//2
    model_input = Input(shape=input_shape)
    
    # Preprocessor
    x = model_input
    if preprocessor is not None:
        x = preprocessor(x)
        preprocessor_tensor = x
        v_print("PRE - shape: {}".format(x._keras_shape))
    
    # Encoder (downsampling)
    for b in range(0, depth):
        func, kwargs = blocks[b]
        if b==0 and top_block_keep_resolution:
            subsample = False
        else:
            subsample = True
        x = func(**kwargs, subsample=subsample)(x)
        tensors[b] = x
        v_print("BLOCK {} - shape: {}".format(b, x._keras_shape))
        
    # Bottleneck
    func, kwargs = blocks[depth]
    x = func(**kwargs, subsample=True, upsample=True)(x)
    v_print("ACROSS {} - shape: {}".format(depth, x._keras_shape))
    
    # Decoder (upsampling)
    for b in range(0, depth):
        if long_skip:
            concat_x = tensors[depth-b-1]
            n_filters = concat_x._keras_shape[channel_axis]
            x = make_long_skip(prev_x=x,
                               concat_x=concat_x,
                               name=_unique('long_skip_{}'.format(depth-b-1)))
        if b==depth-1 and top_block_keep_resolution:
            upsample = False
        else:
            upsample = True
        func, kwargs = blocks[depth+b+1]
        x = func(**kwargs, upsample=upsample)(x)
        v_print("UP {} - shape: {}".format(depth-b-1, x._keras_shape))
        
    # Skip from preprocessor output to postprocessor input.
    if long_skip and preprocessor_tensor is not None:
        n_filters = preprocessor_tensor._keras_shape[channel_axis]
        x = make_long_skip(prev_x=x,
                           concat_x=preprocessor_tensor,
                           name=_unique('long_skip_top'))
        
    # Postprocessor
    if postprocessor is not None:
        x = postprocessor(x)
    
    # OUTPUT (SOFTMAX)
    if num_classes is not None:
        # Linear classifier
        output = Convolution(filters=num_classes,
                             kernel_size=1,
                             ndim=ndim,
                             activation='linear',
                             kernel_regularizer=_l2(weight_decay),
                             name=_unique('classifier_conv'))(x)
        if ndim==2:
            output = Permute((2,3,1))(output)
        else:
            output = Permute((2,3,4,1))(output)
        if num_classes==1:
            output = Activation('sigmoid')(output)
        else:
            output = Activation(_softmax)(output)
        if ndim==2:
            output_layer = Permute((3,1,2))
        else:
            output_layer = Permute((4,1,2,3))
            output_layer.name = _unique('output')
        output = Permute((3,1,2))(output)
    else:
        # No classifier
        output = x
    
    # MODEL
    model = Model(inputs=model_input, outputs=output)

    return model


def assemble_resunet(input_shape, num_classes, num_init_blocks,
                     num_main_blocks, main_block_depth, init_num_filters,
                     short_skip=True, long_skip=True,
                     long_skip_merge_mode='concat',
                     main_block=None, init_block=None, upsample_mode='repeat',
                     dropout=0., normalization=BatchNormalization,
                     norm_kwargs=None, weight_decay=None, init='he_normal',
                     nonlinearity='relu', ndim=2, verbose=True):
    """
    input_shape : A tuple specifiying the image input shape.
    num_classes : The number of classes in the segmentation output.
    num_init_blocks : The number of blocks of type init_block, above 
        main_blocks. These blocks always have the same number of channels as
        the first convolutional layer in the model. There are `num_init_blocks`
        of these at both the beginning and the end of the network.
    num_main_blocks : The number of blocks of type main_block, below
        init_blocks. These blocks double (halve) the number of channels at each
        downsampling (upsampling) after the first main_block. There are
        `num_main_blocks` of these both in the encoder and the decoder, on
        either side of the bottleneck.
    main_block_depth : An integer or list of integers specifying the number of
        repetitions of each main_block. A list must contain 2*num_main_blocks+1
        values (there are num_main_blocks on the contracting path and on the 
        expanding path, as well as as one on the across path). Zero is not a
        valid depth.
    init_num_filters : The number of filters in the first and last
        convolutions (preprocessor, postprocessor). Also the number of filters
        in every init_block. Each main_block doubles (halves) the number of 
        filters for each decrease (increase) in resolution.
    short_skip : A boolean specifying whether to use ResNet-like shortcut
        connections from the input of each block to its output. The inputs are
        summed with the outputs.
    long_skip : A boolean specifying whether to use long skip connections
        from the downward path to the upward path. These can either concatenate
        or sum features across.
    long_skip_merge_mode : Either or 'sum', 'concat' features across skip.
    main_block : A layer defining the main_block (bottleneck by default).
    init_block : A layer defining the init_block (basic_block_mp by default).
    upsample_mode : Either 'repeat' or 'conv'. With 'repeat', rows and colums
        are repeated as in nearest neighbour interpolation. With 'conv',
        upscaling is done via transposed convolution.
    dropout : A float [0, 1] specifying the dropout probability, introduced in
        every block.
    normalization : The normalization to apply to layers (by default: batch
        normalization). If None, no normalization is applied.
    norm_kwargs : Keyword arguments to pass to batch norm layers. For batch
        normalization, default momentum is 0.9.
    weight_decay : The weight decay (L2 penalty) used in every convolution 
        (float).
    init : A string specifying (or a function defining) the initializer for
        layers.
    nonlinearity : A string (or function defining) the nonlinearity.
    ndim : The spatial dimensionality of the input and output (either 2 or 3).
    verbose : A boolean specifying whether to print messages about model   
        structure during construction (if True).
    """
    
    '''
    Determine channel axis.
    '''
    channel_axis = get_channel_axis()
    
    '''
    By default, use depth 2 bottleneck for main_block
    '''
    if main_block is None:
        main_block = bottleneck
    if init_block is None:
        init_block = basic_block_mp
    
    '''
    main_block_depth can be a list per block or a single value 
    -- ensure the list length is correct (if list) or convert to list
    '''
    if hasattr(main_block_depth, '__len__'):
        if len(main_block_depth)!=2*num_main_blocks+1:
            raise ValueError("main_block_depth must have " 
                             "`2*num_main_blocks+1` values when " 
                             "passed as a list")
    else:
        main_block_depth = [main_block_depth]*(2*num_main_blocks+1)
        
    '''
    ndim must be only 2 or 3.
    '''
    if ndim not in [2, 3]:
        raise ValueError("ndim must be either 2 or 3")
            
    '''
    If BatchNormalization is used and norm_kwargs is not set, set default
    kwargs.
    '''
    if norm_kwargs is None:
        if normalization == BatchNormalization:
            norm_kwargs = {'momentum': 0.9,
                           'scale': True,
                           'center': True,
                           'axis': channel_axis}
        else:
            norm_kwargs = {}
            
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'skip': short_skip,
                    'dropout': dropout,
                    'weight_decay': weight_decay,
                    'normalization': normalization,
                    'norm_kwargs': norm_kwargs,
                    'upsample_mode': upsample_mode,
                    'nonlinearity': nonlinearity,
                    'init': init,
                    'ndim': ndim}
    
    '''
    Single convolution as preprocessor.
    '''
    preprocessor = Convolution(filters=init_num_filters,
                               kernel_size=3,
                               ndim=ndim,
                               kernel_initializer=init,
                               padding='same',
                               kernel_regularizer=_l2(weight_decay))
    
    '''
    Norm + nonlin + conv as postprocessor.
    '''
    def _postprocessor(x):
        out = normalization(**norm_kwargs)(x)
        out = get_nonlinearity(nonlinearity)(out)
        out = Convolution(filters=init_num_filters,
                          kernel_size=3,
                          ndim=ndim,
                          kernel_initializer=init,
                          padding='same',
                          kernel_regularizer=_l2(weight_decay))(out)
        return out
    postprocessor = _postprocessor
    
    '''
    Assemble all necessary blocks.
    '''
    blocks_down = []
    blocks_across = []
    blocks_up = []
    
    # Down, init_block
    for i in range(num_init_blocks):
        kwargs = {'block_function': init_block,
                  'filters': init_num_filters,
                  'repetitions': 1}
        kwargs.update(block_kwargs)
        blocks_down.append((residual_block, kwargs))
        
    # Down, main_block
    for i in range(num_main_blocks):
        kwargs = {'block_function': main_block,
                  'filters': init_num_filters*(2**i),
                  'repetitions': main_block_depth[i]}
        kwargs.update(block_kwargs)
        blocks_down.append((residual_block, kwargs))
        
    # Bottleneck, main_block
    kwargs = {'block_function': main_block,
              'filters': init_num_filters*(2**num_main_blocks),
              'repetitions': main_block_depth[num_main_blocks]}
    kwargs.update(block_kwargs)
    blocks_across.append((residual_block, kwargs))
    
    # Up, main_block
    for i in range(num_main_blocks-1, -1, -1):
        kwargs = {'block_function': main_block,
                  'filters': init_num_filters*(2**i),
                  'repetitions': main_block_depth[-i-1]}
        kwargs.update(block_kwargs)
        blocks_up.append((residual_block, kwargs))
    
    # Up, init_block
    for i in range(num_init_blocks):
        kwargs = {'block_function': init_block,
                  'filters': init_num_filters,
                  'repetitions': 1}
        kwargs.update(block_kwargs)
        blocks_up.append((residual_block, kwargs))
        
    blocks = blocks_down + blocks_across + blocks_up
    
    '''
    Assemble model.
    '''
    model = assemble_model(input_shape=input_shape,
                           num_classes=num_classes,
                           blocks=blocks,
                           preprocessor=preprocessor,
                           postprocessor=postprocessor,
                           long_skip=long_skip,
                           long_skip_merge_mode=long_skip_merge_mode,
                           ndim=ndim,
                           verbose=verbose)
    return model



def assemble_unet(input_shape, num_classes, init_num_filters=64,
                  num_pooling=4, short_skip=False, long_skip=True,
                  long_skip_merge_mode='concat', upsample_mode='repeat',
                  dropout=0., normalization=None, norm_kwargs=None,
                  weight_decay=None, init='he_normal', nonlinearity='relu',
                  ndim=2, verbose=True, **block_kwargs):
    """
    input_shape : A tuple specifiying the image input shape.
    num_classes : The number of classes in the segmentation output.
    init_num_filters : The number of filters in the first pair and last pair
        of convolutions in the network. With every downsampling, the number of
        filters is doubled; with every upsampling, it is halved.
    num_pooling : The number of pooling (and thus upsampling) operations to 
        perform in the network.
    short_skip : A boolean specifying whether to use ResNet-like shortcut
        connections from the input of each block to its output. The inputs are
        summed with the outputs.
    long_skip : A boolean specifying whether to use long skip connections
        from the downward path to the upward path. These can either concatenate
        or sum features across.
    long_skip_merge_mode : Either or 'sum', 'concat' features across skip.
    upsample_mode : Either 'repeat' or 'conv'. With 'repeat', rows and colums
        are repeated as in nearest neighbour interpolation. With 'conv',
        upscaling is done via transposed convolution.
    dropout : A float in [0, 1.] specifying the dropout probability in the 
        bottleneck and in the first subsequent block, as in the UNet.
    normalization : The normalization to apply to layers (none by default).
        Recommended to pass keras's BatchNormalization when using 
        short_skip==True.
    norm_kwargs : Keyword arguments to pass to normalization layers. If using
        BatchNormalization, kwargs are autoset with a momentum of 0.9.
    weight_decay : The weight decay (L2 penalty) used in every convolution 
        (float).
    init : A string specifying (or a function defining) the initializer for
        layers.
    nonlinearity : The nonlinearity to use, passed as a string or a function.
    ndim : The spatial dimensionality of the input and output (either 2 or 3).
    verbose : A boolean specifying whether to print messages about model   
        structure during construction (if True).
    """
    
    '''
    Determine channel axis.
    '''
    channel_axis = get_channel_axis()
    
    '''
    ndim must be only 2 or 3.
    '''
    if ndim not in [2, 3]:
        raise ValueError("ndim must be either 2 or 3")
            
    '''
    If BatchNormalization is used and norm_kwargs is not set, set default
    kwargs.
    '''
    if norm_kwargs is None:
        if normalization == BatchNormalization:
            norm_kwargs = {'momentum': 0.9,
                           'scale': True,
                           'center': True,
                           'axis': channel_axis}
        else:
            norm_kwargs = {}
            
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'skip': short_skip,
                   'weight_decay': weight_decay,
                   'normalization': normalization,
                   'norm_kwargs': norm_kwargs,
                   'nonlinearity': nonlinearity,
                   'upsample_mode': upsample_mode,
                   'init': init,
                   'ndim': ndim,
                   'halve_features_on_upsample': halve_features_on_upsample}
    
    '''
    No sub/up-sampling at beginning, end.
    '''
    preprocessor = unet_block(filters=init_num_filters, **block_kwargs)
    postprocessor = unet_block(filters=init_num_filters, **block_kwargs)
    
    '''
    Assemble all necessary blocks.
    '''
    blocks_down = []
    blocks_across = []
    blocks_up = []
    for i in range(1, num_pooling):
        kwargs = {'filters': init_num_filters*(2**i)}
        kwargs.update(block_kwargs)
        blocks_down.append((unet_block, kwargs))
    kwargs = {'filters': init_num_filters*(2**num_pooling),
              'dropout': dropout}
    kwargs.update(block_kwargs)
    blocks_across.append((unet_block, kwargs))
    for i in range(num_pooling-1, 0, -1):
        kwargs = {'filters': init_num_filters*(2**i)}
        if i==num_pooling-1:
            kwargs['dropout'] = dropout
        kwargs.update(block_kwargs)
        blocks_up.append((unet_block, kwargs))
    blocks = blocks_down + blocks_across + blocks_up
    
    '''
    Assemble model.
    '''
    model = assemble_model(input_shape=input_shape,
                           num_classes=num_classes,
                           blocks=blocks,
                           preprocessor=preprocessor,
                           postprocessor=postprocessor,
                           long_skip=long_skip,
                           long_skip_merge_mode=long_skip_merge_mode,
                           ndim=ndim,
                           verbose=verbose)
    return model


def assemble_vnet(input_shape, num_classes, init_num_filters=32,
                  num_pooling=4, short_skip=True, long_skip=True,
                  long_skip_merge_mode='concat', upsample_mode='repeat',
                  dropout=0., normalization=None, norm_kwargs=None,
                  init=VarianceScaling(scale=3., mode='fan_avg'),
                  weight_decay=None, nonlinearity='prelu', ndim=3,
                  verbose=True):
    """
    input_shape : A tuple specifiying the image input shape.
    num_classes : The number of classes in the segmentation output.
    init_num_filters : The number of filters in the first pair and last pair
        of convolutions in the network. With every downsampling, the number of
        filters is doubled; with every upsampling, it is halved.
    num_pooling : The number of pooling (and thus upsampling) operations to 
        perform in the network.
    short_skip : A boolean specifying whether to use ResNet-like shortcut
        connections from the input of each block to its output. The inputs are
        summed with the outputs.
    long_skip : A boolean specifying whether to use long skip connections
        from the downward path to the upward path. These can either concatenate
        or sum features across.
    long_skip_merge_mode : Either or 'sum', 'concat' features across skip.
    upsample_mode : Either 'repeat' or 'conv'. With 'repeat', rows and colums
        are repeated as in nearest neighbour interpolation. With 'conv',
        upscaling is done via transposed convolution.
    dropout : A float in [0, 1.], specifying dropout probability.
    normalization : The normalization to apply to layers (none by default).
        Recommended to pass keras's BatchNormalization when using 
        short_skip==True.
    norm_kwargs : Keyword arguments to pass to normalization layers. If using
        BatchNormalization, kwargs are autoset with a momentum of 0.9.
    init : A string specifying (or a function defining) the initializer for
        layers.
    weight_decay : The weight decay (L2 penalty) used in every convolution 
        (float).
    nonlinearity : The nonlinearity to use, passed as a string or a function.
    ndim : The spatial dimensionality of the input and output (either 2 or 3).
    verbose : A boolean specifying whether to print messages about model   
        structure during construction (if True).
    """
    
    '''
    Determine channel axis.
    '''
    channel_axis = get_channel_axis()
    
    '''
    ndim must be only 2 or 3.
    '''
    if ndim not in [2, 3]:
        raise ValueError("ndim must be either 2 or 3")
            
    '''
    If BatchNormalization is used and norm_kwargs is not set, set default
    kwargs.
    '''
    if norm_kwargs is None:
        if normalization == BatchNormalization:
            norm_kwargs = {'momentum': 0.9,
                           'scale': True,
                           'center': True,
                           'axis': channel_axis}
        else:
            norm_kwargs = {}
            
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'skip': short_skip,
                    'weight_decay': weight_decay,
                    'normalization': normalization,
                    'norm_kwargs': norm_kwargs,
                    'init': init,
                    'nonlinearity': nonlinearity,
                    'upsample_mode': upsample_mode,
                    'dropout': dropout,
                    'ndim': ndim}
    
    '''
    No sub/up-sampling at beginning, end.
    '''
    kwargs = {'num_conv': 1}
    kwargs.update(block_kwargs)
    preprocessor = vnet_block(filters=init_num_filters, **kwargs)
    postprocessor = vnet_block(filters=init_num_filters, **kwargs)
    
    '''
    Assemble all necessary blocks.
    '''
    blocks_down = []
    blocks_across = []
    blocks_up = []
    for i in range(1, num_pooling):
        kwargs = {'filters': init_num_filters*(2**i)}
        if i==1:
            kwargs['num_conv'] = 2
        else:
            kwargs['num_conv'] = 3
        kwargs.update(block_kwargs)
        blocks_down.append((vnet_block, kwargs))
    kwargs = {'filters': init_num_filters*(2**num_pooling),
              'num_conv': 3}
    kwargs.update(block_kwargs)
    blocks_across.append((vnet_block, kwargs))
    for i in range(num_pooling-1, 0, -1):
        kwargs = {'filters': init_num_filters*(2**i)}
        if i==1:
            kwargs['num_conv'] = 2
        else:
            kwargs['num_conv'] = 3
        kwargs.update(block_kwargs)
        blocks_up.append((vnet_block, kwargs))
    blocks = blocks_down + blocks_across + blocks_up
    
    '''
    Assemble model.
    '''
    model = assemble_model(input_shape=input_shape,
                           num_classes=num_classes,
                           blocks=blocks,
                           preprocessor=preprocessor,
                           postprocessor=postprocessor,
                           long_skip=long_skip,
                           long_skip_merge_mode=long_skip_merge_mode,
                           ndim=ndim,
                           verbose=verbose)
    return model


def assemble_fcdensenet(input_shape, num_classes, block_depth, 
                        num_blocks=11, init_num_filters=48, growth_rate=16,
                        long_skip=True, skip_merge_mode='concat', 
                        upsample_mode='repeat', dropout=0.,
                        normalization=BatchNormalization, norm_kwargs=None,
                        init='he_uniform', weight_decay=None,
                        nonlinearity='relu', ndim=2, verbose=True):
    """
    input_shape : A tuple specifiying the image input shape.
    num_classes : The number of classes in the segmentation output.
    block_depth : An integer or list of integers specifying the number of
        convolutions in each dense block. A list must contain num_blocks values
        (there are an equal number of blocks on the contracting and expanding
        paths, as well as as one bottleneck on the across path). Zero is a
        valid depth (the block still sub/up-samples).
    num_blocks : The total number of dense blocks in the network. Must be an
        odd number.
    init_num_filters : The number of filters in the first pair and last pair
        of convolutions in the network. With every downsampling, the number of
        filters is doubled; with every upsampling, it is halved.
    growth_rate : The linear rate with which the number of filters increases
        after each convolution, when using 'concat' skip_merge_mode.
        In 'sum' mode, this argument simply sets the number of filters for
        every convolution layer except the first and last ones (preprocessor
        and postprocessor).
        If set to None, the number of filters for each dense_block will double
        after each pooling operation and halve after each upsampling operation.
    long_skip : A boolean specifying whether to use long skip connections
        from the downward path to the upward path. These can either concatenate
        or sum features across.
    skip_merge_mode : Either or 'sum', 'concat' features across skip.
    upsample_mode : Either 'repeat' or 'conv'. With 'repeat', rows and colums
        are repeated as in nearest neighbour interpolation. With 'conv',
        upscaling is done via transposed convolution.
    dropout : A float in [0, 1.], specifying dropout probability.
    normalization : The normalization to apply to layers (none by default).
        Recommended to pass keras's BatchNormalization when using 
        short_skip==True.
    norm_kwargs : Keyword arguments to pass to normalization layers. If using
        BatchNormalization, kwargs are autoset with a momentum of 0.9.
    init : A string specifying (or a function defining) the initializer for
        layers.
    weight_decay : The weight decay (L2 penalty) used in every convolution 
        (float).
    nonlinearity : The nonlinearity to use, passed as a string or a function.
    ndim : The spatial dimensionality of the input and output (either 2 or 3).
    verbose : A boolean specifying whether to print messages about model   
        structure during construction (if True).
    """
    
    '''
    Determine channel axis.
    '''
    channel_axis = get_channel_axis()
        
    '''
    Make sure num_blocks is odd.
    '''
    if not num_blocks % 2:
        raise ValueError("`num_blocks` must be odd")
    
    '''
    block_depth can be a list per block or a single value 
    -- ensure the list length is correct (if list) or convert to list
    '''
    if hasattr(block_depth, '__len__'):
        if len(block_depth)!=num_blocks:
            raise ValueError("block_depth must have `num_blocks` values when " 
                             "passed as a list")
    else:
        block_depth = [block_depth]*num_blocks
        
    '''
    ndim must be only 2 or 3.
    '''
    if ndim not in [2, 3]:
        raise ValueError("ndim must be either 2 or 3")
            
    '''
    If BatchNormalization is used and norm_kwargs is not set, set default
    kwargs.
    '''
    if norm_kwargs is None:
        if normalization == BatchNormalization:
            norm_kwargs = {'momentum': 0.9,
                           'scale': True,
                           'center': True,
                           'axis': channel_axis}
        else:
            norm_kwargs = {}
            
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'dropout': dropout,
                    'weight_decay': weight_decay,
                    'normalization': normalization,
                    'norm_kwargs': norm_kwargs,
                    'upsample_mode': upsample_mode,
                    'skip_merge_mode': skip_merge_mode,
                    'nonlinearity': nonlinearity,
                    'init': init,
                    'ndim': ndim}
    if growth_rate is not None:
        block_kwargs['filters'] = growth_rate
    
    '''
    Single convolution as preprocessor.
    '''
    def _preprocessor(x):
        out = Convolution(filters=init_num_filters,
                          kernel_size=3,
                          ndim=ndim,
                          kernel_initializer=init,
                          padding='same',
                          kernel_regularizer=_l2(weight_decay))(x)
        return out
    preprocessor = _preprocessor
    
    '''
    No postprocessor!
    '''
    postprocessor = lambda x:x
    
    '''
    Assemble all necessary blocks.
    '''
    blocks_down = []
    blocks_across = []
    blocks_up = []
    n_blocks_side = num_blocks//2   # on one side
    
    # Down (Encoder)
    for i in range(0, n_blocks_side):
        kwargs = {'block_depth': block_depth[i],
                  'merge_input': True}
        if growth_rate is None:
            kwargs['filters'] = init_num_filters*(2**i)
        kwargs.update(block_kwargs)
        blocks_down.append((dense_block, kwargs))
    
    # NOTE: merge_input should be False for bottleneck and decoder.
    
    # Bottleneck, main_block
    kwargs = {'block_depth': block_depth[n_blocks_side],
              'merge_input': False}
    if growth_rate is None:
        kwargs['filters'] = init_num_filters*(2**n_blocks_side)
    kwargs.update(block_kwargs)
    blocks_across.append((dense_block, kwargs))
    
    # Up (Decoder)
    for i in range(n_blocks_side-1, -1, -1):
        kwargs = {'block_depth': block_depth[-i-1],
                  'merge_input': False}
        if growth_rate is None:
            kwargs['filters'] = init_num_filters*(2**i)
        kwargs.update(block_kwargs)
        blocks_up.append((dense_block, kwargs))
        
    blocks = blocks_down + blocks_across + blocks_up
    
    '''
    Assemble model.
    '''
    model = assemble_model(input_shape=input_shape,
                           num_classes=num_classes,
                           blocks=blocks,
                           preprocessor=preprocessor,
                           postprocessor=postprocessor,
                           long_skip=long_skip,
                           long_skip_merge_mode=skip_merge_mode,
                           top_block_keep_resolution=True,
                           ndim=ndim,
                           verbose=verbose)
    return model
