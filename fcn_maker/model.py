from __future__ import (print_function,
                        division)
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from .blocks import (convolution,
                     batch_normalization,
                     get_nonlinearity,
                     merge,
                     crop_stack,
                     bottleneck,
                     basic_block,
                     tiny_block,
                     repeat_block,
                     unet_block,
                     vnet_block,
                     dense_block,
                     identity_block)
    
    
class fcn(torch.nn.Module):
    """
    in_channels : Number of channels in the input.
    num_classes : The number of classes in the segmentation output. If None,
        no classifier will be assembled.
    blocks : A list of tuples, each containing a block function and a
        dictionary of keyword arguments to pass to it. The length must be
        odd-valued. The first set of blocks before the middle one is assumed
        to be on the downsampling (encoder) path; the second set after the
        middle one is assumed to be on the upsampling (decoder) path. The 
        first and last block do not change resolution. If instead of a tuple,
        a None is passed, the corresponding block will simply preserve its
        input, passing it onto the next block.
    long_skip : A boolean specifying whether to use long skip connections
        from the downward path to the upward path. These can either concatenate
        or sum features across.
    long_skip_merge_mode : Either or 'sum', 'concat' features across skip.
    conv_padding : Whether to use zero-padding for convolutions. If True, the
        output size is the same as the input size; if False, the output is
        smaller than the input size.
    init : A string specifying (or a function defining) the initializer for
        the layers that adapt features along long skip connections.
    ndim : The spatial dimensionality of the input and output (either 2 or 3).
    verbose : A boolean specifying whether to print messages about model   
        structure during construction (if True).
    """
    def __init__(self, in_channels, num_classes, blocks,
                 long_skip=True, long_skip_merge_mode='concat',
                 conv_padding=True, init='kaiming_normal', ndim=2,
                 verbose=True):
        super(fcn, self).__init__()
        
        # Block list must be of odd length.
        if len(blocks)%2 != 1:
            raise ValueError('blocks list must be of odd length')
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("ndim must be either 2 or 3")
        
        # `None` block tuples are replaced by identity_block.
        last_out_channels = in_channels
        blocks = list(blocks)
        for i, block in enumerate(blocks):
            if block is None:
                blocks[i] = (identity_block,
                             {'num_filters': last_out_channels})
                last_out_channels = blocks[i].out_channels
            
        self.in_channels = in_channels
        self.out_channels = None            # Computed later.
        self.num_classes = num_classes
        self.blocks = blocks
        self.long_skip = long_skip
        self.long_skip_merge_mode = long_skip_merge_mode
        self.conv_padding = conv_padding
        self.init = init
        self.ndim = ndim
        self.verbose = verbose
        
        # Function to print if verbose==True
        def v_print(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)
            else:
                return None
        
        '''
        Instantiate blocks.
        '''
        self.blocks_instantiated = []
        depth = len(blocks)//2
        
        # Encoder (downsampling)
        last_out_channels = in_channels
        for b in range(0, depth):
            func, kwargs = blocks[b]
            block = func(subsample=True if b>0 else False,
                         in_channels=last_out_channels,
                         **kwargs)
            last_out_channels = block.out_channels
            self.blocks_instantiated.append(block)
            self._modules['down_{}'.format(b)] = block
            v_print("DOWN {} - {} : in_channels={}, out_channels={}"
                    "".format(b, func,
                              block.in_channels, block.out_channels))
        
        # Bottleneck
        func, kwargs = blocks[depth]
        block = func(subsample=True,
                     upsample=True,
                     in_channels=last_out_channels,
                     **kwargs)
        last_out_channels = block.out_channels
        self.blocks_instantiated.append(block)
        self._modules['across'] = block
        v_print("ACROSS {} - {} : in_channels={}, out_channels={}"
                "".format(depth, func,
                          block.in_channels, block.out_channels))
                
        # Decoder (upsampling)
        for b in range(0, depth):
            if long_skip:
                concat_block = self._modules['down_{}'.format(depth-b-1)]
                concat_channels = concat_block.out_channels
                if long_skip_merge_mode=='concat':
                    last_out_channels += concat_channels
                elif long_skip_merge_mode=='sum':
                    last_out_channels = concat_channels
            func, kwargs = blocks[depth+b+1]
            block = func(upsample=True if b<depth-1 else False,
                         in_channels=last_out_channels,
                         **kwargs)
            last_out_channels = block.out_channels
            self.blocks_instantiated.append(block)
            self._modules['up_{}'.format(depth-b-1)] = block
            v_print("UP {} - {} : in_channels={}, out_channels={}"
                    "".format(b, func,
                              block.in_channels, block.out_channels))
        
        # Identify and set out_channels.
        self.out_channels = block.out_channels
                
        '''
        Helper function to create a long skip connection with concatenation.
        '''
        class merge_long_skip(torch.nn.Module):
            def __init__(self, in_channels, concat_channels,
                         long_skip_merge_mode):
                super(merge_long_skip, self).__init__()
                self.in_channels = in_channels
                self.concat_channels = concat_channels
                self.long_skip_merge_mode = long_skip_merge_mode
                self.conv = None
                if long_skip_merge_mode=='concat':
                    self.out_channels = in_channels+concat_channels
                elif long_skip_merge_mode=='sum':
                    self.out_channels = concat_channels
                    if in_channels != concat_channels:
                        self.conv = convolution(in_channels=in_channels,
                                                out_channels=concat_channels,
                                                kernel_size=1,
                                                ndim=ndim,
                                                init=init,
                                                padding=0)
                else:
                    raise ValueError("long_skip_merge_mode must be either "
                                     "`concat` or `sum`, not `{}`"
                                     "".format(long_skip_merge_mode))
                
            def forward(self, x, x_concat):
                # Adjust number of channels to match x_concat.
                if self.conv is not None:
                    x = self.conv(x)
                    
                # Spatially crop the tensors to the smallest dimensions 
                # between them. Center tensors before cropping.
                x, x_concat = crop_stack([x, x_concat])
                
                # Merge.
                merged = merge([x, x_concat], mode=self.long_skip_merge_mode)
                
                return merged
              
        '''
        Set up long skips.
        '''
        long_skip_list = []
        if long_skip:
            for i in range(len(self.blocks_instantiated)//2):
                concat_block = self.blocks_instantiated[i]
                block = self.blocks_instantiated[-i-2]
                if (isinstance(concat_block, identity_block) and
                    isinstance(block, identity_block)):
                    # These blocks do nothing. Don't make a long skip.
                    long_skip_list.append(None)
                skip = merge_long_skip(\
                                    in_channels=block.out_channels,
                                    concat_channels=concat_block.out_channels,
                                    long_skip_merge_mode=long_skip_merge_mode)
                long_skip_list.append(skip)
                self._modules['long_skip_{}'.format(i)] = skip
        self.long_skip_list = long_skip_list[::-1]   # reverse order
        
        '''
        Set up linear classifer.
        '''
        self.classifier = None
        if num_classes is not None:
            in_channels = self._modules['up_0'].out_channels
            self.classifier = convolution(in_channels=in_channels,
                                          out_channels=num_classes,
                                          kernel_size=1,
                                          ndim=ndim)
            self._modules['classifier'] = self.classifier
    
    def forward(self, input):
        '''
        Connect all the blocks on the contracting and expanding paths.
        '''
    
        # Book keeping
        tensors = {}
        depth = len(self.blocks)//2
    
        # Encoder (downsampling)
        x = input
        for b in range(0, depth):
            block = self.blocks_instantiated[b]
            x = tensors[b] = block(x)
        
        # Bottleneck
        block = self.blocks_instantiated[depth]
        x = block(x)
    
        # Decoder (upsampling)
        for b in range(0, depth):
            if self.long_skip and self.long_skip_list[b] is not None:
                x_concat = tensors[depth-b-1]
                x = self.long_skip_list[b](x, x_concat)
            block = self.blocks_instantiated[depth+b+1]
            x = block(x)
    
        # Output
        if self.classifier is not None:
            x = self.classifier(x)
            if self.num_classes==1:
                x = F.sigmoid(x)
            else:
                # Softmax that works on ND inputs.
                e = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
                s = torch.sum(e, dim=1, keepdim=True)
                x = e / s
        
        return x


def assemble_resunet(in_channels, num_classes, num_init_blocks,
                     num_main_blocks, main_block_depth, init_num_filters,
                     short_skip=True, long_skip=True,
                     long_skip_merge_mode='concat',
                     main_block=None, init_block=None, upsample_mode='repeat',
                     dropout=0., normalization=batch_normalization,
                     norm_kwargs=None, conv_padding=True,
                     init='kaiming_normal', nonlinearity='ReLU', ndim=2,
                     verbose=True):
    """
    in_channels : Number of channels in the input.
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
        convolutions. Also the number of filters in every init_block. Each
        main_block doubles (halves) the number of  filters for each decrease
        (increase) in resolution.
    short_skip : A boolean specifying whether to use ResNet-like shortcut
        connections from the input of each block to its output. The inputs are
        summed with the outputs.
    long_skip : A boolean specifying whether to use long skip connections
        from the downward path to the upward path. These can either concatenate
        or sum features across.
    long_skip_merge_mode : Either or 'sum', 'concat' features across skip.
    main_block : A layer defining the main_block (bottleneck by default).
    init_block : A layer defining the init_block (tiny_block by default).
    upsample_mode : Either 'repeat' or 'conv'. With 'repeat', rows and colums
        are repeated as in nearest neighbour interpolation. With 'conv',
        upscaling is done via transposed convolution.
    dropout : A float [0, 1] specifying the dropout probability, introduced in
        every block.
    normalization : The normalization to apply to layers (by default: batch
        normalization). If None, no normalization is applied.
    norm_kwargs : Keyword arguments to pass to batch norm layers. For batch
        normalization, default momentum is 0.9.
    conv_padding : Whether to use zero-padding for convolutions. If True, the
        output size is the same as the input size; if False, the output is
        smaller than the input size.
    init : A string specifying (or a function defining) the initializer for
        layers.
    nonlinearity : A string (or function defining) the nonlinearity.
    ndim : The spatial dimensionality of the input and output (either 2 or 3).
    verbose : A boolean specifying whether to print messages about model   
        structure during construction (if True).
    """
    
    '''
    By default, use depth 2 bottleneck for main_block
    '''
    if main_block is None:
        main_block = bottleneck
    if init_block is None:
        init_block = tiny_block
    
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
    If batch_normalization is used and norm_kwargs is not set, set default
    kwargs.
    '''
    if norm_kwargs is None:
        if normalization == batch_normalization:
            norm_kwargs = {'momentum': 0.1,
                           'affine': True}
        else:
            norm_kwargs = {}
            
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'skip': short_skip,
                    'dropout': dropout,
                    'normalization': normalization,
                    'norm_kwargs': norm_kwargs,
                    'upsample_mode': upsample_mode,
                    'nonlinearity': nonlinearity,
                    'conv_padding': conv_padding,
                    'init': init,
                    'ndim': ndim}
    
    '''
    Assemble all necessary blocks.
    '''
    blocks_down = []
    blocks_across = []
    blocks_up = []
    
    # The first block is just a convolution.
    # No normalization or nonlinearity on input.
    kwargs = {'num_filters': init_num_filters,
              'skip': False,
              'normalization': None,
              'nonlinearity': None,
              'dropout': dropout,
              'conv_padding': conv_padding,
              'init': init,
              'ndim': ndim}
    blocks_down.append((tiny_block, kwargs))
    
    # Down, init_block
    for i in range(num_init_blocks):
        kwargs = {'block_function': init_block,
                  'num_filters': init_num_filters,
                  'repetitions': 1}
        kwargs.update(block_kwargs)
        blocks_down.append((repeat_block, kwargs))
        
    # Down, main_block
    for i in range(num_main_blocks):
        kwargs = {'block_function': main_block,
                  'num_filters': init_num_filters*(2**i),
                  'repetitions': main_block_depth[i]}
        kwargs.update(block_kwargs)
        blocks_down.append((repeat_block, kwargs))
        
    # Bottleneck, main_block
    kwargs = {'block_function': main_block,
              'num_filters': init_num_filters*(2**num_main_blocks),
              'repetitions': main_block_depth[num_main_blocks]}
    kwargs.update(block_kwargs)
    blocks_across.append((repeat_block, kwargs))
    
    # Up, main_block
    for i in range(num_main_blocks-1, -1, -1):
        kwargs = {'block_function': main_block,
                  'num_filters': init_num_filters*(2**i),
                  'repetitions': main_block_depth[-i-1]}
        kwargs.update(block_kwargs)
        blocks_up.append((repeat_block, kwargs))
    
    # Up, init_block
    for i in range(num_init_blocks):
        kwargs = {'block_function': init_block,
                  'num_filters': init_num_filters,
                  'repetitions': 1}
        kwargs.update(block_kwargs)
        blocks_up.append((repeat_block, kwargs))
        
    # The last block is just a convolution.
    # As requested, normalization and nonlinearity applied on input.
    kwargs = {'num_filters': init_num_filters,
              'skip': False,
              'normalization': normalization,
              'nonlinearity': nonlinearity,
              'dropout': dropout,
              'conv_padding': conv_padding,
              'init': init,
              'ndim': ndim}
    blocks_up.append((tiny_block, kwargs))
        
    blocks = blocks_down + blocks_across + blocks_up
    
    '''
    Assemble model.
    '''
    model = fcn(in_channels=in_channels,
                num_classes=num_classes,
                blocks=blocks,
                long_skip=long_skip,
                long_skip_merge_mode=long_skip_merge_mode,
                conv_padding=conv_padding,
                ndim=ndim,
                verbose=verbose)
    return model


def assemble_unet(in_channels, num_classes, init_num_filters=64,
                  num_pooling=4, short_skip=False, long_skip=True,
                  long_skip_merge_mode='concat', upsample_mode='conv',
                  dropout=0., normalization=None, norm_kwargs=None,
                  conv_padding=True, init='kaiming_normal', 
                  nonlinearity='ReLU',  halve_features_on_upsample=True,
                  ndim=2, verbose=True):
    """
    in_channels : Number of channels in the input.
    num_classes : The number of classes in the segmentation output.
    init_num_filters : The number of filters used in the convolutions of the
        first and lost blocks in the network. With every downsampling, the
        number of filters is doubled; with every upsampling, it is halved.
        There are two convolutions in a unet_block so a a list/tuple of
        two values can be passed to set each convolution separately. For
        example, the original 2D UNet uses init_num_filters=64 or (64, 64)
        while the original 3D UNet uses init_num_filters=(32, 64).
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
        Recommended to pass batch_normalization when using short_skip==True.
    norm_kwargs : Keyword arguments to pass to normalization layers. If using
        batch_normalization, kwargs are autoset with a momentum of 0.9.
    conv_padding : Whether to use zero-padding for convolutions. If True, the
        output size is the same as the input size; if False, the output is
        smaller than the input size.
    init : A string specifying (or a function defining) the initializer for
        layers.
    nonlinearity : The nonlinearity to use, passed as a string or a function.
    halve_features_on_upsample : As in the original 2D UNet, have each block
        halve the number of feature maps when upsampling. This is not done in
        the original 3D UNet.
    ndim : The spatial dimensionality of the input and output (either 2 or 3).
    verbose : A boolean specifying whether to print messages about model   
        structure during construction (if True).
    """
    
    '''
    ndim must be only 2 or 3.
    '''
    if ndim not in [2, 3]:
        raise ValueError("ndim must be either 2 or 3")
            
    '''
    If batch_normalization is used and norm_kwargs is not set, set default
    kwargs.
    '''
    if norm_kwargs is None:
        if normalization == batch_normalization:
            norm_kwargs = {'momentum': 0.1,
                           'affine': True}
        else:
            norm_kwargs = {}
            
    '''
    init_num_filters could be a list
    '''
    if hasattr(init_num_filters, '__len__'):
        init_num_filters = np.array(init_num_filters)
        if len(init_num_filters) != 2:
            raise ValueError("init_num_filters must be an int "
                             "or a length 2 iterable")
            
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'skip': short_skip,
                    'normalization': normalization,
                    'norm_kwargs': norm_kwargs,
                    'nonlinearity': nonlinearity,
                    'upsample_mode': upsample_mode,
                    'conv_padding': conv_padding,
                    'init': init,
                    'ndim': ndim,
                    'halve_features_on_upsample': halve_features_on_upsample}
    
    '''
    Assemble all necessary blocks.
    '''
    blocks_down = []
    blocks_across = []
    blocks_up = []
    for i in range(0, num_pooling):
        kwargs = {'num_filters': init_num_filters*(2**i)}
        kwargs.update(block_kwargs)
        blocks_down.append((unet_block, kwargs))
    kwargs = {'num_filters': init_num_filters*(2**num_pooling),
              'dropout': dropout}
    kwargs.update(block_kwargs)
    blocks_across.append((unet_block, kwargs))
    for i in range(num_pooling-1, -1, -1):
        kwargs = {'num_filters': init_num_filters*(2**i)}
        if i==num_pooling-1:
            kwargs['dropout'] = dropout
        kwargs.update(block_kwargs)
        blocks_up.append((unet_block, kwargs))
    blocks = blocks_down + blocks_across + blocks_up
    
    '''
    Assemble model.
    '''
    model = fcn(in_channels=in_channels,
                num_classes=num_classes,
                blocks=blocks,
                long_skip=long_skip,
                long_skip_merge_mode=long_skip_merge_mode,
                conv_padding=conv_padding,
                ndim=ndim,
                verbose=verbose)
    return model


def assemble_vnet(in_channels, num_classes, init_num_filters=32,
                  num_pooling=4, short_skip=True, long_skip=True,
                  long_skip_merge_mode='concat', upsample_mode='conv',
                  dropout=0., normalization=None, norm_kwargs=None,
                  conv_padding=True, init='xavier_uniform',
                  nonlinearity='PReLU', ndim=3, verbose=True):
    """
    in_channels : Number of channels in the input.
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
    norm_kwargs : Keyword arguments to pass to normalization layers. If using
        batch_normalization, kwargs are autoset with a momentum of 0.9.
    conv_padding : Whether to use zero-padding for convolutions. If True, the
        output size is the same as the input size; if False, the output is
        smaller than the input size.
    init : A string specifying (or a function defining) the initializer for
        layers.
    nonlinearity : The nonlinearity to use, passed as a string or a function.
    ndim : The spatial dimensionality of the input and output (either 2 or 3).
    verbose : A boolean specifying whether to print messages about model   
        structure during construction (if True).
    """
    
    '''
    ndim must be only 2 or 3.
    '''
    if ndim not in [2, 3]:
        raise ValueError("ndim must be either 2 or 3")
            
    '''
    If batch_normalization is used and norm_kwargs is not set, set default
    kwargs.
    '''
    if norm_kwargs is None:
        if normalization == batch_normalization:
            norm_kwargs = {'momentum': 0.1,
                           'affine': True}
        else:
            norm_kwargs = {}
            
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'skip': short_skip,
                    'normalization': normalization,
                    'norm_kwargs': norm_kwargs,
                    'conv_padding': conv_padding,
                    'init': init,
                    'nonlinearity': nonlinearity,
                    'upsample_mode': upsample_mode,
                    'dropout': dropout,
                    'ndim': ndim}
    
    '''
    Assemble all necessary blocks.
    '''
    blocks_down = []
    blocks_across = []
    blocks_up = []
    for i in range(0, num_pooling):
        kwargs = {'num_filters': init_num_filters*(2**i)}
        if i==0:
            kwargs['num_conv'] = 1
            kwargs['normalization'] = None
            kwargs['nonlinearity'] = None
        elif i==1:
            kwargs['num_conv'] = 2
        else:
            kwargs['num_conv'] = 3
        kwargs.update(block_kwargs)
        blocks_down.append((vnet_block, kwargs))
    kwargs = {'num_filters': init_num_filters*(2**num_pooling),
              'num_conv': 3}
    kwargs.update(block_kwargs)
    blocks_across.append((vnet_block, kwargs))
    for i in range(num_pooling-1, -1, -1):
        kwargs = {'num_filters': init_num_filters*(2**i)}
        if i==0:
            kwargs['num_conv'] = 1
        elif i==1:
            kwargs['num_conv'] = 2
        else:
            kwargs['num_conv'] = 3
        kwargs.update(block_kwargs)
        blocks_up.append((vnet_block, kwargs))
    blocks = blocks_down + blocks_across + blocks_up
    
    '''
    Assemble model.
    '''
    model = fcn(in_channels=in_channels,
                num_classes=num_classes,
                blocks=blocks,
                long_skip=long_skip,
                long_skip_merge_mode=long_skip_merge_mode,
                conv_padding=conv_padding,
                ndim=ndim,
                verbose=verbose)
    return model


def assemble_fcdensenet(in_channels, num_classes,
                        block_depth=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4], 
                        num_blocks=11, init_num_filters=48, growth_rate=16,
                        long_skip=True, skip_merge_mode='concat', 
                        upsample_mode='conv', dropout=0.2,
                        normalization=batch_normalization, norm_kwargs=None,
                        conv_padding=True, init='kaiming_uniform',
                        nonlinearity='ReLU', ndim=2, verbose=True):
    """
    in_channels : Number of channels in the input.
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
        every convolution layer except the first and last ones.
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
    norm_kwargs : Keyword arguments to pass to normalization layers. If using
        batch_normalization, kwargs are autoset with a momentum of 0.9.
    conv_padding : Whether to use zero-padding for convolutions. If True, the
        output size is the same as the input size; if False, the output is
        smaller than the input size.
    init : A string specifying (or a function defining) the initializer for
        layers.
    nonlinearity : The nonlinearity to use, passed as a string or a function.
    ndim : The spatial dimensionality of the input and output (either 2 or 3).
    verbose : A boolean specifying whether to print messages about model   
        structure during construction (if True).
    """
        
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
    If batch_normalization is used and norm_kwargs is not set, set default
    kwargs.
    '''
    if norm_kwargs is None:
        if normalization == batch_normalization:
            norm_kwargs = {'momentum': 0.1,
                           'affine': True}
        else:
            norm_kwargs = {}
            
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'dropout': dropout,
                    'normalization': normalization,
                    'norm_kwargs': norm_kwargs,
                    'upsample_mode': upsample_mode,
                    'skip_merge_mode': skip_merge_mode,
                    'nonlinearity': nonlinearity,
                    'conv_padding': conv_padding,
                    'init': init,
                    'ndim': ndim}
    if growth_rate is not None:
        block_kwargs['num_filters'] = growth_rate
    
    '''
    Assemble all necessary blocks.
    '''
    blocks_down = []
    blocks_across = []
    blocks_up = []
    n_blocks_side = num_blocks//2   # on one side
    
    
    # The first block is a convolution followed by a dense block.
    class first_block(torch.nn.Module):
        def __init__(self, in_channels, subsample=False, upsample=False):
            super(first_block, self).__init__()
            self.in_channels = in_channels
            self.subsample = subsample
            self.upsample = upsample
            self.conv = convolution(in_channels=in_channels,
                                    out_channels=init_num_filters,
                                    kernel_size=3,
                                    ndim=ndim,
                                    init=init,
                                    padding=int(conv_padding))
            kwargs = {'in_channels': init_num_filters,
                      'block_depth': block_depth[0],
                      'merge_input': True}
            kwargs.update(block_kwargs)
            if growth_rate is None:
                kwargs['filters'] = init_num_filters
            self.block = dense_block(**kwargs)
            self.out_channels = self.block.out_channels
            
        def forward(self, input):
            out = self.conv(input)
            out = self.block(out)
            return out
    blocks_down.append((first_block, {}))
    
    # Down (Encoder)
    for i in range(1, n_blocks_side):
        kwargs = {'block_depth': block_depth[i],
                  'merge_input': True}
        if growth_rate is None:
            kwargs['num_filters'] = init_num_filters*(2**i)
        kwargs.update(block_kwargs)
        blocks_down.append((dense_block, kwargs))
        
    # NOTE: merge_input should be False for bottleneck and decoder.
    
    # Bottleneck, main_block
    kwargs = {'block_depth': block_depth[n_blocks_side],
              'merge_input': False}
    if growth_rate is None:
        kwargs['num_filters'] = init_num_filters*(2**n_blocks_side)
    kwargs.update(block_kwargs)
    blocks_across.append((dense_block, kwargs))
    blocks = blocks_down + blocks_across
    
    # Up (Decoder)
    for i in range(n_blocks_side-1, 0, -1):
        kwargs = {'block_depth': block_depth[-i-1],
                  'merge_input': False}
        if growth_rate is None:
            kwargs['num_filters'] = init_num_filters*(2**i)
        kwargs.update(block_kwargs)
        blocks_up.append((dense_block, kwargs))
        
    # The last block is just a convolution.
    # As requested, normalization and nonlinearity applied on input.
    kwargs = {'num_filters': init_num_filters,
              'skip': False,
              'normalization': normalization,
              'nonlinearity': nonlinearity,
              'dropout': dropout,
              'conv_padding': conv_padding,
              'init': init,
              'ndim': ndim}
    blocks_up.append((tiny_block, kwargs))
        
    blocks = blocks_down + blocks_across + blocks_up
    
    '''
    Assemble model.
    '''
    model = fcn(in_channels=in_channels,
                num_classes=num_classes,
                blocks=blocks,
                long_skip=long_skip,
                long_skip_merge_mode=skip_merge_mode,
                conv_padding=conv_padding,
                ndim=ndim,
                verbose=verbose)
    return model
