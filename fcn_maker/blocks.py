from __future__ import (print_function,
                        division)
import torch
from torch.functional import F


"""
Return a nonlinearity from the core library or return the provided function.
"""
def get_nonlinearity(nonlin):
    if nonlin is None:
        # Return identity module (should be a module).
        class identity_activation(torch.nn.Module):
            def __init__(self):
                super(identity_activation, self).__init__()
            def forward(self, input):
                return input
        module = identity_activation
    elif isinstance(nonlin, str):
        # Find the nonlinearity by name.
        try:
            module = getattr(torch.nn.modules.activation, nonlin)
        except AttributeError:
            raise ValueError("Specified nonlinearity ({}) not found."
                             "".format(nonlin))
    else:
        # Not a name and no length; assume a module is passed.
        module = nonlin
    return module()


"""
Return an initializer from the core library or return the provided function.
"""
def get_initializer(init):
    if init is None:
        func = lambda x:x
    elif isinstance(init, str):
        # Find the initializer by name.
        try:
            func = getattr(torch.nn.init, init)
        except AttributeError:
            raise ValueError("Specified initializer ({}) not found."
                             "".format(init))
    else:
        # Not a name; assume a function is passed instead.
        func = init
    return func

"""
Select 2D or 3D as argument (ndim) and initialize weights on creation.
"""
class convolution(torch.nn.Module):
    def __init__(self, ndim=2, init=None, padding=None,
                 padding_mode='constant', *args, **kwargs):
        super(convolution, self).__init__()
        if ndim==2:
            conv = torch.nn.Conv2d
        elif ndim==3:
            conv = torch.nn.Conv3d
        else:
            ValueError("ndim must be 2 or 3")
        self.ndim = ndim
        self.init = init
        self.padding = padding
        self.padding_mode = padding_mode
        self.op = conv(*args, **kwargs)
        self.in_channels = self.op.in_channels
        self.out_channels = self.op.out_channels
        if init is not None:
            self.op.weight.data = get_initializer(init)(self.op.weight.data)

    def forward(self, input):
        out = input
        if self.padding is not None:
            padding = self.padding
            if not hasattr(padding, '__len__'):
                padding = [self.padding]*self.ndim*2
            out = F.pad(out, pad=padding, mode=self.padding_mode, value=0)
        out = self.op(out)
        return out

class convolution_transpose(torch.nn.Module):
    def __init__(self, ndim=2, init=None, *args, **kwargs):
        super(convolution_transpose, self).__init__()
        if ndim==2:
            conv = torch.nn.ConvTranspose2d
        elif ndim==3:
            conv = torch.nn.ConvTranspose3d
        else:
            ValueError("ndim must be 2 or 3")
        self.ndim = ndim
        self.init = init
        self.op = conv(*args, **kwargs)
        self.in_channels = self.op.in_channels
        self.out_channels = self.op.out_channels
        if init is not None:
            self.op.weight.data = get_initializer(init)(self.op.weight.data)

    def forward(self, input):
        return self.op(input)

def max_pooling(ndim=2, *args, **kwargs):
    if ndim==2:
        return torch.nn.MaxPool2d(*args, **kwargs)
    elif ndim==3:
        return torch.nn.MaxPool3d(*args, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")

def batch_normalization(ndim=2, *args, **kwargs):
    if ndim==1:
        return torch.nn.BatchNorm1d(*args, **kwargs)
    if ndim==2:
        return torch.nn.BatchNorm2d(*args, **kwargs)
    elif ndim==3:
        return torch.nn.BatchNorm3d(*args, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")

def instance_normalization(ndim=2, *args, **kwargs):
    if ndim==1:
        return torch.nn.InstanceNorm1d(*args, **kwargs)
    if ndim==2:
        return torch.nn.InstanceNorm2d(*args, **kwargs)
    elif ndim==3:
        return torch.nn.InstanceNorm3d(*args, **kwargs)
    else:
        raise ValueError("ndim must be 1, 2, or 3")


"""
Helper to perform tensor merging.
"""
def merge(tensors, mode):
    if mode not in ['sum', 'concat']:
        raise ValueError("Unrecognized merge mode: {}".format(mode))
    out = None
    if mode=='sum':
        out = tensors[0]
        for t in tensors[1:]:
            out = out + t
    elif mode=='concat':
        out = torch.cat(tensors, dim=1)
    return out


"""
Helper to center all tensors and spatially crop to the smallest dimensions.
"""
def crop_stack(tensors):
    ndim = tensors[0].ndimension()
    for t in tensors:
        if not t.ndimension()==ndim:
            raise ValueError("All tensors passed to crop_stack must have the "
                             "same number of dimensions.")

    # Find smallest length for each dimension.
    spatial_dims = range(2, ndim)
    min_lengths = {}
    for dim in spatial_dims:
        for t in tensors:
            if dim not in min_lengths or t.size()[dim] < min_lengths[dim]:
                min_lengths[dim] = t.size()[dim]

    # Center and crop.
    out_tensors = []
    for t in tensors:
        indices = [slice(None, None)]*ndim
        for dim in spatial_dims:
            if t.size()[dim] > min_lengths[dim]:
                offset = (t.size()[dim]-min_lengths[dim])//2
                indices[dim] = slice(offset, min_lengths[dim]+offset)

        # tensor must be indexed with tuple of slices, not list
        out_tensors.append(t[tuple(indices)])

    return out_tensors


"""
Helper to adjust the spatial size of a tensor, centering it and zero-padding 
and cropping, as necessary.
"""
def adjust_to_size(tensor, size, padding_mode='constant'):
    ndim = len(size)
    if tensor.ndimension()-2!=ndim:
        raise ValueError("`tensor` has {} spatial dimensions while `size` "
                         "has {}.".format(tensor.ndimension()-2, ndim))
    
    # Determine size difference.
    t_size = tensor.size()[2:]
    diff = [size[i]-t_size[i] for i in range(ndim)]
    
    # Determine cropping indices
    indices_crop = [slice(None, None)]*(ndim+2)
    for dim, d in enumerate(diff):
        if d<0:
            indices_crop[dim+2] = slice(-d//2, t_size[dim]-(-d//2+d%2))
    indices_crop = tuple(indices_crop)
        
    # Zero-pad and crop.
    if sum([d>0 for d in diff]):    # Needs zero-padding
        out_tensor = tensor
        pad = []
        for dim, d in enumerate(diff):
            pad.extend([int(d//2), int(d//2+d%2)])
        out_tensor = F.pad(out_tensor, pad=pad, mode=padding_mode, value=0)
    elif sum([d<0 for d in diff]):    # Needs cropping
        out_tensor = tensor[indices_crop]
    else:
        out_tensor = tensor
        
    return out_tensor
    
    
"""
Return AlphaDropout if nonlinearity is 'SELU', else Dropout.
"""
def get_dropout(dropout, nonlin=None):
    if nonlin=='SELU':
        return torch.nn.AlphaDropout(dropout)
    return torch.nn.Dropout(dropout)


"""
Helper function to subsample. Simple 2x decimation.
"""
class do_subsample(torch.nn.Module):
    def __init__(self, ndim):
        super(do_subsample, self).__init__()
        if ndim not in [2, 3]:
            raise ValueError('ndim must be 2 or 3')
        self.ndim = ndim

    def forward(self, input):
        out = None
        if self.ndim==2:
            out = input[:,:,::2,::2]
        elif self.ndim==3:
            out = input[:,:,::2,::2,::2]
        return out


"""
Helper function to execute some upsampling mode.

conv_kwargs are: in_channels, out_channels, kernel_size
"""
class do_upsample(torch.nn.Module):
    def __init__(self, mode, ndim, init=None, **conv_kwargs):
        super(do_upsample, self).__init__()
        if mode=='repeat':
            self.op = torch.nn.Upsample(scale_factor=2)
        elif mode=='conv':
            self.op = convolution_transpose(ndim=ndim,
                                            stride=2,
                                            init=init,
                                            **conv_kwargs)
            self.in_channels = self.op.in_channels
            self.out_channels = self.op.out_channels
        else:
            raise ValueError("Unrecognized upsample_mode: {}"
                             "".format(upsample_mode))
        self.mode = mode
        self.ndim = ndim
        self.init = init

    def forward(self, input):
        return self.op(input)


"""
Helper to build a norm -> ReLU -> conv block
This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
"""
class norm_nlin_conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 subsample=False, upsample=False, upsample_mode='repeat',
                 nonlinearity='ReLU', normalization=batch_normalization,
                 norm_kwargs=None, conv_padding=True, padding_mode='constant',
                 init='kaiming_normal_', ndim=2):
        super(norm_nlin_conv, self).__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subsample = subsample
        self.upsample = upsample
        self.upsample_mode = upsample_mode
        self.nonlinearity = nonlinearity
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.init = init
        self.ndim = ndim
        if normalization is not None:
            self._modules['norm'] = normalization(ndim=ndim,
                                                  num_features=in_channels,
                                                  **norm_kwargs)
        self._modules['nlin'] = get_nonlinearity(nonlinearity)
        if upsample:
            self._modules['upsample'] = do_upsample(mode=upsample_mode,
                                                    ndim=ndim,
                                                    in_channels=in_channels,
                                                    out_channels=in_channels,
                                                    kernel_size=2)
        stride = 1
        if subsample:
            stride = 2
        padding = kernel_size//2 if conv_padding else 0
        self._modules['conv'] = convolution(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            ndim=ndim,
                                            stride=stride,
                                            init=init,
                                            padding=padding,
                                            padding_mode=padding_mode)

    def forward(self, input):
        out = input
        for op in self._modules.values():
            out = op(out)
        return out


"""
Adds a shortcut between input and residual block and merges them with 'sum'.
"""
class shortcut(torch.nn.Module):
    def __init__(self, in_channels, out_channels, subsample, upsample,
                 upsample_mode='repeat', init='kaiming_normal_', ndim=2):
        super(shortcut, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.subsample = subsample
        self.upsample = upsample
        self.upsample_mode = upsample_mode
        self.init = init
        self.ndim = ndim

        # Downsample input
        if subsample:
            self._modules['subsample'] = do_subsample(ndim=ndim)

        # Upsample input
        if upsample:
            self._modules['upsample'] = do_upsample(mode=upsample_mode,
                                                    ndim=ndim,
                                                    in_channels=in_channels,
                                                    out_channels=in_channels,
                                                    kernel_size=2,
                                                    init=init)

        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        if in_channels != out_channels:
            self._modules['conv'] = convolution(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=1,
                                                ndim=ndim,
                                                init=init)

    def forward(self, input, residual):
        shortcut = input
        for op in self._modules.values():
            shortcut = op(shortcut)
        shortcut, residual = crop_stack([shortcut, residual])
        return residual+shortcut


"""
Defaults block class - defines arguments that all blocks must have.

NOTE: Blocks are expected to have an out_channels attribute but not an
      out_channels argument. This attribute must be computed in the block.
"""
class block_abstract(torch.nn.Module):
    def __init__(self, in_channels, num_filters, subsample, upsample):
        super(block_abstract, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.subsample = subsample
        self.upsample = upsample

    def forward(self, input):
        raise NotImplemented()

    def _register_modules(self, modules):
        if isinstance(modules, dict):
            self._modules.update(modules)
        else:
            if not hasattr(modules, '__len__'):
                modules = [modules]
            i = 0
            for m in modules:
                while 'layer_{}'.format(i) in self._modules:
                    i += 1
                self._modules['layer_{}'.format(i)] = m

    def get_out_channels(self):
        if not hasattr(self, 'out_channels'):
            raise NotImplemented("Blocks are expected to have an "
                                 "out_channels attribute but not an "
                                 "out_channels argument. This attribute must "
                                 "be computed in the block.")
        return self.out_channels


"""
Identity block - do nothing except handle subsampling + upsampling.
"""
class identity_block(block_abstract):
    def __init__(self, in_channels, num_filters, subsample=False,
                 upsample=False, upsample_mode='repeat', ndim=2,
                 init='kaiming_normal_'):
        super(identity_block, self).__init__(in_channels, num_filters,
                                             subsample, upsample)
        self.out_channels = num_filters
        self.upsample_mode = upsample_mode
        self.ndim = ndim
        self.init = init
        self.op = []
        if subsample:
            self.op += [do_subsample(ndim=ndim)]
        if upsample:
            self.op += [do_upsample(mode=upsample_mode,
                                    ndim=ndim,
                                    in_channels=in_channels,
                                    out_channels=num_filters,
                                    kernel_size=2,
                                    init=init)]
                                 
        self._register_modules(self.op)

    def forward(self, input):
        out = input
        for op in self.op:
            out = op(out)
        return out


"""
Bottleneck architecture for > 34 layer resnet.
Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
Returns a representation with num_filters*4 channels.
"""
class bottleneck(block_abstract):
    def __init__(self, in_channels, num_filters, subsample=False,
                 upsample=False, upsample_mode='repeat', skip=True,
                 dropout=0., normalization=batch_normalization,
                 norm_kwargs=None, conv_padding=True, padding_mode='constant',
                 kernel_size=3, init='kaiming_normal_', nonlinearity='ReLU',
                 ndim=2):
        super(bottleneck, self).__init__(in_channels, num_filters,
                                         subsample, upsample)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.out_channels = num_filters*4
        self.upsample_mode = upsample_mode
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []
        self.op += [norm_nlin_conv(in_channels=in_channels,
                                   out_channels=num_filters,
                                   kernel_size=1,
                                   subsample=subsample,
                                   normalization=normalization,
                                   norm_kwargs=norm_kwargs,
                                   conv_padding=conv_padding,
                                   padding_mode=padding_mode,
                                   init=init,
                                   nonlinearity=nonlinearity,
                                   ndim=ndim)]
        self.op += [norm_nlin_conv(in_channels=num_filters,
                                   out_channels=num_filters,
                                   kernel_size=kernel_size,
                                   normalization=normalization,
                                   norm_kwargs=norm_kwargs,
                                   conv_padding=conv_padding,
                                   padding_mode=padding_mode,
                                   init=init,
                                   nonlinearity=nonlinearity,
                                   ndim=ndim)]
        self.op += [norm_nlin_conv(in_channels=num_filters,
                                   out_channels=num_filters*4,
                                   kernel_size=1,
                                   upsample=upsample,
                                   upsample_mode=upsample_mode,
                                   normalization=normalization,
                                   norm_kwargs=norm_kwargs,
                                   conv_padding=conv_padding,
                                   padding_mode=padding_mode,
                                   init=init,
                                   nonlinearity=nonlinearity,
                                   ndim=ndim)]
        if dropout > 0:
            self.op += [get_dropout(dropout, nonlinearity)]
        self._register_modules(self.op)
        self.op_shortcut = None
        if skip:
            self.op_shortcut = shortcut(in_channels=in_channels,
                                        out_channels=num_filters*4,
                                        subsample=subsample,
                                        upsample=upsample,
                                        upsample_mode=upsample_mode,
                                        init=init,
                                        ndim=ndim)
            self._register_modules({'shortcut': self.op_shortcut})

    def forward(self, input):
        out = input
        for op in self.op:
            out = op(out)
        if self.skip:
            out = self.op_shortcut(input, out)
        return out


"""
Basic 3 X 3 convolution blocks.
Use for resnet with layers <= 34
Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
"""
class basic_block(block_abstract):
    def __init__(self, in_channels, num_filters, subsample=False,
                 upsample=False, upsample_mode='repeat', skip=True, dropout=0.,
                 normalization=batch_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', nonlinearity='ReLU', ndim=2):
        super(basic_block, self).__init__(in_channels, num_filters,
                                          subsample, upsample)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.out_channels = num_filters
        self.upsample_mode = upsample_mode
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []
        self.op += [norm_nlin_conv(in_channels=in_channels,
                                   out_channels=num_filters,
                                   kernel_size=kernel_size,
                                   subsample=subsample,
                                   normalization=normalization,
                                   norm_kwargs=norm_kwargs,
                                   conv_padding=conv_padding,
                                   padding_mode=padding_mode,
                                   init=init,
                                   nonlinearity=nonlinearity,
                                   ndim=ndim)]
        if dropout > 0:
            self.op += [get_dropout(dropout, nonlin=nonlinearity)]
        self.op += [norm_nlin_conv(in_channels=num_filters,
                                   out_channels=num_filters,
                                   kernel_size=kernel_size,
                                   upsample=upsample,
                                   upsample_mode=upsample_mode,
                                   normalization=normalization,
                                   norm_kwargs=norm_kwargs,
                                   conv_padding=conv_padding,
                                   padding_mode=padding_mode,
                                   init=init,
                                   nonlinearity=nonlinearity,
                                   ndim=ndim)]
        self._register_modules(self.op)
        self.op_shortcut = None
        if skip:
            self.op_shortcut = shortcut(in_channels=in_channels,
                                        out_channels=num_filters,
                                        subsample=subsample,
                                        upsample=upsample,
                                        upsample_mode=upsample_mode,
                                        init=init,
                                        ndim=ndim)
            self._register_modules({'shortcut': self.op_shortcut})

    def forward(self, input):
        out = input
        for op in self.op:
            out = op(out)
        if self.skip:
            out = self.op_shortcut(input, out)
        return out


"""
A single basic 3x3 convolution.
"""
class tiny_block(block_abstract):
    def __init__(self, in_channels, num_filters, subsample=False,
                 upsample=False, upsample_mode='repeat', skip=True, dropout=0.,
                 normalization=batch_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', nonlinearity='ReLU', ndim=2):
        super(tiny_block, self).__init__(in_channels, num_filters,
                                             subsample, upsample)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.out_channels = num_filters
        self.upsample_mode = upsample_mode
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []
        if normalization is not None:
            self.op += [normalization(ndim=ndim,
                                      num_features=in_channels,
                                      **norm_kwargs)]
        self.op += [get_nonlinearity(nonlinearity)]
        if subsample:
            self.op += [max_pooling(kernel_size=2, ndim=ndim)]
        padding = kernel_size//2 if conv_padding else 0
        self.op += [convolution(in_channels=in_channels,
                                out_channels=num_filters,
                                kernel_size=kernel_size,
                                ndim=ndim,
                                init=init,
                                padding=padding,
                                padding_mode=padding_mode)]
        if dropout > 0:
            self.op += [get_dropout(dropout, nonlinearity)]
        if upsample:
            self.op += [do_upsample(mode=upsample_mode,
                                    ndim=ndim,
                                    in_channels=num_filters,
                                    out_channels=num_filters,
                                    kernel_size=2,
                                    init=init)]
        self._register_modules(self.op)
        self.op_shortcut = None
        if skip:
            self.op_shortcut = shortcut(in_channels=in_channels,
                                        out_channels=num_filters,
                                        subsample=subsample,
                                        upsample=upsample,
                                        upsample_mode=upsample_mode,
                                        init=init,
                                        ndim=ndim)
            self._register_modules({'shortcut': self.op_shortcut})

    def forward(self, input):
        out = input
        for op in self.op:
            out = op(out)
        if self.skip:
            out = self.op_shortcut(input, out)
        return out

"""
Builds a block with repeating sub-blocks.
"""
class repeat_block(block_abstract):
    def __init__(self, block_function, in_channels, num_filters, repetitions,
                 skip=True, dropout=0., subsample=False, upsample=False,
                 upsample_mode='repeat', normalization=batch_normalization,
                 norm_kwargs=None, conv_padding=True, padding_mode='constant',
                 kernel_size=3, init='kaiming_normal_', nonlinearity='ReLU',
                 ndim=2):
        super(repeat_block, self).__init__(in_channels, num_filters,
                                           subsample, upsample)
        if repetitions<=0:
            raise ValueError("block repetitions (block depth) must be greater "
                            "than zero")
        if norm_kwargs is None:
            norm_kwargs = {}
        self.block_function = block_function
        self.repetitions = repetitions
        self.skip = skip
        self.dropout = dropout
        self.upsample_mode = upsample_mode
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.blocks = []
        last_out_channels = None
        for i in range(repetitions):
            subsample_i = subsample if i==0 else False
            upsample_i = upsample if i==repetitions-1 else False
            in_channels_i = in_channels if i==0 else last_out_channels
            block = block_function(in_channels=in_channels_i,
                                   num_filters=num_filters,
                                   skip=skip,
                                   dropout=dropout,
                                   subsample=subsample_i,
                                   upsample=upsample_i,
                                   upsample_mode=upsample_mode,
                                   normalization=normalization,
                                   norm_kwargs=norm_kwargs,
                                   nonlinearity=nonlinearity,
                                   conv_padding=conv_padding,
                                   padding_mode=padding_mode,
                                   kernel_size=kernel_size,
                                   init=init,
                                   ndim=ndim)
            last_out_channels = block.get_out_channels()
            self.blocks.append(block)
        self.out_channels = block.out_channels
        self._register_modules(self.blocks)

    def forward(self, input):
        out = input
        for op in self.blocks:
            out = op(out)
        return out


"""
Two basic 3x3 convolutions with 2x2 conv upsampling, as in the UNet.
Subsampling, upsampling, and dropout handled as in the UNet.
"""
class unet_block(block_abstract):
    def __init__(self, in_channels, num_filters, subsample=False,
                 upsample=False, upsample_mode='conv',
                 halve_features_on_upsample=True, skip=False, dropout=0.,
                 normalization=None, norm_kwargs=None, conv_padding=True,
                 padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', nonlinearity='ReLU', ndim=2):
        super(unet_block, self).__init__(in_channels, num_filters,
                                         subsample, upsample)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.upsample_mode = upsample_mode
        self.halve_features_on_upsample = halve_features_on_upsample
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []

        # Filters can be an int or a tuple/list
        if hasattr(num_filters, '__len__'):
            num_filters_1, num_filters_2 = num_filters
        else:
            num_filters_1 = num_filters_2 = num_filters

        if subsample:
            self.op += [max_pooling(kernel_size=2, ndim=ndim)]
        padding = kernel_size//2 if conv_padding else 0
        self.op += [convolution(in_channels=in_channels,
                                out_channels=num_filters_1,
                                kernel_size=kernel_size,
                                ndim=ndim,
                                init=init,
                                padding=padding,
                                padding_mode=padding_mode)]
        self.op += [norm_nlin_conv(in_channels=num_filters_1,
                                   out_channels=num_filters_2,
                                   kernel_size=kernel_size,
                                   normalization=normalization,
                                   norm_kwargs=norm_kwargs,
                                   conv_padding=conv_padding,
                                   padding_mode=padding_mode,
                                   init=init,
                                   nonlinearity=nonlinearity,
                                   ndim=ndim)]
        if normalization is not None:
            self.op += [normalization(ndim=ndim,
                                      num_features=num_filters_2,
                                      **norm_kwargs)]
        self.op += [get_nonlinearity(nonlinearity)]
        if dropout > 0:
            self.op += [get_dropout(dropout, nonlinearity)]
        out_channels = num_filters_2
        if upsample:
            # "up-convolution" in standard 2D unet halves the number of
            # feature maps - but not in the standard 3D unet. It's just a
            # user-settable option in this block, regardless of ndim.
            if halve_features_on_upsample:
                out_channels_up = num_filters_2//2
                if upsample_mode=='repeat':
                    self.op += [convolution(in_channels=num_filters_2,
                                            out_channels=out_channels_up,
                                            kernel_size=1,
                                            ndim=ndim,
                                            init=init)]
            else:
                out_channels_up = num_filters_2
            self.op += [do_upsample(mode=upsample_mode,
                                    ndim=ndim,
                                    kernel_size=2,
                                    in_channels=num_filters_2,
                                    out_channels=out_channels_up,
                                    init=init)]

            self.op += [get_nonlinearity(nonlinearity)]
            out_channels = out_channels_up
        self.out_channels = out_channels
        self._register_modules(self.op)
        self.op_shortcut = None
        if skip:
            self.op_shortcut = shortcut(in_channels=in_channels,
                                        out_channels=out_channels,
                                        subsample=subsample,
                                        upsample=upsample,
                                        upsample_mode=upsample_mode,
                                        init=init,
                                        ndim=ndim)
            self._register_modules({'shortcut': self.op_shortcut})

    def forward(self, input):
        out = input
        for op in self.op:
            out = op(out)
        if self.skip:
            out = self.op_shortcut(input, out)
        return out


"""
Processing block as in the VNet.
"""
class vnet_block(block_abstract):
    def __init__(self, in_channels, num_filters, num_conv=3, subsample=False,
                 upsample=False, upsample_mode='conv', skip=True, dropout=0.,
                 normalization=None, norm_kwargs=None, conv_padding=True,
                 padding_mode='constant', kernel_size=5, init='xavier_uniform',
                 nonlinearity='ReLU', ndim=3):
        super(vnet_block, self).__init__(in_channels, num_filters,
                                         subsample, upsample)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.out_channels = num_filters
        self.num_conv = num_conv
        self.upsample_mode = upsample_mode
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []
        if subsample:
            self.op += [convolution(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=2,
                                    stride=2,
                                    ndim=ndim,
                                    init=init,
                                    padding=0)]
        for i in range(num_conv):
            in_channels_i = in_channels if i==0 else num_filters
            self.op += [norm_nlin_conv(in_channels=in_channels_i,
                                       out_channels=num_filters,
                                       kernel_size=kernel_size,
                                       normalization=normalization,
                                       norm_kwargs=norm_kwargs,
                                       conv_padding=conv_padding,
                                       padding_mode=padding_mode,
                                       init=init,
                                       nonlinearity=nonlinearity,
                                       ndim=ndim)]

            if dropout > 0:
                self.op += [get_dropout(dropout, nonlinearity)]
        self._register_modules(self.op)
        self.op_shortcut = None
        if skip:
            self.op_shortcut = shortcut(in_channels=in_channels,
                                        out_channels=num_filters,
                                        subsample=subsample,
                                        upsample=False,
                                        upsample_mode=upsample_mode,
                                        init=init,
                                        ndim=ndim)
            self._register_modules({'shortcut': self.op_shortcut})
        self.op_upsample = []
        out_channels = num_filters
        if upsample:
            # "up-convolution" also halves the number of feature maps.
            if normalization is not None:
                self.op_upsample += [normalization(ndim=ndim,
                                                   num_features=num_filters,
                                                   **norm_kwargs)]
            self.op_upsample += [get_nonlinearity(nonlinearity)]
            self.op_upsample += [do_upsample(mode=upsample_mode,
                                             ndim=ndim,
                                             in_channels=num_filters,
                                             out_channels=num_filters//2,
                                             kernel_size=2,
                                             init=init)]
            self.op_upsample += [get_nonlinearity(nonlinearity)]
            out_channels = num_filters//2
            self._register_modules(self.op_upsample)
        self.out_channels = out_channels

    def forward(self, input):
        out = input
        for op in self.op:
            out = op(out)
        if self.skip:
            out = self.op_shortcut(input, out)
        for op in self.op_upsample:
            out = op(out)
        return out


"""
Dense block (as in a DenseNet), as implemented in the 100 layer Tiramisu.

NOTE: Unlike in other blocks, out_channels is set automatically, depending on
      the following arguments: num_filters, skip_merge_mode, merge_input.

paper : https://arxiv.org/abs/1611.09326 (version 2)
code  : https://github.com/SimJeg/FC-DenseNet
        commit ee933144949d82ada32198e49d76b708f60e4
"""
class dense_block(block_abstract):
    def __init__(self, in_channels, num_filters, block_depth=4,
                 subsample=False, upsample=False, upsample_mode='conv',
                 skip_merge_mode='concat', merge_input=True, dropout=0.,
                 normalization=batch_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 init='kaiming_uniform', nonlinearity='ReLU', ndim=2):
        super(dense_block, self).__init__(in_channels, num_filters,
                                          subsample, upsample)
        if norm_kwargs is None:
            norm_kwargs = {}
        out_channels = num_filters*block_depth + in_channels*merge_input
        if skip_merge_mode=='sum':
            out_channels = num_filters
        self.out_channels = out_channels
        self.block_depth = block_depth
        self.updample_mode = upsample_mode
        self.skip_merge_mode = skip_merge_mode
        self.merge_input = merge_input
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []

        # Transition down (preserve num out_channels)
        if subsample:
            self.op += [norm_nlin_conv(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=1,
                                       normalization=normalization,
                                       norm_kwargs=norm_kwargs,
                                       init=init,
                                       nonlinearity=nonlinearity,
                                       ndim=ndim)]
            if dropout > 0:
                self.op += [get_dropout(dropout, nonlinearity)]
            self.op += [max_pooling(kernel_size=2, ndim=ndim)]

        # If 'sum' mode, make the channel dimension match.
        if skip_merge_mode=='sum':
            if in_channels != num_filters:
                self.op += [convolution(in_channels=in_channels,
                                        out_channels=num_filters,
                                        kernel_size=1,
                                        ndim=ndim,
                                        init=init,
                                        padding=0)]

        # Build the dense block.
        self.op_dense = []
        for i in range(block_depth):
            op = []
            in_channels_i = in_channels + num_filters*i
            if skip_merge_mode=='sum':
                in_channels_i = num_filters
            op += [norm_nlin_conv(in_channels=in_channels_i,
                                  out_channels=num_filters,
                                  kernel_size=kernel_size,
                                  normalization=normalization,
                                  norm_kwargs=norm_kwargs,
                                  conv_padding=conv_padding,
                                  padding_mode=padding_mode,
                                  init=init,
                                  nonlinearity=nonlinearity,
                                  ndim=ndim)]
            self._register_modules(op)
            if dropout > 0:
                op += [get_dropout(dropout, nonlinearity)]
                self._register_modules(op)
            self.op_dense.append(op)

        # Transition up (maintain num out_channels)
        self.op_upsample = []
        if upsample:
            self.op_upsample += [do_upsample(mode=upsample_mode,
                                             ndim=ndim,
                                             in_channels=out_channels,
                                             out_channels=out_channels,
                                             kernel_size=3,
                                             init=init)]

        self._register_modules(self.op+self.op_upsample)

    def forward(self, input):
        # Prepare input.
        out = input
        for op in self.op:
            out = op(out)

        # Build dense block.
        tensors = [out]
        for op in self.op_dense:
            if hasattr(op, '__len__'):
                for sub_op in op:
                    out = sub_op(out)
            else:
                out = op(out)
            tensors.append(out)
            tensors = crop_stack(tensors)
            out = merge(tensors, mode=self.skip_merge_mode)
        tensors = crop_stack(tensors)

        # Block's output - merge input in?
        #
        # Regardless, all representations inside the block (all conv outputs)
        # are merged together, forming a dense skip pattern.
        out = tensors[-1]
        if self.merge_input:
            # Merge the block's input into its output.
            if len(tensors) > 1:
                out = merge(tensors, mode=self.skip_merge_mode)
        else:
            # Avoid merging the block's input into its output.
            # With this, one can avoid exponential growth in num of
            # out_channels.
            if len(tensors[1:]) > 1:
                out = merge(tensors[1:], mode=self.skip_merge_mode)

        # Upsample
        for op in self.op_upsample:
            out = op(out)

        return out
