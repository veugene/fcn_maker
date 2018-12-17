# FCN Maker #

Build any FCN.

This code provides a simple way to (re)create any encoder-decoder based fully convolutional network (FCN) for segmentation or keypoint detection. Simple recipes are provided for multiple published methods. These methods can be easily tweaked in many ways. The user can also create custom FCN networks.

__Uses:__ pytorch

## Example Models ##

Currently, the following are trivial to instantiate:
* ResUNet [1]
* 2D UNet [2]
* 3D UNet [3]
* VNet [4]
* FC-DenseNet (Tiramisu) [5]

The following variants will soon be added:
* Recombinator Network [6]
* DeconvNet [7]
* SegNet [8]

[1] Michal Drozdzal & Eugene Vorontsov, et al. "The importance of skip connections in biomedical image segmentation." International Workshop on Large-Scale Annotation of Biomedical Data and Expert Label Synthesis. Springer International Publishing, 2016.
https://arxiv.org/abs/1608.04117

[2] Olaf Ronneberger, et al. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2015.
https://arxiv.org/abs/1505.04597

[3] Özgün Çiçek, et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer International Publishing, 2016.
https://arxiv.org/abs/1606.06650

[4] Fausto Milletari, et al. "V-net: Fully convolutional neural networks for volumetric medical image segmentation." 3D Vision (3DV), 2016 Fourth International Conference on. IEEE, 2016.
https://arxiv.org/abs/1606.04797

[5] Simon Jégou, et al. "The one hundred layers tiramisu: Fully convolutional densenets for semantic segmentation." Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on. IEEE, 2017.
https://arxiv.org/abs/1611.09326

[6] Sina Honari, et al. "Recombinator networks: Learning coarse-to-fine feature aggregation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.
https://arxiv.org/abs/1511.07356

[7] Hyeonwoo Noh, et al. "Learning deconvolution network for semantic segmentation." Proceedings of the IEEE International Conference on Computer Vision. 2015.
(earlier version at https://arxiv.org/abs/1505.04366)

[8] Vijay Badrinarayanan, et al. "Segnet: A deep convolutional encoder-decoder architecture for image segmentation." arXiv preprint arXiv:1511.00561 (2015).
https://arxiv.org/abs/1511.00561


### ResUNet ###

```python
fcn_maker.models.assemble_resunet(in_channels, num_classes, num_init_blocks,
    num_main_blocks, main_block_depth, init_num_filters, short_skip=True,
    long_skip=True, long_skip_merge_mode='concat', main_block=None,
    init_block=None, upsample_mode='repeat', dropout=0., 
    normalization=batch_normalization, norm_kwargs=None, conv_padding=True,
    init='kaiming_normal_', nonlinearity='ReLU', ndim=2, verbose=True)
```

This model was introduced in [1] to analyze the utility of short and long skip connections and normalization. 

#### Arguments ####
* __in_channels__ : Number of channels in the input.
* __num_classes__ : The number of classes in the segmentation output.
* __num_init_blocks__ : The number of blocks of type init_block, above main_blocks. These blocks always have the same number of channels as the first convolutional layer in the model. There are num_init_blocks of these at both the beginning and the end of the network.
* __num_main_blocks__ : The number of blocks of type main_block, below init_blocks. These blocks double (halve) the number of channels at each subsampling (upsampling) after the first main_block. There are num_main_blocks of these both in the encoder and the decoder, on either side of the bottleneck.
* __main_block_depth__ : An integer or list of integers specifying the number of repetitions of each main_block. A list must contain 2*num_main_blocks+1 values (there are num_main_blocks on the contracting path and on the  expanding path, as well as as one on the across path). Zero is not a valid depth.
* __init_num_filters__ : The number of filters in the first and last convolutions. Also the number of filters in every init_block. Each main_block doubles (halves) the number of filters for each decrease (increase) in resolution.
* __short_skip__ : A boolean specifying whether to use ResNet-like shortcut connections from the input of each block to its output. The inputs are summed with the outputs.
* __long_skip__ : A boolean specifying whether to use long skip connections from the downward path to the upward path. These can either concatenate or sum features across.
* __long_skip_merge_mode__ : Either or 'sum', 'concat' features across skip.
* __main_block__ : A layer defining the main_block (bottleneck by default).
* __init_block__ : A layer defining the init_block (tiny_block by default).
* __upsample_mode__ : Either 'repeat' or 'conv'. With 'repeat', rows and colums are repeated as in nearest neighbour interpolation. With 'conv', upscaling is done via transposed convolution.
* __dropout__ : A float [0, 1] specifying the dropout probability, introduced in every block.
* __normalization__ : The normalization to apply to layers (by default: batch normalization). If None, no normalization is applied.
* __norm_kwargs__ : Keyword arguments to pass to batch norm layers. For batch normalization, default momentum is 0.9.
* __conv_padding__ : Whether to use zero-padding for convolutions. If True, the output size is the same as the input size; if False, the output is smaller than the input size.
* __init__ : A string specifying (or a function defining) the initializer for layers.
* __nonlinearity__ : A string (or function defining) the nonlinearity.
* __ndim__ : The spatial dimensionality of the input and output (either 2 or 3).
* __verbose__ : A boolean specifying whether to print messages about model structure during construction (if True).

#### Example ####

The variant described in [1] can be instantiated with:
```python
from fcn_maker.model import assemble_resunet
assemble_resunet(in_channels=1,
                 num_classes=1,
                 num_init_blocks=2,
                 num_main_blocks=3,
                 main_block_depth=[3, 8, 10, 3, 10, 8, 3],
                 init_num_filters=32)
```

In a ResUNet, the first and last operations (before the classifier) are always convolutions. All blocks are arranged between the first and last convolution. The model constructor, `assemble_resunet`, expects two categories of blocks: `init_block` and `main_block`.

`init_block` : blocks at the beginning and end of the network. These always output representations with `init_num_filters` channels.

`main_block` : all blocks in between. These double the number of features with each subsample operation and halve the number of features with each upsample operation.

In the paper, the blocks are set thus:  
`init_block` : `fcn_maker.blocks.tiny_block`  
`main_block` : `fcn_maker.blocks.bottleneck`

### UNet ###

```python
fcn_maker.models.assemble_unet(in_channels, num_classes, init_num_filters=64,
    num_pooling=4, short_skip=False, long_skip=True,
    long_skip_merge_mode='concat', upsample_mode='conv', dropout=0.,
    normalization=None, norm_kwargs=None, conv_padding=True,
    init='kaiming_normal_', nonlinearity='ReLU',
    halve_features_on_upsample=True, ndim=2, verbose=True)
```

This model was introduced in [2] for ISBI EM neuronal segmentation. It extended the long skip connections from earlier FCN work to concatenating long skips from encoder to decoder in an encoder-decoder architecture like that of [7].

#### Arguments ####
* __in_channels__ : Number of channels in the input.
* __num_classes__ : The number of classes in the segmentation output.
* __init_num_filters__ : The number of filters used in the convolutions of the first and lost blocks in the network. With every downsampling, the number of filters is doubled; with every upsampling, it is halved.  
There are two convolutions in a unet_block so a a list/tuple of two values can be passed to set each convolution separately. For example, the original 2D UNet uses init_num_filters=64 or (64, 64) while the original 3D UNet uses init_num_filters=(32, 64).
* __num_pooling__ : The number of pooling (and thus upsampling) operations to perform in the network.
* __short_skip__ : A boolean specifying whether to use ResNet-like shortcut connections from the input of each block to its output. The inputs are summed with the outputs.
* __long_skip__ : A boolean specifying whether to use long skip connections from the downward path to the upward path. These can either concatenate or sum features across.
* __long_skip_merge_mode__ : Either or 'sum', 'concat' features across skip.
* __upsample_mode__ : Either 'repeat' or 'conv'. With 'repeat', rows and colums are repeated as in nearest neighbour interpolation. With 'conv', upscaling is done via transposed convolution.
* __dropout__ : A float [0, 1] specifying the dropout probability, introduced in every block.
* __normalization__ : The normalization to apply to layers (none by default). Recommended to pass batch_normalization when using short_skip==True.
* __norm_kwargs__ : Keyword arguments to pass to batch norm layers. For batch normalization, default momentum is 0.9.
* __conv_padding__ : Whether to use zero-padding for convolutions. If True, the output size is the same as the input size; if False, the output is smaller than the input size.
* __init__ : A string specifying (or a function defining) the initializer for layers.
* __nonlinearity__ : A string (or function defining) the nonlinearity.
* __halve_features_on_upsample__ : As in the original 2D UNet, have each block halve the number of feature maps when upsampling. This is not done in the original 3D UNet.
* __ndim__ : The spatial dimensionality of the input and output (either 2 or 3).
* __verbose__ : A boolean specifying whether to print messages about model structure during construction (if True).

#### Examples ####

The 2D variant described in [2] can be instantiated with:
```python
from fcn_maker.model import assemble_unet
assemble_unet(in_channels=(1, None, None), num_classes=1)
```

The 3D variant described in [3] can be instantiated with:
```python
from fcn_maker.model import assemble_unet
assemble_unet(in_channels=1,
              init_num_filters=(32, 64),
              num_classes=1, ndim=3, num_pooling=3,
              halve_features_on_upsample=False)
```

### VNet ###

```python
fcn_maker.models.assemble_vnet(in_channels, num_classes, init_num_filters=32,
    num_pooling=4, short_skip=True, long_skip=True,
    long_skip_merge_mode='concat', upsample_mode='conv', dropout=0.,
    normalization=None, norm_kwargs=None, conv_padding=True,
    init='xavier_uniform_', nonlinearity='PReLU', ndim=3, verbose=True)
```

This model was introduced in [4] for prostate segmentation in MRI.

#### Arguments ####
* __in_channels__ : Number of channels in the input.
* __num_classes__ : The number of classes in the segmentation output.
* __init_num_filters__ : The number of filters in the first pair and last pair of convolutions in the network. With every downsampling, the number of filters is doubled; with every upsampling, it is halved.
* __num_pooling__ : The number of pooling (and thus upsampling) operations to perform in the network.
* __short_skip__ : A boolean specifying whether to use ResNet-like shortcut connections from the input of each block to its output. The inputs are summed with the outputs.
* __long_skip__ : A boolean specifying whether to use long skip connections from the downward path to the upward path. These can either concatenate or sum features across.
* __long_skip_merge_mode__ : Either or 'sum', 'concat' features across skip.
* __upsample_mode__ : Either 'repeat' or 'conv'. With 'repeat', rows and colums are repeated as in nearest neighbour interpolation. With 'conv', upscaling is done via transposed convolution.
* __dropout__ : A float [0, 1] specifying the dropout probability, introduced in every block.
* __normalization__ : The normalization to apply to layers (none by default).
* __norm_kwargs__ : Keyword arguments to pass to batch norm layers. For batch normalization, default momentum is 0.9.
* __conv_padding__ : Whether to use zero-padding for convolutions. If True, the output size is the same as the input size; if False, the output is smaller than the input size.
* __init__ : A string specifying (or a function defining) the initializer for layers.
* __nonlinearity__ : A string (or function defining) the nonlinearity.
* __ndim__ : The spatial dimensionality of the input and output (either 2 or 3).
* __verbose__ : A boolean specifying whether to print messages about model structure during construction (if True).

#### Example ####

The model in [4] can be instantiated with:

```python
from fcn_maker.model import assemble_vnet
model = assemble_vnet(in_channels=1, num_classes=1)
```

### FC-DenseNet (100 layers Tiramisu) ###

```python
fcn_maker.models.assemble_fcdensenet(in_channels, num_classes,
    block_depth=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4], num_blocks=11,
    init_num_filters=48, growth_rate=16, long_skip=True,
    skip_merge_mode='concat', upsample_mode='conv', dropout=0.2,
    normalization=batch_normalization, norm_kwargs=None, conv_padding=True,
    init='kaiming_uniform_', nonlinearity='ReLU', ndim=2, verbose=True)
```

This model was introduced in [5] for prostate segmentation in MRI.

#### Arguments ####
* __in_channels__ : Number of channels in the input.
* __num_classes__ : The number of classes in the segmentation output.
* __block_depth__ : An integer or list of integers specifying the number of convolutions in each dense block. A list must contain num_blocks values (there are an equal number of blocks on the contracting and expanding paths, as well as as one bottleneck on the across path). Zero is a valid depth (the block still sub/up-samples).
* __num_blocks__ : The total number of dense blocks in the network. Must be an odd number.
* __init_num_filters__ : The number of filters in the first pair and last pair of convolutions in the network. With every downsampling, the number of filters is doubled; with every upsampling, it is halved.
* __growth_rate__ : The linear rate with which the number of filters increases after each convolution, when using 'concat' skip_merge_mode. In 'sum' mode, this argument simply sets the number of filters for every convolution layer except the first and last ones.  
If set to None, the number of filters for each dense_block will double after each pooling operation and halve after each upsampling operation.
* __long_skip__ : A boolean specifying whether to use long skip connections from the downward path to the upward path. These can either concatenate or sum features across.
* __skip_merge_mode__ : Either or 'sum', 'concat' features across skip.
* __upsample_mode__ : Either 'repeat' or 'conv'. With 'repeat', rows and colums are repeated as in nearest neighbour interpolation. With 'conv', upscaling is done via transposed convolution.
* __dropout__ : A float [0, 1] specifying the dropout probability, introduced in every block.
* __normalization__ : The normalization to apply to layers (none by default).
* __norm_kwargs__ : Keyword arguments to pass to batch norm layers. For batch normalization, default momentum is 0.9.
* __conv_padding__ : Whether to use zero-padding for convolutions. If True, the output size is the same as the input size; if False, the output is smaller than the input size.
* __init__ : A string specifying (or a function defining) the initializer for layers.
* __nonlinearity__ : A string (or function defining) the nonlinearity.
* __ndim__ : The spatial dimensionality of the input and output (either 2 or 3).
* __verbose__ : A boolean specifying whether to print messages about model structure during construction (if True).

#### Example ####

The 103 layer FC-DenseNet model in [5] can be instantiated with:

```python
from fcn_maker.model import assemble_fcdensenet
model = assemble_fcdensenet(in_channels=1, num_classes=11)
```

One possible change is to make all skip connections (long skips and those in the dense blocks) merge via summation rather than concatenation. When doing so, it is suggested to set `growth_rate=None` so that the number of filters doubles (halves) in each block with each subsampling (upsampling) as in the ResUNet, the UNet, and the VNet.

```python
from fcn_maker.model import assemble_fcdensenet
model = assemble_fcdensenet(in_channels=1, num_classes=11,
                            skip_merge_mode='sum', growth_rate=None)
```

## Build a Custom FCN ##

An FCN can be built from a list of blocks (modules). Blocks can be found in `fcn_maker.blocks` or written by the user. These are assembled together by `fcn_maker.models.fcn` with the assumption that the desired architecture is an encoder-decoder network. Optionally, representations can be skipped along _long skip connections_ from the encoder (down path) to the decoder (up path). See [1] for details.

### The model module ###

```python
class fcn(in_channels, num_classes, blocks, long_skip=True, long_skip_merge_mode='concat', init='kaiming_normal_', ndim=2, verbose=True)
```

The standard model structure is composed of the following sets of blocks:
1. Encoder blocks
2. Bottleneck (across blocks)
3. Decoder blocks

__(1)__ Blocks in the encoder spatially subsample their inputs (as in a CNN).  
__(2)__ A single block that spatially subsamples its input and spatially upsamples its output.  
__(3)__ Blocks in the decoder spatially upsample their outputs (inverse of a CNN).

Note: the very first block and the very last block do not change spatial resolution.

The `fcn` class assumes that there are an equal number of blocks in the encoder and decoder paths. Further, it assumes that blocks are paired across the two paths. For an FCN with 11 blocks, blocks in the block list are paired thus (matching numbers form pairs):

```
(   encoder   )     (   decoder   )
(1, 2, 3, 4, 5) (6) (5, 4, 3, 2, 1)
           ( bottleneck )
```

For each pair, a _long skip connection_ is (optionally) created from the encoder to the decoder.

All blocks are provided to `fcn` in a single list, in feedforward order, input to output (as listed above). The `fcn` class takes care of initializing the blocks. Thus, each block in the block list is specified as a `(block, kwargs)` tuple, with all keyward arguments included in `kwargs`.  The class automatically specifies which blocks change resolution.

Although blocks are paired, the model does not have to be symmetric. The two blocks in a pair can be completely different; any block could also be effectively skipped by replacing it with a placeholder. This is done by specifying `None` instead of a `(block, kwargs)` tuple in the block list.

A `None` is internally replaced by `identity_block` which does nothing (beyond a possible automatic change in spatial resolution, as needed). A long skip connection is not created for any pair with such a placeholder block.

#### Arguments ####

* __in_channels__ : Number of channels in the input.
* __num_classes__ : The number of classes in the segmentation output. If None, no classifier will be assembled.
* __blocks__ : A list of tuples, each containing a block function and a dictionary of keyword arguments to pass to it. The length must be odd-valued. The first set of blocks before the middle one is assumed to be on the downsampling (encoder) path; the second set after the middle one is assumed to be on the upsampling (decoder) path. The  first and last block do not change resolution. If instead of a tuple, a None is passed, the corresponding block will simply preserve its input, passing it onto the next block.
* __long_skip__ : A boolean specifying whether to use long skip connections from the downward path to the upward path. These can either concatenate or sum features across.
* __long_skip_merge_mode__ : Either or 'sum', 'concat' features across skip.
* __conv_padding__ : Whether to use zero-padding for convolutions. If True, the output size is the same as the input size; if False, the output is smaller than the input size.
* __init__ : A string specifying (or a function defining) the initializer for the layers that adapt features along long skip connections.
* __ndim__ : The spatial dimensionality of the input and output (either 2 or 3).
* __verbose__ : A boolean specifying whether to print messages about modelstructure during construction (if True).

### Available blocks ###

* __bottleneck__ (with residual_block)
* __basic_block__ (with residual_block)
* __tiny_block__ (with residual_block)
* __unet_block__
* __vnet_block__
* __dense_block__

### Simple example: custom model ###

The following defines a new U-Net style model with ResNet style "basic" blocks, each with 32 filters but using an ELU nonlinearity instead of ReLU. We will use 5 blocks in the encoder, 1 in the bottleneck, and 5 in the decoder.

```python
from torch.nn import Conv2d
from fcn_maker.blocks import basic_block
from fcn_maker.model import fcn

block_kwargs = {'num_filters': 32,
                'nonlinearity': 'ELU'}
block_list = [(basic_block, block_kwargs)] * (5+1+5)
model = fcn(in_channels=(3, None, None), num_classes=10,
            blocks=block_list, long_skip=True, ndim=2)
```

Arguments to change spatial resolution (`subsample` and `upsample`) are not included in `block_kwargs`. This is because the model will handle this automatically, as described above.

### Custom blocks ###

A custom block inherits from `fcn_maker.blocks.block_abstract` (itself a torch module). This base class defines the expected arguments and attributes of a block class. Expected arguments and attributes are:
* __in_channels__ : The number of channels in the input.
* __num_filters__ : The number of filters to apply to input data.
* __subsample__ : Whether to do spatial subsampling (x2 in each dimension).
* __upsample__ : Whether to do spatial upsampling (x2 in each dimension).

Note that there is no `out_channels` argument -- the block is expected to determine this value and set it as an object attribute during initialization. This allows the number of channels in the output to be dependent on any number of arguments.

As in any torch module, the `forward` method must be defined.

The expected block structure is to optionally do subsampling at the input of the block and optionally do upsampling at the output of the block.

This example code illustrates the expected pattern (given some arbitrary modules `upsample_operation` and `subsample_operation`):

```python
class custom_block(fcn_maker.blocks.block_abstract):
    def __init__(self, in_channels, num_filters, subsample, upsample):
        super(custom_block, self).__init__()
        self.op = []
        if subsample:
            self.op.append(subsample_operation)  
        
        # Add more operations
        # ...
        # ...
        
        if upsample:
            self.op.append(upsample_operation)
    
    def forward(self, input):
        out = input
        for op in self.op:
            out = op(out)
        return out
```

## Example Task: ISBI EM segmentation ##

ISBI neuronal EM segmentation challenge. Example runs a ResUNet from [1].

Download data files from `http://brainiac2.mit.edu/isbi_challenge/home` into:
```
/tmp/datasets/isbi_2012_em/
```

Run example_resunet_isbi-em.py
