# FCN Maker #

This code provides a simple way to recreate any encoder-decoder based fully convolutional network (FCN) for segmentation or keypoint detection. Simple recipes are provided for multiple published methods. These methods can be easily tweaked in a multitude of ways. The user can also create custom FCN networks in a simple way.

## Build a Custom FCN ##

An FCN can be built out of provided blocks or blocks written by the user. Existing blocks can be found in `fcn_maker.blocks`. These are assembled together by `fcn_maker.models.assemble_model` with the assumption that the desired architecture is an encoder-decoder network. Optionally, representations can be skipped along _long skip connections_ from the encoder (down path) to the decoder (up path).

### Assemble model ###

```python
def assemble_model(input_shape, num_classes, blocks, preprocessor=None, postprocessor=None, long_skip=True, long_skip_merge_mode='concat', init='he_normal', weight_decay=0.0001, ndim=2, verbose=True)
```

The standard model shape follows the following 5 steps.
1. Preprocessor
2. Subsampling blocks
3. Bottleneck (across blocks)
4. Upsampling blocks
5. Postprocessor

Blocks in the encoder subsample their inputs. Blocks in the encoder upsample their outputs. The bottlenecks sits between them, in the center of the network. The input to the bottleneck is subsampled while the output from the bottleneck is upsampled. See [1] for an example description of this kind of network structure.

#### Arguments ####

* __input_shape__ : A tuple specifiying the image input shape.
* __num_classes__ : The number of classes in the segmentation output. If None, no classifier will be assembled.
* __blocks__ : A list of tuples, each containing a block function and a dictionary of keyword arguments to pass to it. The length must be odd-valued. The first set of blocks before the middle one is assumed to be on the subsampling (encoder) path; the second set after the middle one is assumed to be on the upsampling (decoder) path.  
If instead of a tuple, a None is passed, the corresponding block will
simply preserve its input, passing it onto the next block.
* __preprocessor__ : A block/layer/model. The model input is run through the preprocessor before being passed to the first block.
* __postprocessor__ : A block/layer/model. The output of the last block is passed through the postprocessor before being passed to the classifier.
* __long_skip__ : A boolean specifying whether to use long skip connections from the downward path to the upward path. These can either concatenate or sum features across.
* __long_skip_merge_mode__ : Either or 'sum', 'concat' features across skip.
* __init__ : A string specifying (or a function defining) the initializer for the layers that adapt features along long skip connections.
* __weight_decay__ : The weight decay (L2 penalty) used in layers that adapt long skip connections.
* __ndim__ : The spatial dimensionality of the input and output (either 2 or 3).
* __verbose__ : A boolean specifying whether to print messages about modelstructure during construction (if True).

### Available blocks ###

* __bottleneck__ (with residual_block)
* __basic_block__ (with residual_block)
* __basic_block_mp__ (with residual_block)
* __unet_block__
* __vnet_block__
* __dense_block__

### Custom blocks ###

A block should act as a keras layer. A simple way of doing this is to define it as a closure, with a function taking parameters and returing a function which takes a tensor. See `fcn_maker.blocks` for examples.

Every block is expected by `assemble_model` to have the following arguments:
* __subsample__ : bool that specifies that the block must perform 2x (in each spatial dim) subsampling
* __upsample__ : bool that specifies that the block must perform 2x (in each spatial dim) upsampling

The expected block structure is to do subsampling at the input of the block and do upsampling at the output of the block.

This pseudocode illustrates the expected pattern:

```python
def custom_block(upsample, subsample, *args, **kwargs):
    def f(x):
        if upsample:
            x = do_upsample(x)

        # Do processing of x here
        # ...
        # ...

        if subsample:
            x = do_subsample(x)
        
        return x
    return f
```

### Simple example ###

The following defines a new U-Net style model with ResNet style bottleneck blocks, each with 32 filters. We will use 5 blocks in the encoder, 2 in the bottleneck, and 5 in the decoder. As described above, each block in the decoder performs subsampling of its inputs and each block in the encoder performs upsampling of its outputs.

This model could do 2D segmentation (but don't use it - there are much better models in the example section).

```python
from keras.layers import Conv2D
from fcn_maker.blocks import bottleneck, norm_nlin_conv
from fcn_maker.model import assemble_model

block_kwargs = block_kwargs = {'filters': 32,
                               'skip': True,
                               'normalization': BatchNormalization,
                               'nonlinearity': 'relu',
                               'ndim': 2}
blocks = [(bottleneck, block_kwargs)] * (5+1+5)
preprocessor = Conv2D(filters=32, kernel_size=3, padding='same')
postprocessor = norm_nlin_conv(filters=32, kernel_size=3, nonlinearity='relu')
model = assemble_model(input_shape=(3, None, None), num_classes=10,
                       blocks=blocks, preprocessor=preprocessor,
                       postprocessor=postprocessor, long_skip=True, ndim=2)
```

### Symmetry breaking ###

Some symmetry is assumed by `assemble_model` in that the `blocks` list is expected to have an odd number of blocks, with blocks paired across either side of the middle entry. The middle entry is the bottleneck block. The set of blocks before the middle one is on the subsampling (encoder) path. The set after the middle one is on the upsampling (decoder) path.

Blocks are paired across encoder and decoder in order to synchronize subsampling and upsampling operations and to bridge the encoder and decoder with long skip connections at those points. However, the network does not have to be symmetric since different types of blocks of any depth could be used in the encoder and decoder. If desired, any block could be skipped by passing None instead of the `(block, kwargs)` tuple in the list. This still allows for subsampling and upsampling, as needed, without performing any other processing (the `identity_block` is used by default).

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
fcn_maker.models.assemble_resunet(input_shape, num_classes, num_init_blocks, num_main_blocks, main_block_depth, init_num_filters, short_skip=True, long_skip=True, long_skip_merge_mode='concat', main_block=None, init_block=None, upsample_mode='repeat', dropout=0., normalization=BatchNormalization, norm_kwargs=None, weight_decay=None, init='he_normal', nonlinearity='relu', ndim=2, verbose=True)
```

This model was introduced in [1] to analyze the utility of short and long skip connections and normalization. 

#### Arguments ####
* __input_shape__ : A tuple specifiying the image input shape.
* __num_classes__ : The number of classes in the segmentation output.
* __num_init_blocks__ : The number of blocks of type init_block, above main_blocks. These blocks always have the same number of channels as the first convolutional layer in the model. There are num_init_blocks of these at both the beginning and the end of the network.
* __num_main_blocks__ : The number of blocks of type main_block, below init_blocks. These blocks double (halve) the number of channels at each subsampling (upsampling) after the first main_block. There are num_main_blocks of these both in the encoder and the decoder, on either side of the bottleneck.
* __main_block_depth__ : An integer or list of integers specifying the number of repetitions of each main_block. A list must contain 2*num_main_blocks+1 values (there are num_main_blocks on the contracting path and on the  expanding path, as well as as one on the across path). Zero is not a valid depth.
* __init_num_filters__ : The number of filters in the first and last convolutions (preprocessor, postprocessor). Also the number of filters in every init_block. Each main_block doubles (halves) the number of filters for each decrease (increase) in resolution.
* __short_skip__ : A boolean specifying whether to use ResNet-like shortcut connections from the input of each block to its output. The inputs are summed with the outputs.
* __long_skip__ : A boolean specifying whether to use long skip connections from the downward path to the upward path. These can either concatenate or sum features across.
* __long_skip_merge_mode__ : Either or 'sum', 'concat' features across skip.
* __main_block__ : A layer defining the main_block (bottleneck by default).
* __init_block__ : A layer defining the init_block (basic_block_mp by defau
* __upsample_mode__ : Either 'repeat' or 'conv'. With 'repeat', rows and colums are repeated as in nearest neighbour interpolation. With 'conv', upscaling is done via transposed convolution.
* __dropout__ : A float [0, 1] specifying the dropout probability, introduced in every block.
* __normalization__ : The normalization to apply to layers (by default: batch normalization). If None, no normalization is applied.
* __norm_kwargs__ : Keyword arguments to pass to batch norm layers. For batch normalization, default momentum is 0.9. weight_decay : The weight decay (L2 penalty) used in every convolution (float).
* __weight_decay__ : The weight decay (L2 penalty) used in every convolution (float).
* __init__ : A string specifying (or a function defining) the initializer for layers.
* __nonlinearity__ : A string (or function defining) the nonlinearity.
* __ndim__ : The spatial dimensionality of the input and output (either 2 or 3).
* __verbose__ : A boolean specifying whether to print messages about model structure during construction (if True).

#### Example ####

The variant described in [1] can be instantiated with:
```python
from fcn_maker.model import assemble_resunet
assemble_resunet(input_shape=(1, None, None),
                 num_classes=1,
                 num_init_blocks=2,
                 num_main_blocks=3,
                 main_block_depth=[3, 8, 10, 3, 10, 8, 3],
                 init_num_filters=32)
```

The construction in `assemble_resunet` assumes that there are two categories of blocks: `init_block` and `main_block`. The `init_block` blocks are found at the beginning and end of the network, just after the preprocessor and just before the post-processor (here, both are assumed to be a single convolution layer). They preserve the number of filters while reducing feature resolution at the encoder path (down path) and increasing feature resolution at the decoder path (up path). All other blocks are of the `main_block` category; these double the number of features with each subsampling and halve the number of features with each upsampling. In the paper, the blocks are set thus:  
`init_block` : `fcn_maker.blocks.basic_block_mp`  
`main_block` : `fcn_maker.blocks.bottleneck`


### UNet ###

```python
fcn_maker.models.assemble_unet(input_shape, num_classes, init_num_filters=64, num_pooling=4, short_skip=False, long_skip=True, long_skip_merge_mode='concat', upsample_mode='conv', dropout=0., normalization=None, norm_kwargs=None, weight_decay=None, init='he_normal', nonlinearity='relu', halve_features_on_upsample=True, ndim=2, verbose=True)
```

This model was introduced in [2] for ISBI EM neuronal segmentation. It extended the long skip connections from earlier FCN work to concatenating long skips from encoder to decoder in an encoder-decoder architecture like that of [7].

#### Arguments ####
* __input_shape__ : A tuple specifiying the image input shape.
* __num_classes__ : The number of classes in the segmentation output.
* __init_num_filters__ : The number of filters used in the convolutions of the first and lost blocks in the network. With every downsampling, the number of filters is doubled; with every upsampling, it is halved.  
There are two convolutions in a unet_block so a a list/tuple of two values can be passed to set each convolution separately. For example, the original 2D UNet uses init_num_filters=64 or (64, 64) while the original 3D UNet uses init_num_filters=(32, 64).
* __num_pooling__ : The number of pooling (and thus upsampling) operations to perform in the network.
* __short_skip__ : A boolean specifying whether to use ResNet-like shortcut connections from the input of each block to its output. The inputs are summed with the outputs.
* __long_skip__ : A boolean specifying whether to use long skip connections from the downward path to the upward path. These can either concatenate or sum features across.
* __long_skip_merge_mode__ : Either or 'sum', 'concat' features across skip.
* __upsample_mode__ : Either 'repeat' or 'conv'. With 'repeat', rows and colums are repeated as in nearest neighbour interpolation. With 'conv', upscaling is done via transposed convolution.
* __dropout__ : A float [0, 1] specifying the dropout probability, introduced in every block.
* __normalization__ : The normalization to apply to layers (none by default). Recommended to pass keras's BatchNormalization when using short_skip==True.
* __norm_kwargs__ : Keyword arguments to pass to batch norm layers. For batch normalization, default momentum is 0.9. weight_decay : The weight decay (L2 penalty) used in every convolution (float).
* __weight_decay__ : The weight decay (L2 penalty) used in every convolution (float).
* __init__ : A string specifying (or a function defining) the initializer for layers.
* __nonlinearity__ : A string (or function defining) the nonlinearity.
* __halve_features_on_upsample__ : As in the original 2D UNet, have each block halve the number of feature maps when upsampling. This is not done in the original 3D UNet.
* __ndim__ : The spatial dimensionality of the input and output (either 2 or 3).
* __verbose__ : A boolean specifying whether to print messages about model structure during construction (if True).

#### Examples ####

The 2D variant described in [2] can be instantiated with:
```python
from fcn_maker.model import assemble_unet
assemble_unet(input_shape=(1, None, None), num_classes=1)
```

The 3D variant described in [3] can be instantiated with:
```python
from fcn_maker.model import assemble_unet
assemble_unet(input_shape=(1, None, None, None),
              init_num_filters=(32, 64),
              num_classes=1, ndim=3, num_pooling=3,
              halve_features_on_upsample=False)
```

### VNet ###

```python
fcn_maker.models.assemble_vnet(input_shape, num_classes, init_num_filters=32, num_pooling=4, short_skip=True, long_skip=True, long_skip_merge_mode='concat', upsample_mode='conv', dropout=0., normalization=None, norm_kwargs=None, init=VarianceScaling(scale=3., mode='fan_avg'), weight_decay=None, nonlinearity='prelu', ndim=3, verbose=True)
```

This model was introduced in [4] for prostate segmentation in MRI.

#### Arguments ####
* __input_shape__ : A tuple specifiying the image input shape.
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
* __init__ : A string specifying (or a function defining) the initializer for layers.
* __weight_decay__ : The weight decay (L2 penalty) used in every convolution (float).
* __nonlinearity__ : A string (or function defining) the nonlinearity.
* __ndim__ : The spatial dimensionality of the input and output (either 2 or 3).
* __verbose__ : A boolean specifying whether to print messages about model structure during construction (if True).

#### Example ####

The model in [4] can be instantiated with:

```python
from fcn_maker.model import assemble_vnet
model = assemble_vnet(input_shape=(1, None, None, None), num_classes=1)
```

### FC-DenseNet (100 layers Tiramisu) ###

```python
fcn_maker.models.assemble_fcdensenet(input_shape, num_classes, block_depth=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4], num_blocks=11, init_num_filters=48, growth_rate=16, long_skip=True, skip_merge_mode='concat', upsample_mode='conv', dropout=0.2, normalization=BatchNormalization, norm_kwargs=None, init='he_uniform', weight_decay=0.0001, nonlinearity='relu', ndim=2, verbose=True)
```

This model was introduced in [5] for prostate segmentation in MRI.

#### Arguments ####
* __input_shape__ : A tuple specifiying the image input shape.
* __num_classes__ : The number of classes in the segmentation output.
* __block_depth__ : An integer or list of integers specifying the number of convolutions in each dense block. A list must contain num_blocks values (there are an equal number of blocks on the contracting and expanding paths, as well as as one bottleneck on the across path). Zero is a valid depth (the block still sub/up-samples).
* __num_blocks__ : The total number of dense blocks in the network. Must be an odd number.
* __init_num_filters__ : The number of filters in the first pair and last pair of convolutions in the network. With every downsampling, the number of filters is doubled; with every upsampling, it is halved.
* __growth_rate__ : The linear rate with which the number of filters increases after each convolution, when using 'concat' skip_merge_mode. In 'sum' mode, this argument simply sets the number of filters for every convolution layer except the first and last ones (preprocessor and postprocessor).  
If set to None, the number of filters for each dense_block will double after each pooling operation and halve after each upsampling operation.
* __long_skip__ : A boolean specifying whether to use long skip connections from the downward path to the upward path. These can either concatenate or sum features across.
* __skip_merge_mode__ : Either or 'sum', 'concat' features across skip.
* __upsample_mode__ : Either 'repeat' or 'conv'. With 'repeat', rows and colums are repeated as in nearest neighbour interpolation. With 'conv', upscaling is done via transposed convolution.
* __dropout__ : A float [0, 1] specifying the dropout probability, introduced in every block.
* __normalization__ : The normalization to apply to layers (none by default).
* __norm_kwargs__ : Keyword arguments to pass to batch norm layers. For batch normalization, default momentum is 0.9.
* __init__ : A string specifying (or a function defining) the initializer for layers.
* __weight_decay__ : The weight decay (L2 penalty) used in every convolution (float).
* __nonlinearity__ : A string (or function defining) the nonlinearity.
* __ndim__ : The spatial dimensionality of the input and output (either 2 or 3).
* __verbose__ : A boolean specifying whether to print messages about model structure during construction (if True).

#### Example ####

The 103 layer FC-DenseNet model in [5] can be instantiated with:

```python
from fcn_maker.model import assemble_fcdensenet
model = assemble_fcdensenet(input_shape=(1, None, None), num_classes=11)
```

One possible change is to make all skip connections (long skips and those in the dense blocks) merge via summation rather than concatenation. When doing so, it is suggested to set `growth_rate=None` so that the number of filters doubles (halves) in each block with each subsampling (upsampling) as in the ResUNet, the UNet, and the VNet.

```python
from fcn_maker.model import assemble_fcdensenet
model = assemble_fcdensenet(input_shape=(1, None, None), num_classes=11,
                            skip_merge_mode='sum', growth_rate=None)
```


## Example Task: ISBI EM segmentation ##

ISBI neuronal EM segmentation challenge. Example runs a ResUNet from [1].

Download data files into
```
/tmp/datasets/isbi_2012_em/
```

Run example_resunet_isbi-em.py
