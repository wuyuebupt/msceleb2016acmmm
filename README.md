# msceleb2016acmm



## Requirements
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)

See the [installation instructions](https://github.com/facebook/fb.resnet.torch/INSTALL.md) for a step-by-step guide.
- Install [Torch](http://torch.ch/docs/getting-started.html) on a machine with CUDA GPU
- Install [cuDNN v4 or v5](https://developer.nvidia.com/cudnn) and the Torch [cuDNN bindings](https://github.com/soumith/cudnn.torch/tree/R4)
- Download the [ImageNet](http://image-net.org/download-images) dataset and [move validation images](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) to labeled subfolders

If you already have Torch installed, update `nn`, `cunn`, and `cudnn`.



## Usage:
th classify_6_2.lua list


## Results:

hillary_0_FaceId-0.jpg m.0d06m5:0.95839496697635	
forward time: 0.59
	
trump_0_FaceId-0.jpg m.0cqt90:0.70081973075867	
forward time: 0.32	




## Notes:

m.0cqt90	"Donald Trump"@en

m.0d06m5	"Hillary Rodham Clinton"@en
