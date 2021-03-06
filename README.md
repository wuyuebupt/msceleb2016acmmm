# msceleb2016acmm



## Requirements

See the [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) for a step-by-step guide.
- Install [Torch](http://torch.ch/docs/getting-started.html) on a machine with CUDA GPU
- Install [cuDNN v4 or v5](https://developer.nvidia.com/cudnn) and the Torch [cuDNN bindings](https://github.com/soumith/cudnn.torch/tree/R4)


## Models download

Download [models.tar](https://drive.google.com/file/d/0B0W0rPKOeIByX3VVcGdjWUNwejA/view?usp=sharing) into the same directory of this code.

Then unzip the model file by "tar -xvf models.tar"

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

## Reference

Dataset: [MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition, ECCV 2016](http://link.springer.com/chapter/10.1007%2F978-3-319-46487-9_6)

MSR Image Recognition Challenge (IRC)@ACM Multimedia 2016: https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/

Our ACM MM 2016 Multimedia Grand Challenge Paper: [Deep Convolutional Neural Network with Independent Softmax for Large Scale Face Recognition](http://dl.acm.org/citation.cfm?doid=2964284.2984060)
