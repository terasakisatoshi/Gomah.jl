# Gomah.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://terasakisatoshi.github.io/Gomah.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://terasakisatoshi.github.io/Gomah.jl/dev)
[![CircleCI](https://circleci.com/gh/terasakisatoshi/Gomah.jl/tree/master.svg?style=svg)](https://circleci.com/gh/terasakisatoshi/Gomah.jl/tree/master)
[![Build Status](https://travis-ci.org/terasakisatoshi/Gomah.jl.svg?branch=master)](https://travis-ci.org/terasakisatoshi/Gomah.jl)
[![Codecov](https://codecov.io/gh/terasakisatoshi/Gomah.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/terasakisatoshi/Gomah.jl)
[![Coveralls](https://coveralls.io/repos/github/terasakisatoshi/Gomah.jl/badge.svg?branch=master)](https://coveralls.io/github/terasakisatoshi/Gomah.jl?branch=master)


# About this repo

- This repo provides train/inference procedure of MNIST model known as Hello World of Deep Learning on [Julia](https://julialang.org/) runtime.
These techniques are based on [Chainer](https://chainer.org/) and [PyCall.jl](https://github.com/JuliaPy/PyCall.jl).
- NEW: Add feature to convert ResNet50/Chainer -> ResNet50/Flux.jl

# Usage

## How to install
- This package is not registered as official julia package, so called 野良(nora), which means we should specify repository url:
- Note that Package `Gomah.jl` depends on `PyCall.jl`. So before installing, We recommend set environment variable in Julia.

```julia
$ julia
julia> ENV["PYTHON"] = Sys.which("python3")
pkg> add https://github.com/terasakisatoshi/Gomah.jl.git
julia> using Gomah
```

## Call Chainer script from Julia via `Gomah.jl`

- `PyCall.jl` provides interface between Python and Julia.
- This means you can construct training script of Chainer on Julia environment.
- If you are familiar with some Deep learnig framework, checkout our `src/mnist.jl`.
- We provide an example of training MNIST classifier.

```
$ julia
julia> using Gomah
julia> train()
epoch       main/loss   main/accuracy  validation/main/loss  validation/main/accuracy  elapsed_time
1           0.56446     0.840621       0.28946               0.914727                  3.49408       
2           0.248238    0.927429       0.212789              0.937381                  5.90154       
3           0.189395    0.945183       0.175694              0.948423                  8.43833       
4           0.152917    0.95592        0.145494              0.957898                  10.832        
5           0.128008    0.963307       0.128785              0.962386                  13.2714       
6           0.108765    0.968306       0.121768              0.961222                  15.6906       
7           0.0948146   0.97286        0.103434              0.969201                  18.361        
8           0.0854993   0.974762       0.10229               0.96818                   20.7444       
9           0.0746463   0.977655       0.0916977             0.972739                  23.1146       
10          0.0663983   0.980598       0.0889726             0.972573                  25.5239    
julia> predict()
accuracy for test set = 97.31 [%]
```

# Convert ResNet/Chainer -> ResNet/Flux.jl

- We found the structure (shape) of parameter i.e. weight of Chainer is similar that of Flux.
- The structure of weight of Convolution of Chainer is `NCHW`. On the other hand, Conv of Flux.jl has weight its shape is `WHCN`, where `N` is batchsize, `H` (resp. `W`) is height (resp. width) of kernel and `C` is num of channel.
- We provided script to convert ResNet50 of Chainer to that of Flux.jl
- Here is a example of How to use converted model. What you have to do is ...
  - Install chainer and chainercv
  - Install Flux.jl, PyCall, Gomah.jl
  - Prepare sample RGB image. e.g. `pineapple.png`
  - Run the following the script.

```julia
using Gomah
using Gomah: L, np, reversedims
using Flux
using PyCall

py"""
import chainer
import chainercv
import numpy as np
num = 50
PyResNet = chainercv.links.model.resnet.ResNet
resnet = PyResNet(num, pretrained_model = "imagenet")
img=chainercv.utils.read_image("pineapple.png",np.float32)
img=chainercv.transforms.resize(img,(224,224))
_imagenet_mean = np.array(
            [123.15163084, 115.90288257, 103.0626238],
            dtype=np.float32
        )[:, np.newaxis, np.newaxis]
img=img-_imagenet_mean
img=np.expand_dims(img,axis=0)
resnet.pick=resnet.layer_names
with chainer.using_config('train', False):
    pyret=resnet(img)
    result=np.squeeze(pyret[-1].array)
    chidx=int(np.argmax(result))
    chprob=100*float(result[chidx])
    print(chidx)
    print(chprob)
"""

num = 50
myresnet = ResNet(num)
Flux.testmode!(myresnet, true)
img = reversedims(py"img");
@show size(img)
ret, d = myresnet(img);
flidx = argmax(ret)
flprob = 100ret[argmax(ret)]
@show flidx,flprob
# Python: The index of array begins with 0
# Julia: THe index of array begin with 1
@show Int(py"chidx") == flidx[1]-1
# For some reason, our converted model outputs a slightly different output.
@show Float32(py"chprob") - flprob
```

# Why Gomah(ごまぁ）?

- My favorite thing [ごまちゃん](http://gogo-gomachan.com/)
- Inspired by [Menoh](https://github.com/pfnet-research/menoh)
- Gomah.jl will be promoted as DNN inference library (in the future)

# Acknowledgement

- Script for training MNIST is based on [this notebook](https://gist.github.com/regonn/d2acf5a20a1b3ec34d8e483af510b4a3).
  - See also [Julia で Chainer を動かすぞ💪](https://speakerdeck.com/regonn/julia-de-chainer-wodong-kasuzo?slide=2)
- Construct Chain is based on [Flux.jl](https://github.com/FluxML/Flux.jl)
- Basic implementation of ResNet is taken from [ChainerCV](https://github.com/chainer/chainercv) project.
