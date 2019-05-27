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

# Usage

- This package is not registered as official julia package, so called 野良(nora), which means we should specify repository url:

```
$ julia
pkg> add https://github.com/terasakisatoshi/Gomah.jl.git
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

# Why Gomah(ごまぁ）?

- My favorite thing [ごまちゃん](http://gogo-gomachan.com/)
- Inspired by [Menoh](https://github.com/pfnet-research/menoh)
- Gomah.jl will be promoted as DNN inference library (in the future)

# Acknowledgement

- Script for training MNIS is based on [this notebook](https://gist.github.com/regonn/d2acf5a20a1b3ec34d8e483af510b4a3).
  - See also [Julia で Chainer を動かすぞ💪](https://speakerdeck.com/regonn/julia-de-chainer-wodong-kasuzo?slide=2)
