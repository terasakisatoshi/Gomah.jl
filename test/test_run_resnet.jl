using Gomah
using Gomah: L, np, reversedims
using Flux
using PyCall

using Test
using BenchmarkTools

py"""
import chainer
import chainercv
import numpy as np
num = 50
PyResNet = chainercv.links.model.resnet.ResNet
resnet = PyResNet(num, pretrained_model = "imagenet")
img=chainercv.utils.read_image("pineapple.png",dtype=np.float32,alpha="ignore")
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

@testset "regression" begin
    num = 50
    myres = ResNet(num)
    Flux.testmode!.(myres.layers)
    img = reversedims(py"img")
    @show size(img), typeof(img)
    ret, name2data = myres(img)
    for (i, name) in enumerate(myres.layer_names)
        pyr = reversedims(py"pyret[$i-1].array")
        flr = name2data[name]
        @show name, size(flr)
        @test size(pyr) == size(flr)
        @show maximum(abs.(pyr .- flr))
    end
    flidx = argmax(ret)
    flprob = 100ret[argmax(ret)]
    @show flidx, flprob
    @test Int(py"chidx") == flidx[1] - 1
    @show Float32(py"chprob") - flprob
end

@testset "benchmark" begin
    num = 50
    img = reversedims(py"img")
    myres = ResNet(num)
    chainmodel = Chain(myres.layers...)
    Flux.testmode!(chainmodel)
    @time chainmodel(img)
    @time chainmodel(img)
    @time chainmodel(img)
    @time chainmodel(img)
    @time chainmodel(img)
    @time chainmodel(img)
end
