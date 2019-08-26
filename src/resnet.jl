using Gomah
using Gomah: L, np, reversedims
using PyCall
using Flux
using Statistics

chainercv = pyimport("chainercv")

activations = Dict("relu" => Flux.relu)

function Conv2DBNActiv(link)
    conv = ch2conv(link.conv)
    if link.activ == nothing
        activ = Flux.identity
    else
        activ = activations[link.activ.__name__]
        #activ = Flux.relu
    end
    bn = ch2bn(link.bn, activ)
    c=Chain([conv, bn]...)
    Flux.testmode!(c,true)
    return c
end

struct BottleNeckA
    layers
    residual_conv
    function BottleNeckA(link::PyObject)
        layers = Conv2DBNActiv.([link.conv1, link.conv2, link.conv3])
        residual_conv = Conv2DBNActiv(link.residual_conv)
        new(Chain(layers...), residual_conv)
    end
end

Flux.@treelike BottleNeckA

function (bottleneck::BottleNeckA)(x)
    h1 = bottleneck.layers(x)
    h2 = bottleneck.residual_conv(x)
    y = Flux.relu.(h1 .+ h2)
    return y
end


struct BottleNeckB
    chain
    function BottleNeckB(link::PyObject)
        layers = [link.conv1, link.conv2, link.conv3]
        new(Chain(Conv2DBNActiv.(layers)...))
    end
end

Flux.@treelike BottleNeckB

function (bottleneckB::BottleNeckB)(x)
    h = bottleneckB.chain(x)
    Flux.relu.(h .+ x)
end

function ResBlock(link)
    layers = Any[]
    push!(layers, BottleNeckA(link.a))
    for name in link.layer_names[2:end]
        chlay = py"getattr"(link, name)
        push!(layers, BottleNeckB(chlay))
    end
    Chain(layers...)
end

struct ResNet
    layers
    layer_names
end

function ResNet(num::Int)
    PyResNet = chainercv.links.model.resnet.ResNet
    pyresnet = PyResNet(num, pretrained_model = "imagenet")
    #dummyX = np.ones((1, 3, 224, 224), dtype = np.float32)
    #resnet(dummyX)
    #resnet101=ResNet(101, pretrained_model="imagenet")
    #resnet152=ResNet(152, pretrained_model="imagenet")
    @show pyresnet.layer_names
    layers=[
        Conv2DBNActiv(pyresnet.conv1),
        MaxPool((3, 3), pad = (1, 1), stride = (2, 2)),
        ResBlock(pyresnet.res2),
        ResBlock(pyresnet.res3),
        ResBlock(pyresnet.res4),
        ResBlock(pyresnet.res5),
        x -> mean(x, dims = (1, 2)),
        x -> reshape(x, :, size(x, 4)),
        ch2dense(pyresnet.fc6),
        softmax,
    ]
    ResNet(layers, pyresnet.layer_names)
end

function (res::ResNet)(x)
    h = x
    d=Dict()
    for (name, lay) in zip(res.layer_names, res.layers)
        h = lay(h)
        d[name] = h
    end
    return h,d
end

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
    idx=np.argmax(result)
    print(idx)
    print(result[idx])
"""

num = 50
r = ResNet(num)
Flux.testmode!(r, true)
img=reversedims(py"img");
@show size(img)
ret,d = r(img)
@show argmax(ret), 100ret[argmax(ret)]

pyret0=reversedims(py"pyret[0].array");
using Test
@assert size(pyret0)==size(d["conv1"])

pyret0[:,:,1,1]
d["conv1"][:,:,1,1]
