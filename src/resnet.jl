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
        #activ = activations[link.activ.__name__]
        activ = Flux.relu
    end
    bn = ch2bn(link.bn, activ)
    Chain([conv, bn]...)
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

struct MyRes
    layers
end

num = 50
const PyResNet = chainercv.links.model.resnet.ResNet
const resnet = PyResNet(num, pretrained_model = "imagenet")

function myres()
    dummyX = np.ones((1, 3, 224, 224), dtype = np.float32)
    resnet(dummyX)
    #resnet101=ResNet(101, pretrained_model="imagenet")
    #resnet152=ResNet(152, pretrained_model="imagenet")
    @show resnet.layer_names
    Chain(
        Conv2DBNActiv(resnet.conv1),
        MaxPool((3, 3), pad = (1, 1), stride = (2, 2)),
        ResBlock(resnet.res2),
        ResBlock(resnet.res3),
        ResBlock(resnet.res4),
        ResBlock(resnet.res5),
        x -> mean(x, dims = (1, 2)),
        x -> reshape(x, :, size(x, 4)),
        ch2dense(resnet.fc6),
        softmax,
    )
end


py"""
import chainercv
import numpy as np
img=chainercv.utils.read_image("pineapple.png",np.float32)
img=chainercv.transforms.resize(img,(224,224))
_imagenet_mean = np.array(
            [123.15163084, 115.90288257, 103.0626238],
            dtype=np.float32
        )[:, np.newaxis, np.newaxis]
img=img-_imagenet_mean
img=np.expand_dims(img,axis=0)
"""
r = myres()
Flux.testmode!(r, true)
img=reversedims(py"img")
@show size(img)
ret = r(img)
@show argmax(ret), 100ret[argmax(ret)]
