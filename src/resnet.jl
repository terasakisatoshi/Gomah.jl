using Gomah
using Gomah: L, np
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
        layers=Conv2DBNActiv.([link.conv1,link.conv2,link.conv3])
        residual_conv = Conv2DBNActiv(link.residual_conv)
        new(Chain(layers...),residual_conv)
    end
end

Flux.@treelike BottleNeckA

function (bottleneck::BottleNeckA)(x)
    h1 = bottleneck.layers(x)
    h2 = bottleneck.residual_conv(x)
    y =  Flux.relu.(h1 .+ h2)
    return y
end


struct BottleNeckB
    chain
    function BottleNeckB(link::PyObject)
        layers=[link.conv1,link.conv2,link.conv3]
        new(Chain(Conv2DBNActiv.(layers)...))
    end
end

Flux.@treelike BottleNeckB

function (bottleneckB::BottleNeckB)(x)
    h=bottleneckB.chain(x)
    Flux.relu.(h .+ x)
end

function ResBlock(link)
    layers = Any[]
    push!(layers, BottleNeckA(link.a))
    for name in link.layer_names[2:end]
        chlay=py"getattr"(link, name)
        push!(layers, BottleNeckB(chlay))
    end
    Chain(layers...)
end

struct MyRes
    layers
end

num=50
const PyResNet = chainercv.links.model.resnet.ResNet
const resnet = PyResNet(num, pretrained_model = "imagenet")

function myres()
    dummyX=np.ones((1,3,224,224),dtype=np.float32)
    resnet(dummyX)
    #resnet101=ResNet(101, pretrained_model="imagenet")
    #resnet152=ResNet(152, pretrained_model="imagenet")
    @show resnet.layer_names
    Chain(
        Conv2DBNActiv(resnet.conv1),
        MaxPool((3,3), pad = (1,1), stride = (2,2)),
        ResBlock(resnet.res2),
        ResBlock(resnet.res3),
        ResBlock(resnet.res4),
        ResBlock(resnet.res5),
        x->mean(x,dims=(1,2)),
        x -> reshape(x, :, size(x,4)),
        ch2dense(resnet.fc6),
        softmax,
    )
end


r=myres()
Flux.testmode!(r,true)
ret=r(rand(Float32,224,224,3,1))
size(ret)