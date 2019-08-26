using Flux
using Statistics
using Gomah: chainercv

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
    c = Chain([conv, bn]...)
    Flux.testmode!(c, true)
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
    function ResNet(num::Int)
        PyResNet = chainercv.links.model.resnet.ResNet
        pyresnet = PyResNet(num, pretrained_model = "imagenet")
        #dummyX = np.ones((1, 3, 224, 224), dtype = np.float32)
        #resnet(dummyX)
        #resnet101=ResNet(101, pretrained_model="imagenet")
        #resnet152=ResNet(152, pretrained_model="imagenet")
        @show pyresnet.layer_names
        layers = [
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
        new(layers, pyresnet.layer_names)
    end
end

function (res::ResNet)(x)
    h = x
    d = Dict()
    for (name, lay) in zip(res.layer_names, res.layers)
        h = lay(h)
        d[name] = h
    end
    return h, d
end
