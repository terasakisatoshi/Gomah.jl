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
    idx=np.argmax(result)
    print(idx)
    print(result[idx])
"""

num = 50
myres = ResNet(num)
Flux.testmode!(r, true)
img = reversedims(py"img");
@show size(img)
ret, d = myres(img)
@show argmax(ret), 100ret[argmax(ret)]

pyret0 = reversedims(py"pyret[0].array");
using Test
@assert size(pyret0) == size(d["conv1"])

pyret0[:, :, 1, 1]
d["conv1"][:, :, 1, 1]
