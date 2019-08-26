module Gomah

using PyCall
using PyCall: @pywith
export @pywith

export chainer
# defined in mnist.jl
export train, predict
export debug, get_model
export reversedims
export ch2dense, ch2conv, ch2dwconv, ch2bn
export Conv2DBNActiv, BottleNeckA, BottleNeckB, ResBlock, ResNet


# chainer object
const chainer = PyNULL()
const chainercv = PyNULL()
const math = PyNULL()
const np = PyNULL()
const F = PyNULL()
const L = PyNULL()
const SerialIterator = PyNULL()
const training = PyNULL()
const extensions = PyNULL()
const optimizers = PyNULL()


# module initializer
function __init__()
    copy!(chainer, pyimport("chainer"))
    copy!(chainercv, pyimport("chainercv"))
    copy!(math, pyimport("math"))
    copy!(np, pyimport("numpy"))
    copy!(F, chainer.functions)
    copy!(L, chainer.links)
    copy!(SerialIterator, chainer.iterators.SerialIterator)
    copy!(training, chainer.training)
    copy!(extensions, training.extensions)
    copy!(optimizers, chainer.optimizers)
end


include("mnist.jl")
include("chainer2flux.jl")
include("resnet.jl")

end # module
