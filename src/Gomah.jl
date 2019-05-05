module Gomah

using PyCall

export chainer
# defined in mnist.jl
export train, predict
export debug, get_model

# chainer object
const chainer = PyNULL()
const math = PyNULL()
const np = PyNULL()
const Chain = PyNULL()
const F = PyNULL()
const L = PyNULL()
const SerialIterator = PyNULL()
const training = PyNULL()
const extensions = PyNULL()
const optimizers = PyNULL()


# module initializer
function __init__()
    copy!(chainer, pyimport("chainer"))
    copy!(math, pyimport("math"))
    copy!(np, pyimport("numpy"))
    copy!(Chain, chainer.Chain)
    copy!(F, chainer.functions)
    copy!(L, chainer.links)
    copy!(SerialIterator, chainer.iterators.SerialIterator)
    copy!(training, chainer.training)
    copy!(extensions, training.extensions)
    copy!(optimizers, chainer.optimizers)
end


include("mnist.jl")

end # module
