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

#=
 initialize pyobject to avoid error when define Python Class 
using @pydef mutable struct ...
=# 

function debug()
    batchsize = 128
    epochs = 10
    train_set, _ = get_mnist()
    N=pybuiltin(:len)(train_set)
    train_set, val_set = chainer.datasets.split_dataset(
            train_set,
            convert(Int, 0.9*N),
            Random.shuffle(collect(0:N-1))
        )
    train_iter = SerialIterator(train_set, batchsize)
    test_iter = SerialIterator(val_set, batchsize, repeat=false, shuffle=false)
    model = get_model()
end

include("mnist.jl")

end # module
