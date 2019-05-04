module Gomah

using PyCall

export chainer, train, predict


function train()
    chainer = pyimport("chainer")
    mnist = chainer.datasets.mnist
    mnist.get_mnist()
end


function predict()
    nothing
end
#include("mnist.jl")

end # module
