using Flux
using PyCall
const DTYPE = Float32

function reversedims(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    permutedims(a, N:-1:1)
end

function ch2dense(link)
    W = link.W.array
    b = link.b.array
    Dense(W, b)
end

function ch2conv(link)
    # get weight W and bias b
    W = reversedims(link.W.array)
    # flip kernel data
    W = W[end:-1:1, end:-1:1, :, :]
    b = reversedims(link.b.array)
    pad = link.pad
    Conv(W, b, pad = pad)
end

function ch2dwconv(link)
    # get weight W and bias b
    W = permutedims(link.W.array, (4, 3, 1, 2))
    # flip kernel data
    W = W[end:-1:1, end:-1:1, :, :]
    b = reversedims(link.b.array)
    pad = link.pad
    DepthwiseConv(W, b, pad = pad)
end

function ch2batchnorm(link, sz)
    β = link.beta.array
    γ = link.gamma.array
    ϵ = link.eps
    μ = link.avg_mean
    σ² = link.avg_var
    bn = BatchNorm(sz)
    bn.β = β
    bn.γ = γ
    bn.ϵ = ϵ
    bn.σ² = σ²
    return bn
end

