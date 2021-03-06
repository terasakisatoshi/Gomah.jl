using Flux
using PyCall
const DTYPE = Float32

function reversedims(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    permutedims(a, N:-1:1)
end

function ch2dense(link, σ = Flux.identity)
    W = link.W.array
    b = link.b.array
    Dense(W, b, σ)
end

function ch2conv(link, σ = Flux.identity)
    # get weight W and bias b
    W = reversedims(link.W.array)
    # flip kernel data
    W = W[end:-1:1, end:-1:1, :, :]
    if !isa(link.b, Nothing)
        b = reversedims(link.b.array)
    else
        b = zeros(DTYPE, size(W)[4])
    end
    pad = link.pad
    s = link.stride
    Conv(W, b, σ, pad = pad, stride = s)
end

function ch2dwconv(link, σ = Flux.identity)
    # get weight W and bias b
    W = permutedims(link.W.array, (4, 3, 1, 2))
    # flip kernel data
    W = W[end:-1:1, end:-1:1, :, :]
    if !isa(link.b, Nothing)
        b = reversedims(link.b.array)
    else
        b = zeros(DTYPE, size(W)[4])
    end
    pad = link.pad
    s = link.stride
    DepthwiseConv(W, b, σ, pad = pad, stride = s)
end

function ch2bn(link, λ = Flux.identity)
    β = link.beta.array
    γ = link.gamma.array
    ϵ = DTYPE(link.eps)
    μ = link.avg_mean
    σ² = link.avg_var
    sz = size(μ)[1]
    momentum = DTYPE(0.1) #unused parameter

    #set `false` so as to BatchNorm works as test mode
    activate = false
    bn = BatchNorm(λ, β, γ, μ, σ², ϵ, momentum, false)
    return bn
end
