using Gomah
using Gomah: L, np, @pywith, chainercv
using Flux
using Statistics

const DTYPE = Float32

using Test

function test_ch2dense()
    # get instance of Linear
    INSIZE = 10
    OUTSIZE = 20
    BSIZE = 1
    chlinear = L.Linear(in_size = INSIZE, out_size = OUTSIZE)
    dummyX = 128 * np.ones((BSIZE, INSIZE), dtype = np.float32)
    dtype = chret = reversedims(chlinear(dummyX).array)
    fldense = ch2dense(chlinear)
    flret = fldense(reversedims(dummyX))
    @test size(flret) == size(chret)
    @show maximum(abs.(flret .- chret))
    @show typeof.((fldense.W, fldense.b))
    @test all(isapprox.(flret, chret))
end

function test_ch2conv()
    # get instance of Convolution2D
    INCH = 3
    OUTCH = 8
    inH = 30
    inW = 30
    BSIZE = 1
    KSIZE = 3
    pad = 1
    s = 1
    chconv = Gomah.L.Convolution2D(
        in_channels = INCH,
        out_channels = OUTCH,
        ksize = KSIZE,
        pad = pad,
        stride = s,
    )
    # pass dummy data to create computation graph
    dummyX = np.ones((BSIZE, INCH, inH, inW), dtype = np.float32)
    chret = reversedims(chconv(dummyX).array)
    flconv = ch2conv(chconv)
    flret = flconv(ones(DTYPE, inW, inH, INCH, BSIZE))
    @test size(flret) == size(chret)
    @show maximum(abs.(flret .- chret))
    @show typeof.((flconv.weight, flconv.bias))
    @test all(isapprox.(flret, chret))
end

function test_ch2conv_nobias()
    # get instance of Convolution2D
    INCH = 3
    OUTCH = 8
    inH = 30
    inW = 30
    BSIZE = 1
    KSIZE = 3
    pad = 1
    s = 1
    chconv = Gomah.L.Convolution2D(
        ksize = KSIZE,
        pad = pad,
        stride = s,
        in_channels = INCH,
        out_channels = OUTCH,
        nobias = true,
    )
    # pass dummy data to create computation graph
    dummyX = 128 * np.ones((BSIZE, INCH, inH, inW), dtype = np.float32)
    chret = reversedims(chconv(dummyX).array)
    flconv = ch2conv(chconv)
    flret = flconv(reversedims(dummyX))
    @test size(flret) == size(chret)
    @show maximum(abs.(flret .- chret))
    @test all(isapprox.(abs.(flret - chret), 0, atol = 1e-4))
end

function test_resnetconv1ch2conv()
    # get instance of Convolution2D
    INCH = 2
    OUTCH = 3
    inH = 7
    inW = 7
    BSIZE = 1
    KSIZE = 7
    pad = 3
    s = 2
    chconv = Gomah.L.Convolution2D(
        ksize = KSIZE,
        pad = pad,
        stride = s,
        in_channels = INCH,
        out_channels = OUTCH,
        nobias = true,
    )
    # pass dummy data to create computation graph
    dummyX = 128 * np.ones((BSIZE, INCH, inH, inW), dtype = np.float32)
    chret = reversedims(chconv(dummyX).array)
    flconv = ch2conv(chconv)
    flret = flconv(reversedims(dummyX))
    @test size(flret) == size(chret)
    @show maximum(abs.(flret .- chret))
    @test all(isapprox.(abs.(flret - chret), 0, atol = 1e-4))
end

function test_ch2dwconv()
    # get instance of DepthwiseConvolution2D
    INCH = 10
    MULTIPLIER = 2
    inH = 10
    inW = 10
    BSIZE = 2
    KSIZE = 3
    pad = 1
    s = 1
    chdwconv = Gomah.L.DepthwiseConvolution2D(
        in_channels = INCH,
        channel_multiplier = MULTIPLIER,
        ksize = KSIZE,
        pad = pad,
        stride = s
    )
    # pass dummy data to create computation graph
    dummyX = 128 * np.ones((BSIZE, INCH, inH, inW), dtype = np.float32)
    chret = reversedims(chdwconv(dummyX).array)
    fldwconv = ch2dwconv(chdwconv)
    flret = fldwconv(reversedims(dummyX))
    @test size(flret) == size(chret)
    @show maximum(abs.(flret .- chret))
    @show typeof.((fldwconv.weight, fldwconv.bias))
    @test all(isapprox.(abs.(flret - chret), 0, atol = 1e-4))
end

function test_ch2bn()
    SIZE = 10
    BSIZE = 1
    dummyX = 128 * np.ones((BSIZE, SIZE), dtype = np.float32)
    chbn = L.BatchNormalization(size = SIZE)
    @pywith chainer.using_config("train", false) begin
        chret = reversedims(chbn(dummyX).array)
        flbn = ch2bn(chbn)
        Flux.testmode!(flbn, true)
        flret = flbn(reversedims(dummyX))
        @test size(flret) == size(chret)
        @show maximum(abs.(flret .- chret))
        @test all(isapprox.(abs.(flret - chret), 0, atol = 1e-4))
    end
end


@testset "bn parameter test" begin
    PyResNet = chainercv.links.model.resnet.ResNet
    num = 50
    pyresnet = PyResNet(num, pretrained_model = "imagenet")
    dummyX = rand(Float32, (1, 64, 10, 10))
    dummyX = zeros(Float32, (1, 64, 10, 10))
    pybn = pyresnet.conv1.bn
    flbn = ch2bn(pybn)
    Flux.testmode!(flbn, true)
    @pywith chainer.using_config("train", false) begin
        chret = reversedims(pybn(dummyX).array)
        flret = flbn(reversedims(dummyX))

        β = flbn.β
        γ = flbn.γ
        μ = flbn.μ
        σ² = flbn.σ²
        ϵ = flbn.ϵ
        @show typeof.((β, γ, μ, σ², ϵ))
        @test isapprox(β, pybn.beta.array)
        @test isapprox(γ, pybn.gamma.array)
        @test isapprox(μ, pybn.avg_mean)
        @test isapprox(σ², pybn.avg_var)
        @test isapprox(ϵ, pybn.eps)
    end
end

@testset "converter" begin
    test_ch2dense()
    test_ch2conv()
    test_ch2conv_nobias()
    test_resnetconv1ch2conv()
    test_ch2dwconv()
    test_ch2bn()
end
