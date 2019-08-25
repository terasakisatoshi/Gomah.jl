using Gomah
using Gomah: L, np, @pywith
using Flux
const DTYPE = Float32

using Test

function test_ch2dense()
    # get instance of Linear
    INSIZE = 10
    OUTSIZE = 20
    BSIZE = 1
    chlinear = L.Linear(in_size = INSIZE, out_size = OUTSIZE)
    dummyX = np.ones((BSIZE, INSIZE), dtype = np.float32)
    chret = reversedims(chlinear(dummyX).array)
    fldense = ch2dense(chlinear)
    flret = fldense(reversedims(dummyX))
    @test size(flret) == size(chret)
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
        ksize = KSIZE,
        pad = pad,
        stride = s,
        in_channels = INCH,
        out_channels = OUTCH
    )
    # pass dummy data to create computation graph
    dummyX = np.ones((BSIZE, INCH, inH, inW), dtype = np.float32)
    chret = reversedims(chconv(dummyX).array)
    flconv = ch2conv(chconv)
    flret = flconv(ones(DTYPE, inW, inH, INCH, BSIZE))
    @test size(flret) == size(chret)
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
        nobias=true,
    )
    # pass dummy data to create computation graph
    dummyX = np.ones((BSIZE, INCH, inH, inW), dtype = np.float32)
    chret = reversedims(chconv(dummyX).array)
    flconv = ch2conv(chconv)
    flret = flconv(ones(DTYPE, inW, inH, INCH, BSIZE))
    @test size(flret) == size(chret)
    @test all(isapprox.(flret, chret))
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
    dummyX = np.ones((BSIZE, INCH, inH, inW), dtype = np.float32)
    chret = reversedims(chdwconv(dummyX).array)
    fldwconv = ch2dwconv(chdwconv)
    flret = fldwconv(ones(DTYPE, inW, inH, INCH, BSIZE))
    @test size(flret) == size(chret)
    @test all(isapprox.(flret, chret))
end

function test_ch2bn()
    SIZE = 10
    BSIZE = 1
    dummyX = np.ones((BSIZE, SIZE))
    chbn = L.BatchNormalization(size = SIZE)
    @pywith chainer.using_config("train", false) begin
        chret = reversedims(chbn(dummyX).array)
        flbn = ch2bn(chbn)
        Flux.testmode!(flbn)
        flret = flbn(reversedims(dummyX))
        @test size(flret) == size(chret)
        @test all(isapprox.(flret, chret))
    end
end


@testset "converter" begin
    test_ch2dense()
    test_ch2conv()
    test_ch2conv_nobias()
    test_ch2dwconv()
    test_ch2bn()
end
