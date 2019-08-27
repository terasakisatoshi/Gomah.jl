using Gomah
using Gomah: chainercv, reversedims, np, @pywith
using Flux
using Statistics
using Test

const PyResNet = chainercv.links.model.resnet.ResNet
const resnet = PyResNet(50, pretrained_model = "imagenet")

@testset "conv1.conv" begin
    dummyX = 128 * rand(Float32, 1, 3, 224, 224)
    pyconv1 = resnet.conv1.conv
    chret = reversedims(pyconv1(dummyX).array)
    flconv1 = ch2conv(pyconv1)
    Flux.testmode!(flconv1, true)
    flret = flconv1(reversedims(dummyX))

    @show pyconv1.ksize
    @show pyconv1.pad
    @show pyconv1.stride

    @test size(flret) == size(chret)
    @show mean(abs.(flret .- chret))
end

@testset "conv1.bn" begin
    dummyX = rand(Float32,(1, 64,224,224))
    pybn = resnet.conv1.bn
    @pywith chainer.using_config("train", false) begin
        chret = reversedims(pybn(dummyX).array)
        flbn = ch2bn(pybn)
        Flux.testmode!(flbn, true)
        flret = flbn(reversedims(dummyX))
        @show size(flret), size(chret)
        @test size(flret) == size(chret)
        @show mean(abs.(flret .- chret))
    end
end

@testset "Conv2DBNActiv" begin
    dummyX = rand(Float32, (1, 3, 224, 224)) 
    pyconv1 = resnet.conv1
    flconv1 = Conv2DBNActiv(pyconv1)
    Flux.testmode!(flconv1,true)
    @pywith chainer.using_config("train", false) begin
        chret = reversedims(pyconv1(dummyX).array)
        flret = flconv1(reversedims(dummyX))
        @test size(flret) == size(chret)
        @show maximum(abs.(flret-chret))
        @test all(isapprox.(flret,chret,atol=1e-6))
    end
end


@testset "BottleNeckA" begin
    dummyX = rand(Float32, (1,64, 56, 56))
    pyres2 = resnet.res2
    flres2 = ResBlock(pyres2)
    Flux.testmode!(flres2, true)
    @pywith chainer.using_config("train", false) begin
        chret = reversedims(pyres2(dummyX).array)
        flret = flres2(reversedims(dummyX))
        @test size(flret) == size(chret)
        @show maximum(abs.(flret-chret))
        @test all(isapprox.(flret, chret, atol=1e-4))
    end
end
