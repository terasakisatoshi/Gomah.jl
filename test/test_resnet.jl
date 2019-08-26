using Gomah
using Gomah: chainercv, reversedims, np, @pywith
using Flux
using Statistics
using Test

const PyResNet = chainercv.links.model.resnet.ResNet
const resnet = PyResNet(50, pretrained_model = "imagenet")

@testset "conv1.conv" begin
    dummyX = 128 * rand(Float32, 1, 3, 10, 10)
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
    dummyX = rand(Float32,(1, 64,10,10))
    dummyX = zeros(Float32,(1,64,10,10))
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


@testset "bn parameter test" begin
    dummyX = rand(Float32,(1, 64,10,10))
    dummyX = zeros(Float32,(1,64,10,10))
    pybn = resnet.conv1.bn
    @pywith chainer.using_config("train", false) begin
        chret = reversedims(pybn(dummyX).array)
        flbn = ch2bn(pybn)
        Flux.testmode!(flbn, true)
        flret = flbn(reversedims(dummyX))
        
        β = flbn.β 
        γ = flbn.γ 
        μ = flbn.μ 
        σ² = flbn.σ²
        ϵ = flbn.ϵ
        @test isapprox(β, pybn.beta.array)
        @test isapprox(γ, pybn.gamma.array)
        @test isapprox(μ, pybn.avg_mean)
        @test isapprox(σ², pybn.avg_var)
        @test isapprox(ϵ, pybn.eps)
    end
end



