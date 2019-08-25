using Gomah
using Test


@testset "train" begin
    train()
end

@testset "predict" begin
    predict()
end

include("test_converter.jl")

