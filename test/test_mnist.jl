using Gomah

@testset "train" begin
    train()
end

@testset "predict" begin
    predict()
    # after training remove generated things
    rm("result", recursive=true)
end
