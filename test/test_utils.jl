using Test

function reversedims(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    permutedims(a, N:-1:1)
end

@testset "test reversedims" begin
    sz=(4,3,2,1)
    dummyX = rand(sz...)
    size(reversedims(dummyX))
    ret=reversedims(dummyX)
    @test reverse(size(dummyX)) == size(ret)
end