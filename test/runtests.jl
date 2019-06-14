using MonteCarloIntegration
using Test

@testset "Simple Monte Carlo Tests" begin
    for dim = 2:2
        @test isapprox(vegas(x -> 1, zeros(dim), ones(dim), nbins = 1)[1], 1., rtol = 1e-2)
        @test isapprox(vegas(x -> sum(x),zeros(dim), ones(dim), nbins = 1)[1], 1., rtol = 1e-2)
        @test isapprox(vegas(x -> sum(sin.(x)),zeros(dim), ones(dim), nbins = 1)[1], 0.919, rtol = 1e-2)
    end
end
@testset "Standard Integrands" begin
    for dim = 2:2
        @test isapprox(vegas(x -> 1, zeros(dim), ones(dim))[1], 1., rtol = 1e-2)
        @test isapprox(vegas(x -> sum(x), zeros(dim), ones(dim))[1], 1., rtol = 1e-2)
        @test isapprox(vegas(x -> sum(sin.(x)), zeros(dim), ones(dim))[1], 0.919, rtol = 1e-2)
    end
end
