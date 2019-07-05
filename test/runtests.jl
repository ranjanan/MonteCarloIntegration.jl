using MonteCarloIntegration
using HCubature
using Test

integrands = [
              x -> 1,
              x -> sum(x),
              x -> sum(sin.(x))
             ]

batch_f(f) = (pts) -> begin

    npts = size(pts, 1)
    ndims = size(pts, 2)
    fevals = zeros(npts)

    for i = 1:npts
        p = vec(pts[i,:])
        fevals[i] = f(p)
    end

    fevals
end
    

@testset "Standard Integrands" begin

    for f in integrands

        for dim = 1:5
            @info("Dimension = $dim")
            v, _ = vegas(f, zeros(dim), ones(dim))
            h, _ = hcubature(f, zeros(dim), ones(dim))
            @test isapprox(v, h, rtol = 1e-2)
        end
    end
end

@testset "Batched Integrands" begin

    for f in integrands
        
        f2 = batch_f(f)

        for dim = 1:5
            @info("Dimension = $dim")
            v, _ = vegas(f2, zeros(dim), ones(dim), batch = true)
            h, _ = hcubature(f, zeros(dim), ones(dim))
            @test isapprox(v, h, rtol = 1e-2)
        end
    end
end
              

#=@testset "Simple Monte Carlo Tests" begin
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
end=#
