using MonteCarloIntegration
using HCubature
using Test

integrands = [
              x -> 1,
              x -> sum(x),
              x -> sum(sin.(x))
             ]

batch_f(f) = (pts) -> begin
	map(f, pts)
end
    

@testset "Standard Integrands" begin

    for f in integrands

        for dim = 1:5
            @info("Dimension = $dim")
            v = vegas(f, zeros(dim), ones(dim)).integral_estimate
            h, _ = hcubature(f, zeros(dim), ones(dim))
            @test isapprox(v, h, rtol = 1e-2)
        end
    end
end

# Test from HCubature.jl
f(x) = sin(x[1] + 3*sin(2*x[2] + 4*sin(3*x[3]))) 
v = vegas(f, zeros(3), fill(3.0, 3), maxiter = 200).integral_estimate
@test v ≈ -4.78802790509727 rtol=1e-2

@testset "Batched Integrands" begin

    for f in integrands
        
        f2 = batch_f(f)

        for dim = 1:5
            @info("Dimension = $dim")
            v = vegas(f2, zeros(dim), ones(dim), batch = true).integral_estimate
            h, _ = hcubature(f, zeros(dim), ones(dim))
            @test isapprox(v, h, rtol = 1e-2)
        end
    end
end

