using OrdinaryDiffEq, DiffEqMonteCarlo, Distributions, Test

include("koopman.jl")

function f(du,u,p,t)
  du[1] = dx = p[1]*u[1] - u[1]*u[2]
  du[2] = dy = -3*u[2] + u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]
prob = ODEProblem(f,u0,tspan,p)
sol = solve(remake(prob,u0=u0),Tsit5())
cost(sol) = sum(max(x[1]-12,0) for x in sol.u)
u0s = [Uniform(0.25,5.5),Uniform(0.25,5.5)]
ps  = [Uniform(0.5,2.0)]
@time c1, _ = koopman_cost(u0s, ps, cost, prob, Tsit5();saveat=0.1, use_vegas = true)
@time c2 = montecarlo_cost(u0s, ps, cost, prob, Tsit5(); num_monte = 100000, saveat = 0.1)
@show c1, c2

#=function f2(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(f2,u0,tspan,p)
cost(sol) = sum(max(x[1]-6,0) for x in sol.u)
u0s = [Uniform(0.25,5.5),Uniform(0.25,5.5)]
ps  = [Uniform(0.5,2.0), Uniform(0.5, 1.5), Uniform(2.5, 3.5), Uniform(0.5, 1.5)]
@time c1, _ = koopman_cost(u0s, ps, cost, prob, Tsit5();saveat=0.1, use_vegas = true)
@time c2 = montecarlo_cost(u0s, ps, cost, prob, Tsit5(); num_monte = 100000, saveat = 0.1)
@show c1, c2
=#
