module MonteCarloIntegration

abstract type MonteCarloIntegrationResult end
include("vegas.jl")

export vegas

end # module
