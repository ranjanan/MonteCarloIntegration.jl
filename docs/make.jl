using Documenter
using MonteCarloIntegration

makedocs(;
    modules = [MonteCarloIntegration],
    authors = "SciML Contributors",
    sitename = "MonteCarloIntegration.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://docs.sciml.ai/MonteCarloIntegration/stable/",
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/SciML/MonteCarloIntegration.jl.git",
)
