# MonteCarloIntegration.jl

This package provides multidimensional integration algorithms based on Monte Carlo
methods. The biggest advantage of using Monte Carlo methods is that their
convergence rate is **independent of the dimension of the integral**.

Currently, this package provides a routine called VEGAS:

```julia
vegas(f, st, en, kwargs...)
```

VEGAS is a Monte Carlo algorithm for multidimensional integration based on
adaptive importance sampling. It divides each dimension into bins and adaptively
adjusts bin widths so points are sampled from the region where the function has
the highest magnitude.

## Installation

To install MonteCarloIntegration.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("MonteCarloIntegration")
```

## Example

Integrate a Gaussian over a two-dimensional box:

```@example vegas
using MonteCarloIntegration
res = vegas(x -> exp(-sum(abs2, x)), [0.0, 0.0], [1.0, 1.0])
res.integral_estimate, res.standard_deviation
```

## API

```@docs
vegas
```
