# Monte Carlo Integration 

This package provides multidimensional integration 
algorithms based on monte carlo methods. The biggest
advantage of using monte carlo methods is that their
convergence rate is **independent of the dimension of
the integral**. 

Currently, this package only provides a routine 
called VEGAS: 

    vegas(f, st, en, kwargs...)

VEGAS is a Monte Carlo algorithm for 
multidimensional integration based on 
adaptive importance sampling. It divides
each dimension into bins and adaptively adjusts
bin widths so points are sampled from the
region where the function has highest magnitude. 

Arguments:
----------
- st: Array of starting values in each dimension. 
Defaults to zeros(2)
- end: Array of ending values in each dimension. 
Defaults to ones(2)

Kwargs:
------
- nbins: Number of bins in each dimension. 
Defaults to 100. 
- ncalls: Number of function calls per iteration. 
Defaults to 1000.
- maxiter: Maximum number of iterations. 
Defaults to 100.
- rtol: Relative tolerance required. 
Defaults to 1e-4.
- atol: Absolute tolerance required. 
Defaults to 1e-2.
- debug: Prints `abs(sd/I)` every 100 iterations. 
Defaults to false.
- batch: Whether `f` returns batches of function
evaluations. `f` is assumed to take one argument 
`pts`, an `ncalls × `ndims` matrix. Each row
is a unique point and returns an `ncalls` length
vector of function evals. This argument defaults
to false. 

Output:
------
- Estimate for the integral 
- Standard deviation
- χ^2 / (numiter - 1): should be less than 1 
otherwise integral estimate should not be trusted. 

References:
-----------
- Lepage, G. Peter. "A new algorithm for adaptive 
multidimensional integration." Journal of 
Computational Physics 27.2 (1978): 192-203.

### Batch interface

Most of the computation time in an integration
algorithm is usually spent in function evaluations. 
The batch inteface allows users to provide 
batches of function evaluations, instead of supplying
a function directly to be integrated. Users can now
evaluate a number of points in parallel. 

### Roadmap 
- Supporting vector valued functions
- Other integration algorithms
