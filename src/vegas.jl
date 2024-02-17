using Random
using Distributions
using QuasiMonteCarlo

abstract type MonteCarloIntegrationResult end

struct VEGASResult{T1,T2,T3,T4,T5}
	integral_estimate::T1
	standard_deviation::T2
	chi_squared_average::T3
	adaptive_grid::T4
	grid_spacing::T5
end
"""
    vegas(f, st, en, kwargs...)

VEGAS is a Monte Carlo algorithm for 
multidimensional integration based on 
adaptive importance sampling. It divides
each dimension into bins and adaptively adjusts
bin widths so the points sampled from the
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
Defaults to 1000. 
- ncalls: Number of function calls per iteration. 
Defaults to 10000.
- maxiter: Maximum number of iterations. 
Defaults to 10.
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
A VEGASResult object with the following fields:
- Estimate for the integral 
- Standard deviation
- χ^2 / (numiter - 1): should be less than 1 
otherwise integral estimate should not be trusted. 
- final adapted map

References:
-----------
- Lepage, G. Peter. "A new algorithm for adaptive 
multidimensional integration." Journal of 
Computational Physics 27.2 (1978): 192-203.
- Lepage, G. Peter. "Adaptive multidimensional 
integration: VEGAS enhanced." Journal of 
Computational Physics 439 (2021): 110386.
"""
function vegas(func, 
               lb = [0.,0.], 
               ub = [1.,1.];
               maxiter = 10, 
               nbins = 1000, # from paper
               ncalls = 10000,
               rtol = 1e-4, 
               atol = 1e-4,
               debug = false, 
               batch = false,
			   alpha = 1.5)

    @assert length(lb) == length(ub)

    ndim = length(lb)

    # Start out with uniform grid
	x = zeros(nbins+1, ndim)
	for i = 1:(nbins+1)
		x[i,:] .= lb + ((ub .- lb) ./ nbins)*(i-1)
	end
	delx = zeros(nbins, ndim)
	for dim = 1:ndim
		delx[:,dim] .= ((ub[dim] - lb[dim]) / nbins)
	end



    # Initialize all cumulative variables 
    # for integral and sd estimation 
    nevals = 0
    Itot = 0.
    sd = 0.
    integrals = Float64[]
    sigma_squares = Float64[]
    iter = 1

	ymat = zeros(ndim, ncalls)
	xmat = similar(ymat)

	from_y_to_i(y) = floor(Int,nbins*y) + 1
	delta(y) = y*nbins + 1 - from_y_to_i(y)
	from_y_to_x(y,dim) = x[from_y_to_i(y),dim] + delx[from_y_to_i(y),dim]*delta(y)
	J(y,dim) = nbins*delx[from_y_to_i(y),dim]

	imat = map(from_y_to_i, ymat)
	Js = zeros(eltype(ymat), size(ymat, 2))
	Jsf = copy(Js)
    while iter <= maxiter

        # Sample `ncalls` points from this grid
		# Uniform sampling in `y` space
		rand!(ymat) #QuasiMonteCarlo.sample(ncalls, zeros(ndim), ones(ndim), Uniform())



		for dim = 1:ndim
			for j in 1:ncalls
				xmat[dim,j] = from_y_to_x(ymat[dim,j], dim)
			end
		end
		for i = 1:size(ymat,2)
			Js[i] = prod(J(ymat[dim, i],dim) for dim = 1:ndim)
		end

		imat .= from_y_to_i.(ymat)

        # Estimate integral. 
		# Get all fevals in one shot
		if batch
			fevals = func(collect(eachcol(xmat)))
			@assert length(fevals) == ncalls 
		else
			fevals = map(func, eachcol(xmat))
		end
		Jsf .= Js .* fevals

		integral_mc = sum(Jsf) / ncalls
		variance_mc = (sum(x-> x^2, Jsf)/ncalls - integral_mc^2) / (ncalls - 1) + eps()

        nevals += ncalls
        push!(integrals, integral_mc)
        push!(sigma_squares, variance_mc)

		# Now the algorithm tries to make average J^2f^2 
		# the same in every interval. Do this for every dimension
		d = calculate_d(Jsf, imat, nbins, alpha)
	
        # Update grid to make d equal in all intervals
        xnew, delxnew = update_grid(x, delx, d)
		x .= xnew
		delx .= delxnew

        # Calculate integral and s.d upto this point
		Itot = sum(integrals ./ sigma_squares) / sum(1 ./ sigma_squares)
		sd = 1/sqrt(sum(1 ./ sigma_squares))
        
        if debug
            println("Iteration $iter, abs(sd/Itot) = $(abs(sd/Itot))")
        end

        if abs(sd/Itot) < rtol && abs(sd) < atol
            println("Converged in $nevals evaluations")
            break
        end

        iter += 1

    end
    chi_squared = sum(((integrals .- Itot).^2) ./ sigma_squares)
    
	VEGASResult(Itot, sd, chi_squared/(iter-1), x, delx)
end


"""
	calculate_d(Jsf, imat, nbins) -> d

Calculate d matrix given J*f product and
locations of the sampled points.
"""
function calculate_d(Jsf, imat, nbins, alpha = 1.5)

	ndim = size(imat,1)
	d = zeros(nbins, ndim)
	ncalls = length(Jsf)
	ni = ncalls / nbins
	for i = 1:length(Jsf)
		for dim = 1:ndim
			d[imat[dim,i],dim] += (Jsf[i]^2 / ni)
		end
	end
    
	# Regularize and smooth
	dreg = copy(d)
	for dim = 1:ndim
		sumd = sum(d[:,dim])
		dreg[1,dim] = (7d[1,dim] + d[2,dim])/8 
		for j = 2:nbins-1
			dreg[j,dim] = (d[j-1,dim] + 6d[j,dim] + d[j+1,dim]) / 8
		end
		dreg[end,dim] = (d[end-1,dim]+ 7d[end,dim]) / 8
		dreg[:,dim] ./= sumd
	end

	for dim = 1:ndim
		dreg[:,dim] .= ((1 .- dreg[:,dim]) ./ log.(1 ./ dreg[:,dim])).^alpha
	end
	
	dreg
end

"""
	update_grid(x, delx, d) -> newx, newdelx

Function used to update the adaptive grid
given values of the d matrix. The intervals
are adjusted so that the `d` values are equal
in each interval.
"""
function update_grid(x, delx, d)

	nbins, ndims = size(d)
	newx = copy(x)
	for dim = 1:ndims
		newx[1,dim] = x[1,dim]
		newx[end,dim] = x[end,dim]
		i=1
		j=1
		Sd = 0.
		delta_d = sum(d[:,dim]) / nbins
		i+=1
		while i < (nbins+1) 
			while Sd < delta_d 
				Sd+=d[j, dim]
				j+=1
			end
			Sd -= delta_d
			newx[i,dim] = x[j,dim] - ((Sd * delx[j-1,dim])/d[j-1,dim])
			i+=1
		end
	end
	newdelx = copy(delx)
	for i = 1:nbins
		newdelx[i,:] .= newx[i+1,:] .- newx[i,:]
	end
	newx, newdelx
end

"""
	sample_from_adaptive_grid(res, ncalls)

Sample using a finetuned adaptive grid.  
"""
function sample_from_adaptive_grid(res, ncalls)

	x = res.adaptive_grid
	delx = res.grid_spacing

	nbins = size(delx, 1)
	ndim = size(x, 2)

	# Sample `ncalls` points from this grid
	# Uniform sampling in `y` space
	ymat = QuasiMonteCarlo.sample(ncalls, zeros(ndim), ones(ndim), Uniform())

	from_y_to_i(y) = floor(Int,nbins*y) + 1
	delta(y) = y*nbins + 1 - from_y_to_i(y)
	from_y_to_x(y,dim) = x[from_y_to_i(y),dim] + delx[from_y_to_i(y),dim]*delta(y)
	J(y,dim) = nbins*delx[from_y_to_i(y),dim]

	xmat = similar(ymat)
	for dim = 1:ndim
		xmat[dim,:] .= map(y -> from_y_to_x(y, dim), ymat[dim,:])
	end
	map(x -> tuple(x...), eachcol(xmat))
end

