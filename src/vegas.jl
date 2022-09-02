using Random
using Distributions
using QuasiMonteCarlo
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
- Estimate for the integral 
- Standard deviation
- χ^2 / (numiter - 1): should be less than 1 
otherwise integral estimate should not be trusted. 

References:
-----------
- Lepage, G. Peter. "A new algorithm for adaptive 
multidimensional integration." Journal of 
Computational Physics 27.2 (1978): 192-203.
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
               batch = false)

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

    while iter <= maxiter

        # Sample `ncalls` points from this grid
		# Uniform sampling in `y` space
		ymat = QuasiMonteCarlo.sample(ncalls, lb, ub, QuasiMonteCarlo.UniformSample())	

		from_y_to_i(y) = floor(Int,nbins*y) + 1
		delta(y) = y*nbins + 1 - from_y_to_i(y)
		from_y_to_x(y,dim) = x[from_y_to_i(y),dim] + delx[from_y_to_i(y),dim]*delta(y)
		J(y,dim) = nbins*delx[from_y_to_i(y),dim]

		xmat = similar(ymat)
		for dim = 1:ndim
			xmat[dim,:] .= map(y -> from_y_to_x(y, dim), ymat[dim,:])
		end
		Js = zeros(eltype(ymat), size(ymat, 2))
		for i = 1:size(ymat,2)
			Js[i] = prod(J(ymat[dim, i],dim) for dim = 1:ndim)
		end
		imat = map(from_y_to_i, ymat)

        # Estimate integral. 
		# Get all fevals in one shot
		if batch
			fevals = func(collect(eachcol(xmat)))
			@assert length(fevals) == ncalls 
		else
			fevals = map(func, eachcol(xmat))
		end
		Jsf = Js .* fevals

		integral_mc = sum(Jsf) / ncalls
		variance_mc = (sum(Jsf.^2)/ncalls - integral_mc^2) / (ncalls - 1)

        nevals += ncalls
        push!(integrals, integral_mc)
        push!(sigma_squares, variance_mc)

		# Now the algorithm tries to make average J^2f^2 the same in every interval
		# Do this for every dimension
		d = calculate_d(Jsf, imat, nbins)
	
        # Update grid to reflect sub-inc dist
        x, delx = update_grid(x, delx, d)

        oldItot = Itot
        
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
        # M += Minc

    end
    chi_squared = sum(((integrals .- Itot).^2) ./ sigma_squares)


    Itot, sd, chi_squared/(iter-1)
end


"""
Calculate d matrix.
"""
function calculate_d(Jsf, imat, nbins)

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
		dreg[:,dim] .= ((1 .- dreg[:,dim]) ./ log.(1 ./ dreg[:,dim])).^1.5
	end
	
	dreg
end

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
	newdelx[1,:] .= newx[2,:]
	for i = 2:nbins
		newdelx[i,:] .= newx[i+1,:] .- newx[i,:]
	end
	newx, newdelx
end
