using Random
using Distributions
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
"""
function vegas(func, 
               lb = [0.,0.], 
               ub = [1.,1.];
               maxiter = 100, 
               nbins = 1000, # from paper
               ncalls = 1000,
               rtol = 1e-4, 
               atol = 1e-4,
               debug = false, 
               batch = false)

    @assert length(lb) == length(ub)

    ndim = length(lb)

    # Start out with uniform grid
    grid = zeros(nbins, ndim)
    for d = 1:ndim
        grid[:,d] .= fill((ub[d] - lb[d])/nbins, nbins)
    end
    cgrid = cumsum(grid, dims = 1)

    # Initialize all cumulative variables 
    # for integral and sd estimation 
    nevals = 0
    Itot = 0.
    sd = 0.
    integrals = Float64[]
    sigma_squares = Float64[]
    iter = 1

    while iter <= maxiter


        # Sample `M` points from this grid
        pts, bpts = generate_pts(grid, cgrid, ncalls, lb, ub)
        

        # Estimate integral. Note: S stands for J*f, S2 is J^2f^2
		I_mc, avg_S2, fevals = evaluate_at_samples(func, 
                                            pts, 
                                            bpts,
                                            ncalls,
                                            nbins, 
                                            grid, 
                                            batch
                                           )


        σ² = (avg_S2 - I_mc^2) / (ncalls - 1) + eps() # When σ² = 0
        nevals += ncalls
        push!(integrals, I_mc)
        push!(sigma_squares, σ²)

		# Now the algorithm tries to make average J^2f^2 the same in every interval
		# Do this for every dimension
		d = calculate_d(fevals, bpts, grid)
	
        # Estimate sub-increments distribution
        m = calculate_m_dist(fevals, 
                             bpts, 
                             grid, 
                             ndim
                            )

        # Update grid to reflect sub-inc dist
        update_grid!(grid, cgrid, nbins, ncalls, m)

        # Update grid and generate new points
        cumsum!(cgrid, grid, dims=1)

        oldItot = Itot
        
        # Calculate integral and s.d upto this point
        Itot = sum((integrals.^3) ./ sigma_squares) / 
                sum((integrals.^2) ./ sigma_squares)


        sd = Itot * sum((integrals.^2) ./ sigma_squares)^(-0.5)
        
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
    χ² = sum(((integrals .- Itot).^2) ./ sigma_squares)
    @show nevals


    Itot, sd, χ²/(iter-1)
end

function evaluate_at_samples(f, pts, bpts, ncalls, nbins, grid, batch)

    S = 0.
    S² = 0.
    dim = size(pts, 2)

    # Get all fevals in one shot
    if batch
        fevals = f(pts)
        @assert length(fevals) == ncalls 
	else
		fevals = map(f, eachrow(p))
	end

    @inbounds for i = 1:ncalls

		fp = fevals[i]
        # Get probability of that particular point
        prob = 1.
        for d = 1:dim
            prob *= (1/(nbins*grid[bpts[i,d],d]))
        end

        S += (fp / prob)
        S² += (fp / prob)^2

    end

    S/ncalls, S²/ncalls, fevals
end


"""
pts, bpts = generate_pts(grid, cumgrid, M)

Generate `M` points from `grid` which probabilities 
inversely proportional to grid spacings 
"""
function generate_pts(grid, cgrid, ncalls, lb, ub)

    # Get bins and dimension
    dim = size(cgrid, 2)
    nbins = size(cgrid, 1)

    # Each dimension needs ncalls points
    pts = zeros(ncalls, dim)
    bpts = zeros(Int, ncalls, dim)

    for d = 1:dim

        # Remember which bins they come from 
        b = rand(1:nbins, ncalls)

        bpts[:,d] .= b

        idx = 1
        for (i,bin) in enumerate(b)
            if bin == 1
                pts[i,d] = lb[d] + rand(Uniform(0, cgrid[1,d]))
            else
                st = cgrid[bin-1,d]
                en = cgrid[bin,d]
                if st == en
                    continue
                end
                pts[i,d] = lb[d] + rand(Uniform(cgrid[bin-1,d], cgrid[bin,d]))
            end
        end
    end

   pts, bpts
end

"""
Calculate d matrix.
"""
function calculate_d(fevals, bpts, grid)

    ncalls = size(bpts, 1) 
	nbins, ndim = size(grid)
    d = zeros(nbins, ndim)
    probs = zeros(size(grid)...)
	interval_length = map(sum, eachcol(grid))

	# Start with uniform probability everywhere
	# and then weight by length of grid
	probmap = fill(1/nbins, nbins, ndims)
	for i = 1:ndims
		interval = interval_length[i]
		for j = 1:nbins
			probmap[j,i] /= (grid[j,i] / interval)
		end
	end
    
	f2 = fevals .^ 2
    for i = 1:dim 
		for c = 1:ncalls
			bin_i = bpts[c,i]
			d[bin_i,i] += f2 ./ (probmap[bin_i, i])^2
		end
    end
    
	# Regularize and smooth
	dreg = copy(d)
	for i = 1:ndim
		sumd = sum(d[:,i])
		dreg[1,i] = (7d[1] + d[2])/8 
		for j = 2:nbins-1
			dreg[j,i] = (d[j-1] + 6d[j] + d[j+1]) / 8
		end
		dreg[end,i] = (d[end-1,i]+ 7d[end]) / 8
		dreg ./= sumd
	end
	
	dreg = mapreduce(x -> ((1.-x)/log.(1./x)).^0.5,hcat, eachcol(dreg))
end

function update_grid(grid, cgrid, d)

	nbins, ndims = size(d)
	newcgrid = copy(cgrid)
	for dim = 1:ndims
		i = 0
		j = 0
		Sd = 0.
		delta_d = sum(d[:,dim]) / ncalls_per_bin

		while i < nbins
			while Sd < delta_d
				Sd += d[j+1]
				j+=1
			end
			Sd -= delta_d
			newcgrid[i+1] = cgrid[i+1] - ((Sd * grid[i+1])/d[j])
		end
	end
	newgrid = diff(newcgrid)
	newgrid, newcgrid
end


#="""
Calculate
"""
function calculate_m_dist(fevals, bpts, grid, dim)

 
    ncalls = size(bpts, 1) 
    nbins = size(grid, 1)
    m = zeros(nbins, dim)
    probs = zeros(size(grid)...)
    for d = 1:dim
        probs[:,d] .= (1 ./ grid[:,d]) ./ sum(1 ./ grid[:,d])
    end
    
    for d = 1:dim 
        for i = 1:ncalls
            f̄ = sqrt(fevals[i]^2 / 
                                (
                                 (prod(probs[bpts[i,d],:]) / 
                                 probs[bpts[i,d],d]
                                ) 
                                )
                    )
            m[bpts[i,d], d] += f̄ * grid[bpts[i,d],d]
        end
        m[:,d] .+= sum(m[:,d])
        m[:,d] .= m[:,d] ./ sum(m[:,d])
    end
    
    m
end

function update_grid!(grid, cgrid, nbins, ncalls, m)

    dims = size(grid, 2)

    for d = 1:dims
        gridcol = grid[:,d]
        mdim = m[:,d]

        # Count zeros in m 
        z = iszero.(mdim)
        c = sum(z)
        pos = findall(z)

        res = zeros(size(grid, 1))  
        for i = 1:nbins
            mdim[i] == 0 && continue
            res[i] = gridcol[i] / mdim[i]
        end

        # Calculate optm as sum / # non-zero m's
        optm = sum(mdim)/(nbins - c)

        for i = 1:nbins

            # If there are no m's, grid[i] = 0 
            if res[i] == 0 
                gridcol[i] = 0 
                continue
            end

            dist  = extract_from_bins!(mdim, optm, grid, res)
            gridcol[i] = dist

        end
        grid[:,d] .= gridcol
    end    
    
    grid
end=#

function extract_from_bins!(m, optm, grid, res)

    nbins = size(m, 1)
    dist = 0.
    collected = 0
    for k = 1:nbins

        # First, calculate how many required
        required = optm - collected

        (m[k] == 0) && continue

        # If bin has what's required, take everything
        if required <= m[k]
            #res = grid[k] / morig[k]
            dist += (res[k] * required)
            m[k] -= required
            collected += required
        end

        # Update requirements
        required = optm - collected

        # If more is required, take the entire bin
        if required >= m[k]
            #res = grid[k] / morig[k]
            dist += (res[k] * m[k])
            collected += m[k]
            m[k] = 0
        end

        # If collected everything, exit
        if collected >= optm
            break
        end
    end

    dist 
end
