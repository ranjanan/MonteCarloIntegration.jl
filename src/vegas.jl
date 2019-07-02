using Random
using Distributions
"""
    vegas(f, st, en, kwargs...)

VEGAS is a Monte Carlo algorithm for 
multidimensional integration based on 
adaptive importance sampling. It divides
each dimension into bins and adaptively adjusts
bin widths so the points sampled from the
region where the function has highest magnitude

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
Defaults to 10000.
- maxiter - Maximum number of iterations. 
Defaults to 100.
"""
function vegas(func, 
               a = [0.,0.], 
               b = [1.,1.];
               maxiter = 100, 
               nbins = 100, 
               ncalls = 1000,
               Minc = 500,
               K = 1000.,
               α = 1.5, 
               rtol = 1e-4)

    N = nbins
    M = ncalls

    @assert length(a) == length(b)

    ndim = length(a)

    # Random.seed!(0)

    # Start out with uniform grid
    # grid = fill(1/N, N, ndim)
    grid = zeros(N, ndim)
    for d = 1:ndim
        grid[:,d] .= fill((b[d] - a[d])/N, N)
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
        pts, bpts = generate_pts(grid, cgrid, M, a, b)
        

        # Estimate integral
        S, S², fevals = evaluate_at_samples(func, 
                                            pts, 
                                            bpts,
                                            M,
                                            N, 
                                            grid
                                           )


        σ² = (S² - S^2) / (M - 1) + eps() # When σ² = 0
        nevals += M
        push!(integrals, S)
        push!(sigma_squares, σ²)

        # Estimate sub-increments distribution
        m = calculate_m_dist(fevals, 
                             bpts, 
                             grid, 
                             K, 
                             α, 
                             ndim
                            )

        # Update grid to reflect sub-inc dist
        update_grid!(grid, cgrid, N, M, m)

        # Update grid and generate new points
        cumsum!(cgrid, grid, dims=1)

        oldItot = Itot
        
        # Calculate integral and s.d upto this point
        Itot = sum((integrals.^3) ./ sigma_squares) / 
                sum((integrals.^2) ./ sigma_squares)


        sd = Itot * sum((integrals.^2) ./ sigma_squares)^(-0.5)

        if abs(sd/Itot) < rtol 
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

function evaluate_at_samples(f, pts, bpts, M, N, grid)

    S = 0.
    S² = 0.
    fevals = zeros(M)
    dim = size(pts, 2)
    probs = zeros(size(grid)...)

    # Calculate probabilities corresponding
    # to each point in the grid
    for d = 1:dim
        probs[:,d] = (1 ./ grid[:,d]) ./ sum(1 ./ grid[:,d])
    end
    probs .+= eps()

    for i = 1:M

        
        # Extract point
        p = vec(pts[i,:])

        # Get probability of that particular point
        prob = 1.
        for d = 1:dim
            prob *= (1/(N*grid[bpts[i,d],d]))
        end

        # Eval function
        fp = f(p)
        fevals[i] = fp

        S += (fp / prob)
        S² += (fp / prob)^2

    end

    S/M, S²/M, fevals
end


"""
pts, bpts = generate_pts(grid, cumgrid, M)

Generate `M` points from `grid` which probabilities 
inversely proportional to grid spacings 
"""
function generate_pts(grid, cgrid, M, a, b)

    # Get bins and dimension
    dim = size(cgrid, 2)
    N = size(cgrid, 1)

    # Each dimension needs M points
    pts = zeros(M, dim)
    bpts = zeros(Int, M, dim)

    for d = 1:dim

        # Remember which bins they come from 
        b = rand(1:N, M)

        bpts[:,d] .= b

        idx = 1
        for (i,bin) in enumerate(b)
            if bin == 1
                pts[i,d] = a[d] + rand(Uniform(0, cgrid[1,d]))
            else
                st = cgrid[bin-1,d]
                en = cgrid[bin,d]
                if st == en
                    continue
                end
                pts[i,d] = a[d] + rand(Uniform(cgrid[bin-1,d], cgrid[bin,d]))
            end
        end
    end

   pts, bpts
end

function calculate_m_dist(fevals, bpts, grid, K, α, dim)
 
    M = size(bpts, 1) 
    N = size(grid, 1)
    m = zeros(N, dim)
    probs = zeros(size(grid)...)
    for d = 1:dim
        probs[:,d] .= (1 ./ grid[:,d]) ./ sum(1 ./ grid[:,d])
    end
    probs .+= eps()
    
    for d = 1:dim 
        for i = 1:M
            f̄ = sqrt(fevals[i]^2 / 
                                (
                                 (prod(probs[bpts[i,d],:]) / 
                                 probs[bpts[i,d],d]
                                ) 
                                )
                    )
            m[bpts[i,d], d] += f̄ * grid[bpts[i,d],d]
        end
        m[:,d] .= m[:,d] ./ sum(m[:,d])
        m[:,d] .= K .* m[:, d]
        m[:,d] .+= 1 # Ensure m ≠ 0
    end
    
    round.(m)
end

function update_grid!(grid, cgrid, N, M, m)

    dims = size(grid, 2)

    for d = 1:dims
        gridcol = grid[:,d]
        mdim = m[:,d]

        # Count zeros in m 
        z = iszero.(mdim)
        c = sum(z)
        pos = findall(z)

        res = zeros(size(grid, 1))  
        for i = 1:N
            mdim[i] == 0 && continue
            res[i] = gridcol[i] / mdim[i]
        end

        # Calculate optm as sum / # non-zero m's
        optm = sum(mdim)/(N - c)

        for i = 1:N

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
end

function extract_from_bins!(m, optm, grid, res)

    N = size(m, 1)
    dist = 0.
    collected = 0
    for k = 1:N

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
