using Random
using Distributions
"""
    vegas(f, kwargs...)

Monte Carlo integration with adaptive sampling
"""
function vegas(func; 
               ndim = 2, 
               maxiter = 5, 
               N = 2, 
               M = 2, 
               K = 1000.,
               α = 1.5, 
               rtol = 1e-4)

    Random.seed!(0)

    # Start out with uniform grid
    grid = fill(1/N, N, ndim)
    cgrid = cumsum(grid, dims = 1)

    # Sample `M` points from this uniform grid
    pts, bpts = generate_pts(grid, cgrid, M)

    # Initialize all cumulative variables 
    # for integral and sd estimation 
    summation = 0.
    summation2 = 0.
    σ²tot = 0.
    nevals = 0
    Itot = 0.
    sd = 0.
    chi = 0.
    integrals = Float64[]
    sigmas = Float64[]

    for i = 1:maxiter

        display(grid)
        display(bpts)

        # Estimate integral
        S, S², fevals = evaluate_at_samples(func, 
                                            pts, 
                                            bpts,
                                            M,
                                            N, 
                                            grid
                                           )

        σ² = (S² - S^2) * (1 / (M - 1)) |> abs
        σ²tot += 1/σ²
        summation += S^2/σ²
        summation2 += S^3/σ²
        nevals += M
        push!(integrals, S)
        push!(sigmas, σ²)

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

        # Update variables
        cumsum!(cgrid, grid, dims=1)
        pts, bpts = generate_pts(grid, cgrid, M)

        oldItot = Itot
        
        Itot = summation2 / summation
        sd = Itot * (summation)^(-0.5)

        if abs((oldItot - Itot) / oldItot) < rtol 
            @show (oldItot - Itot) / Itot 
            break
        end

    end
    χ = sum(((integrals .- Itot).^2) ./ sigmas)
    @show nevals

    Itot, sd, χ/(M-1)
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
    display(probs)

    for i = 1:M
        
        # Extract point
        p = vec(pts[i,:])

        #= Calculate area
        area = 1.
        for d = 1:dim
            area *= grid[bpts[i,d], d]
        end
        prob = 1 / (N^dim * area)=#

        # Get probability of that particular point
        prob = 0.
        for d = 1:dim
            prob *= probs[bpts[i,d],d]
        end

        # Eval function
        fp = f(p)
        fevals[i] = fp

        S += (fp / prob)
        S² += (fp / prob) ^ 2

    end

    S, S², fevals
end


"""
    pts, bpts = generate_pts(grid, cumgrid, M)

Generate `M` points from `grid` which probabilities 
inversely proportional to grid spacings 
"""
function generate_pts(grid, cgrid, M)

    # Get bins and dimension
    dim = size(cgrid, 2)
    N = size(cgrid, 1)

    # Each dimension needs M points
    pts = zeros(M, dim)
    bpts = zeros(Int, M, dim)

    for d = 1:dim

        # Take a slice of the grid 
        # corresponding the
        g = grid[:,d]
        prob = (1 ./ g) ./ sum(1 ./ g)

        # Create a multinomial distribution to 
        # pick which bins to sample from 
        mult = Multinomial(M, prob)
        samples = rand(mult)

        # Remember which bins they come from 
        bpts[:,d] = reduce(vcat, 
                           map(
                               (x,y) -> repeat([x], y),
                               1:N, 
                               samples
                              ))

        # Sample from corresponding bins
        idx = 1
        for (i,s) in enumerate(samples)
            if i == 1
                samp = rand(
                            Uniform(0, cgrid[1,d]),
                            s
                           )
            else
                samp = rand(
                            Uniform(cgrid[i-1,d],
                                    cgrid[i,d]
                                  ),
                            s
                           )
            end
            isempty(samp) && continue
            pts[idx:idx+s-1, d] .= samp
            idx += s
        end
        @assert idx == M + 1
    end

   pts, bpts
end

#=function calculate_m_dist(fevals, bpts, grid, K, α, dim)
    N = size(grid, 1)
    m = zeros(N, dim)
    for d = 1:dim
        for i = 1:size(bpts, 1)
            m[bpts[i,d],d] += (abs(fevals[i])*grid[bpts[i,d],d])
        end
        m[:,d] .+= 10eps()
        m[:,d] .= m[:,d] ./ sum(m[:,d])
        #m[:,d] .= K .* ((m[:,d] .- 1) .* (1 ./ log.(m[:,d]))) .^ α
        m[:,d] .= K .* m[:, d]
    end
    ceil.(m)
end=#

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
            m[bpts[i,d], d] += (fevals[i]^2 / 
                                (prod(probs[bpts[i,d],:]) / probs[bpts[i,d],d])
                               ) *
                                grid[bpts[i,d],d]
        end
        m[:,d] .= m[:,d] ./ sum(m[:,d])
        # m[:,d] .= K .* ((m[:,d] .- 1) .* (1 ./ log.(m[:,d]))) .^ α
        m[:,d] .= K .* m[:, d]
    end
    
    round.(m)
end


function update_grid!(grid, cgrid, N, M, m)
    dim = size(m, 2)
    for d = 1:dim
        morig = m[:,d]
        mcopy = copy(morig)
        optm = div(sum(morig), N)
        gridcol = grid[:,d]
        res = gridcol ./ morig
        
        for i = 1:N
            dist = extract_from_bins!(mcopy, optm, gridcol, i, morig, res)
            dist == 0 && (dist += 10eps())
            gridcol[i] = dist
        end

        grid[:,d] .= gridcol
    end
    grid
end

function extract_from_bins!(m, optm, grid, i, morig, res)
    N = size(m, 1)
    dist = 0.
    collected = 0
    for k = 1:N

        # First, calculate how many required
        required = optm - collected

        (m[k] == 0) && continue

        # If bin has what's required, take everything
        if required < m[k]
            #res = grid[k] / morig[k]
            dist += (res[k] * required)
            m[k] -= required
            collected += required
        end

        # Update requirements
        required = optm - collected

        # If more is required, take the entire bin
        if required > m[k]
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

    # Take remaining?
    if i==N && sum(m) != 0
        dist += sum(m)*res[end]
    end


    dist
end
