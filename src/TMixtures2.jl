module Mixtures
# My own implementation of mixture models (hard and soft, currently Gaussian only) with split-merge EM
using Distributions, Base.LinAlg, PDMats, StatsBase, StatsFuns, Clustering, Roots
import Base.split, Base.merge
export TMixture, em!, fsmem!, split, merge, score, assignments

"""
    verbose_iter(::Symbol)

Returns true if the verbosity level indicates we should print the log
likelihood on each iteration.
"""
verbose_iter(s::Symbol) = s == :iter

"""
    verbose_op(::Symbol)

Returns true if the verbosity level indicates we should print
information about split and merge operations.
"""
verbose_op(s::Symbol) = s == :iter || s == :op

function regularize_and_factorize!(c::Union{FullNormal,MvTDist}, λ::Real)
    Σmat = c.Σ.mat
    if λ != 0
        for i = diagind(Σmat)
            Σmat[i] += λ
        end
    end
    copy!(c.Σ.chol.factors, Σmat)
    LinAlg.chol!(c.Σ.chol.factors, Val{:U})
    c
end

function weighted_mean_cov!(mu, Σ, tmp, x, w)
    m = size(x, 1)
    n = size(x, 2)
    inv_sw = 1.0 / sum(w)
    Base.LinAlg.BLAS.gemv!('N', inv_sw, x, w, 0.0, mu)

    for j = 1:n
        cj = sqrt(w[j])
        for i = 1:m
            @inbounds tmp[i,j] = (x[i,j] - mu[i]) * cj
        end
    end
    LinAlg.copytri!(Base.LinAlg.BLAS.syrk!('U', 'N', inv_sw, tmp, 0.0, Σ), 'U')
end

type TMixture{T,M<:StridedMatrix}
    components::Vector{MvTDist}
    logπ::Vector{T}

    uniform_logpdf::T
    uniform_logπ::T

    X::M
    assignments::Vector{UInt8}
    λ::T
    fix_ν::Bool
    ll::T

    assignedu::Vector{T}
    maxtmp::Vector{T}
    mahaltmp::Vector{T}
    Xtmp::Matrix{T}
end

function compute_uniform_pdf(X)
    maxes = maximum(X, 2)
    mins = minimum(X, 2)
    -sum(log(maxes - mins))
end

function TMixture{T}(components::Vector{MvTDist}, X::StridedMatrix{T};
                     use_uniform::Bool=false, λ::T=convert(T, 0),
                     logπ::Vector{T}=fill(log(size(X, 1)/(size(X, 1)*length(components)+use_uniform)), length(components)),
                     fix_ν::Bool=false)
    length(components) < typemax(UInt8) || throw(ArgumentError("a maximum of $(typemax(UInt8)) components are supported"))
    TMixture(components, logπ,
             T(use_uniform ? compute_uniform_pdf(X) :-Inf),
             T(use_uniform ? log(1/(size(X, 1)*length(components)+use_uniform)) : -Inf),
             X, zeros(UInt8, size(X, 2)), λ, fix_ν, T(-Inf), zeros(T, size(X, 2)), zeros(T, size(X, 2)),
             zeros(T, size(X, 2)), zero(X))::TMixture{T,typeof(X)}
end

function initial_centers(k::Int, X::StridedMatrix)
    if k <= 0
        throw(ArgumentError("cannot create a mixture model with no components"))
    elseif k == 1
        mean(X, 2)
    else
        kmeans(X, k).centers
    end
end

function TMixture{T}(k::Int, X::StridedMatrix{T}; λ::T=convert(T, 0), centers::Matrix{T}=initial_centers(k, X), use_uniform::Bool=false, ν::T=T(50), fix_ν::Bool=false)
    k < typemax(UInt8) || throw(ArgumentError("a maximum of $(typemax(UInt8)) components are supported"))
    d = size(X, 1)
    sd = diagm(vec(std(X, 2)))/4
    components = MvTDist[MvTDist(ν, centers[:, i], PDMat(copy(sd), cholfact(sd))) for i = 1:k]
    TMixture(components, X; λ=λ, use_uniform=use_uniform, fix_ν=fix_ν)::TMixture{T,typeof(X)}
end

function merge_partial{T,C}(m::TMixture{T,C}, mpartial::TMixture{T,C}, original_components::Vector{Int}, partial_indices::Vector{Int})
    keepcomps = deleteat!(collect(1:length(m.components)), original_components)
    components = [deepcopy(m.components[i])::eltype(m.components) for i in keepcomps]
    append!(components, mpartial.components)
    merged = TMixture(components, m.X; λ=m.λ, use_uniform=m.uniform_logpdf != -Inf, fix_ν=m.fix_ν)
    merged.logπ = m.logπ[keepcomps]
    append!(merged.logπ, mpartial.logπ - StatsFuns.logsumexp(m.logπ[original_components]))

    old_assignments = m.assignments
    assignments = merged.assignments = similar(m.assignments)
    keepcomps_map = zeros(Int, length(m.components)+1)
    keepcomps_map[keepcomps+1] = 1:length(keepcomps)
    for i = 1:length(assignments)
        assignments[i] = keepcomps_map[old_assignments[i]+1]
    end
    partial_assignments = mpartial.assignments
    for i = 1:length(partial_indices)
        assignments[partial_indices[i]] = partial_assignments[i]
    end
    
    merged
end

function emstep!(m::TMixture)
    components = m.components
    logπ = m.logπ
    X = m.X
    assignments = m.assignments
    assignedu = m.assignedu
    maxtmp = m.maxtmp
    mahaltmp = m.mahaltmp
    Xtmp = m.Xtmp
    λ = m.λ
    fix_ν = m.fix_ν

    has_uniform = m.uniform_logpdf != -Inf

    # E step
    fill!(maxtmp, m.uniform_logpdf + m.uniform_logπ)
    fill!(assignments, 0)
    for k = 1:length(components)
        c = components[k]
        # Compute squared Mahalanobis distance
        broadcast!(-, Xtmp, X, c.μ)
        invquad!(mahaltmp, c.Σ, Xtmp)

        shdfhdim, v = Distributions.mvtdist_consts(c)
        lπ = logπ[k]
        for i = 1:size(X, 2)
            # Compute logpdf and see if it is better than for other components
            logpdf = v - shdfhdim * log1p(mahaltmp[i] / c.df)
            l = logpdf + lπ
            if l > maxtmp[i]
                maxtmp[i] = l
                assignments[i] = k
                assignedu[i] = (size(X, 1) + c.df)/(mahaltmp[i] + c.df)
            end
        end
    end
    m.ll = sum(maxtmp)
    isempty(components) && return

    # M step
    nuniform = 0
    if has_uniform
        for i = 1:length(assignments)
            nuniform += assignments[i] == 0
        end
        m.uniform_logπ = log((1+nuniform)/(size(X, 2)+1))
    end
    y = -digamma((size(X, 1) + components[1].df)/2)*(size(X, 2) - nuniform)
    lognsamples = log(size(X, 2)+has_uniform)
    for k = 1:length(components)
        c = components[k]
        incomponent = find(assignments .== k)

        # Update μ and Σ
        # We have actually already computed Xtmp before, so if this is
        # a computationally intensive step, we should just use more memory...
        Xcomp = X[:, incomponent]
        wt = assignedu[incomponent]
        weighted_mean_cov!(c.μ, c.Σ.mat, Xcomp, Xcomp, wt)
        regularize_and_factorize!(c, λ)

        # Update mixing proportions
        logπ[k] = log(length(incomponent)) - lognsamples

        # Update ν
        if !fix_ν
            for i = 1:length(wt)
                y -= log(2*(wt[i]/(c.dim + c.df))) - wt[i]
            end
        end
    end

    # Update degrees of freedom
    if !fix_ν
        dgconst = digamma((size(X, 1) + components[1].df)/2)
        y /= (size(X, 2) - nuniform)
        ν = fzero(ν->-y+log(ν/2)+1-digamma(ν/2), 1.0, 100000.0)
        yplogym1 = 1/(y+log(y)-1)
        # ν = min(2*yplogym1+0.0416*(1+erf(0.6594*log(2.1971*yplogym1))), 100.0)
        for k = 1:length(components)
            components[k] = MvTDist(ν, components[k].μ, components[k].Σ)
        end
    end

    m
end

function nparameters(d::Union{FullNormal,MvTDist})
    n = length(d.μ)
    div(n*(n-1), 2)+n
end

function score(m::TMixture)
    k = length(m.components)
    N = nparameters(m.components[1])
    n = size(m.X, 2)

    penalty = N/2*(k*log(n/12)+sum(m.logπ)) + k/2*log(n/12) + k*(N+1)/2
    @assert penalty > 0
    m.ll - penalty
end


function computeγ{T}(m::TMixture{T})
    components = m.components
    logπ = m.logπ
    X = m.X
    sumtmp = m.mahaltmp
    maxtmp = m.maxtmp

    γ = zeros(T, size(X, 2), length(components))

    fill!(maxtmp, -Inf)
    for k = 1:length(components)
        γ_k = sub(γ, :, k)
        # Compute log(π_k*p(x_n|θ_k))
        logpdf!(γ_k, components[k], X)
        lπ = logπ[k]
        for i = 1:size(X, 2)
            γ_k[i] += lπ
            maxtmp[i] = max(maxtmp[i], γ_k[i])
        end
    end

    # Now compute softmax over rows of γ to complete Eq. 9.23
    fill!(sumtmp, 0)
    for k = 1:length(components), i = 1:size(X, 2)
        sumtmp[i] += γ[i, k] = exp(γ[i, k] - maxtmp[i])
    end

    # Finish the softmax
    broadcast!(*, γ, γ, broadcast!(inv, sumtmp, sumtmp))

    γ
end

function logsumexp(a, b)
    u = max(a, b)
    v = min(a, b)
    u + log1p(exp(v-u))
end

"""
    merge!(out, c1, logπ1, c2, logπ2)

Merge components c1 and c2 into out and return new log mixing coefficient
"""
function merge!{C<:Union{MvTDist,FullNormal}}(out::C, c1::C, logπ1::Real, c2::C, logπ2::Real)
    π1 = exp(logπ1)
    π2 = exp(logπ2)

    # Merge μ
    for m = 1:length(c1.μ)
        out.μ[m] = (π1*c1.μ[m] + π2*c2.μ[m])/(π1+π2)
    end

    # Merge Σ
    for n = 1:size(c1.Σ.mat, 2), m = 1:size(c2.Σ.mat, 1)
        out.Σ.chol.factors[m, n] = out.Σ.mat[m, n] = (π1*c1.Σ.mat[m, n] + π2*c2.Σ.mat[m, n])/(π1+π2)
    end
    LinAlg.chol!(out.Σ.chol.factors, Val{:U})

    # New log mixing coefficient
    logsumexp(logπ1, logπ2)
end

function merge_priority(m::TMixture)
    γ = computeγ(m)
    c = Base.corzm(γ)
    ncandidates = div(size(γ, 2)*(size(γ, 2) - 1), 2)
    candidates = Array(Tuple{Int,Int}, ncandidates)
    priority = zeros(ncandidates)
    n = 1
    for i = 1:size(γ, 2), j = i+1:size(γ, 2)
        candidates[n] = (i, j)
        priority[n] = c[i, j]
        n += 1
    end
    (priority, candidates)
end

function merge{T}(m::TMixture{T}, i::Int, j::Int; tol::T=1e-3, maxiter::Int=200, verbose::Symbol=:none)
    ncomponents = length(m.components)
    mnew = deepcopy(m)
    deleteat!(mnew.logπ, j)
    deleteat!(mnew.components, j)
    assignmap = zeros(Int, ncomponents+1)
    assignmap[[2:j; j+2:ncomponents+1]] = 1:ncomponents-1
    assignmap[j+1] = i
    mnew.assignments = [assignmap[x+1] for x in m.assignments]
    mnew.logπ[i] = merge!(mnew.components[i], m.components[i], m.logπ[i], m.components[j], m.logπ[j])
    em!(mnew; tol=tol, maxiter=maxiter, verbose=verbose)
end

function merge{T}(m::TMixture{T}; tol::T=1e-3, maxiter::Int=200, verbose::Symbol=:none)
    ncomponents = length(m.components)
    ncomponents == 1 && return m

    # Find merge candidates
    (priority, candidates) = merge_priority(m)

    # Try some of the best candidates
    maxcandidates = min(ceil(Int, ncomponents*sqrt(ncomponents)/2), length(priority))
    p = sortperm(priority)
    origscore = score(m)
    for icandidate = length(p):-1:length(p)-maxcandidates+1
        i, j = candidates[p[icandidate]]

        # Do the merge
        verbose_op(verbose) && println("Attempting to merge components $i and $j...")
        mnew = merge(m, i, j; tol=tol, maxiter=maxiter, verbose=verbose)

        newscore = score(mnew)
        if newscore > origscore
            verbose_op(verbose) && println("    score increased ($origscore -> $newscore); accepted")
            return mnew
        else
            verbose_op(verbose) && println("    score decreased ($origscore -> $newscore); rejected")
        end
    end
    return m
end

function split_priority{T}(m::TMixture{T})
    γ = computeγ(m)
    components = m.components
    priority = zeros(length(components))
    fk = zeros(T, size(m.X, 2))
    logpk = zeros(T, size(m.X, 2))
    for k = 1:length(components)
        γ_k = sub(γ, :, k)
        scale!(fk, γ_k, 1/sum(γ_k))
        logpdf!(logpk, components[k], m.X)
        Jsplit = zero(T)
        for i = 1:size(m.X, 2)
            Jsplit += fk[i]*(log(fk[i])-logpk[i])
        end
        priority[k] = Jsplit
    end
    priority
end

function split{T}(m::TMixture{T}, i::Int; tol::T=1e-3, maxiter::Int=200, verbose::Symbol=:none)
    indpartial = find(m.assignments .== i)
    if length(indpartial) < size(m.X, 1)*2
        return m
    end

    mpartial = TMixture(2, m.X[:, indpartial]; λ=m.λ, use_uniform=false, ν=m.components[i].df, fix_ν=true)
    em!(mpartial)
    if length(mpartial.components) == 1
        m
    else
        # Merge the partial EM results into a new model
        mnew = merge_partial(m, mpartial, [i], indpartial)
        em!(mnew; tol=tol, maxiter=maxiter, verbose=verbose)
    end
end

function split{T}(m::TMixture{T}; tol::T=1e-3, maxiter::Int=200, verbose::Symbol=:none)
    # Find split candidates
    priority = split_priority(m)

    # Try each candidate
    p = sortperm(priority)
    origscore = score(m)
    for icandidate = length(p):-1:1
        candidate = p[icandidate]

        # Do the split
        verbose_op(verbose) && println("Attempting to split component $candidate...")
        mnew = split(m, candidate; tol=tol, maxiter=maxiter, verbose=verbose)
        if mnew == m
            verbose_op(verbose) && println("    new component degenerate; rejected")
        else
            newscore = score(mnew)
            if newscore > origscore
                verbose_op(verbose) && println("    score increased ($origscore -> $newscore); accepted")
                return mnew
            else
                verbose_op(verbose) && println("    score decreased ($origscore -> $newscore); rejected")
            end
        end
    end
    return m
end

function drop_degenerate!(m::TMixture)
    assigned = zeros(Int, length(m.components))
    assignments = m.assignments
    for i = 1:length(assignments)
        if assignments[i] != 0
            assigned[assignments[i]] += 1
        end
    end

    assigned_ind = find(assigned .>= size(m.X, 1))
    length(assigned_ind) == length(m.components) && return
    assignmap = zeros(UInt8, length(m.components))
    assignmap[assigned_ind] = 1:length(assigned_ind)
    m.components = m.components[assigned_ind]
    m.logπ = m.logπ[assigned_ind]
    for i = 1:length(assignments)
        if assignments[i] != 0
            assignments[i] = assignmap[assignments[i]]
        end
    end
    m
end

function em!{T}(m::TMixture{T}; tol::T=1e-2, maxiter::Int=200, verbose::Symbol=:none)
    niter = 0
    m.ll = -Inf
    old_assignments = copy(m.assignments)
    while niter < maxiter
        oldll = m.ll
        old_assignments, m.assignments = m.assignments, old_assignments
        emstep!(m)
        drop_degenerate!(m)
        verbose_iter(verbose) && println("    $niter $(m.ll)")
        !isfinite(m.ll) && error("log likelihood is not finite")
        # m.assignments == old_assignments && return m
        m.ll - oldll < tol && return m
        niter += 1
    end
    warn("EM did not converge")
    m
end

function fsmem!{T}(m::TMixture{T}; tol::T=1e-2, maxiter::Int=200, verbose::Symbol=:none)
    verbose_op(verbose) && println("Performing initial EM...")
    em!(m, tol=tol, maxiter=maxiter, verbose=verbose)
    while true
        mouter = m
        while true
            minner = m
            m = merge(m, tol=tol, maxiter=maxiter, verbose=verbose)
            m === minner && break
        end
        while true
            minner = m
            m = split(m, tol=tol, maxiter=maxiter, verbose=verbose)
            m === minner && break
        end
        m === mouter && break
    end
    m
end

assignments{T}(m::TMixture{T}) = m.assignments

end # module
