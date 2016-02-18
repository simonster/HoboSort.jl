module Mixtures
# My own implementation of mixture models (hard and soft, currently Gaussian only) with split-merge EM
using Distributions, Base.LinAlg, PDMats, StatsBase
import Base.split, Base.merge
export Mixture, SoftMixtureFit, HardMixtureFit, em!, fsmem!, split, merge, score, assignments

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

"""
    uninitialized(::Type{D<:Distribution}, d::Int)

Create a new "uninitialized" distribution object of a given
dimensionality.
"""
uninitialized(::Type{FullNormal}, d::Int) =
    FullNormal(Array(Float64, d), PDMat(Array(Float64, d, d), Cholesky{Float64,Matrix{Float64}}(Array(Float64, d, d), 'U')))
uninitialized(::Type{DiagNormal}, d::Int) =
    DiagNormal(Array(Float64, d), PDiagMat(Array(Float64, d), Array(Float64, d)))

type Mixture{T,C<:Distribution}
    components::Vector{C}
    logπ::Vector{T}
end

Mixture(components::Vector) =
    Mixture(components, Array(Float64, length(components)))
function Mixture{C<:Distribution}(::Type{C}, d::Int, k::Int)
    components = Array(C, k)
    for i = 1:k
        components[i] = uninitialized(C, d)
    end
    Mixture(components)
end
Mixture(d::Int, k::Int) = Mixture(FullNormal, d, k)
function Mixture{T,C}(m::Mixture{T,C}, k::Int)
    components = eltype(m.components)[deepcopy(m.components[i]) for i = 1:min(k, length(m.components))]
    for i = length(m.components)+1:k
        push!(components, deepcopy(m.components[end]))
    end
    Mixture(components, Array(T, k))
end

abstract MixtureFit{T,C}

type SoftMixtureFit{T,C,M<:StridedMatrix} <: MixtureFit{T,C}
    m::Mixture{T,C}
    X::M
    γ::Vector{Vector{T}}
    λ::T
    ll::T

    tmp1::Vector{T}
    tmp2::Vector{T}
end

function initialγ(T::Type, n::Int, k::Int)
    q = rand(T, n, k)
    q ./= sum(q, 2)
    [q[:, i] for i = 1:k]
end

function SoftMixtureFit{T}(m::Mixture, X::StridedMatrix{T}; λ::T=convert(T, 0), γ::Vector{Vector{T}}=initialγ(T, size(X, 2), length(m.components)))
    length(γ) == length(m.components) || throw(DimensionMismatch("length of γ must match number of components"))
    for i = 1:length(γ)
        length(γ[i]) == size(X, 2) || throw(DimensionMismatch("length of each element of γ must match number of points"))
    end
    SoftMixtureFit(m, X, γ, λ, T(-Inf),
               zeros(T, size(X, 2)), zeros(T, size(X, 2)))
end

type PartialSoftMixtureFit{T,C,M<:StridedMatrix} <: MixtureFit{T,C}
    m::Mixture{T,C}
    X::M
    γ::Vector{Vector{T}}
    λ::T
    ll::T

    tmp1::Vector{T}
    tmp2::Vector{T}

    original_components::Vector{Int}
    logpdf_adjust::Vector{T}
    ll_adj::T
end

PartialMixtureFit{T}(m::SoftMixtureFit{T}, k::Int) =
    PartialSoftMixtureFit(Mixture(m.m, k), m.X, [zeros(T, size(m.X, 2)) for i = 1:k], m.λ, -Inf,
                          m.tmp1, m.tmp2, Int[], zeros(T, size(m.X, 2)), -Inf)

type HardMixtureFit{T,C,M<:StridedMatrix} <: MixtureFit{T,C}
    m::Mixture{T,C}
    X::M
    assignments::Vector{UInt8}
    λ::T
    ll::T

    tmp1::Vector{T}
    tmp2::Vector{T}
end

function HardMixtureFit{T}(m::Mixture, X::StridedMatrix{T}; λ::T=convert(T, 0), assignments=rand(UInt8(1):UInt8(length(m.components)), size(X, 2)))
    k = length(m.components)
    k < typemax(UInt8) || throw(ArgumentError("a maximum of $(typemax(UInt8)) components are supported"))
    length(assignments) == size(X, 2) || throw(DimensionMismatch("number of assignments must match number of points"))
    HardMixtureFit(m, X, assignments, λ, T(-Inf),
                   zeros(T, size(X, 2)), zeros(T, size(X, 2)))
end

type PartialHardMixtureFit{T,C,M<:StridedMatrix} <: MixtureFit{T,C}
    m::Mixture{T,C}
    X::M
    assignments::Vector{UInt8}
    λ::T
    ll::T

    tmp1::Vector{T}
    tmp2::Vector{T}

    original_components::Vector{Int}
end

PartialMixtureFit{T}(m::HardMixtureFit{T}, k::Int) =
    PartialHardMixtureFit(Mixture(m.m, k), m.X, zeros(UInt8, size(m.X, 2)), m.λ, -Inf,
                          m.tmp1, m.tmp2, Int[])

function setup_partial!{T}(mpartial::PartialSoftMixtureFit{T}, m::SoftMixtureFit, comps)
    mpartial.original_components = comps
    logpdf_adjust = mpartial.logpdf_adjust
    fill!(logpdf_adjust, nextfloat(zero(T)))
    for k in comps
        broadcast!(+, logpdf_adjust, logpdf_adjust, m.γ[k])
    end
    broadcast!(log, logpdf_adjust, logpdf_adjust)
    mpartial.ll_adj = m.ll - sum(logpdf_adjust)
end

function setup_partial!(mpartial::PartialHardMixtureFit, m::HardMixtureFit, comps)
    mpartial.original_components = comps
    old_assignments = m.assignments
    new_assignments = mpartial.assignments
    for i = 1:length(old_assignments)
        v = old_assignments[i]
        u = 0
        for j = 1:length(comps)
            u = ifelse(v == comps[j], j, u)
        end
        new_assignments[i] = u
    end
end

function merge_partial{T,C}(m::SoftMixtureFit{T,C}, mpartial::PartialSoftMixtureFit{T,C})
    keepcomps = deleteat!(collect(1:length(m.m.components)), mpartial.original_components)
    components = [deepcopy(m.m.components[i])::eltype(m.m.components) for i in keepcomps]
    append!(components, mpartial.m.components)
    logπ = m.m.logπ[keepcomps]
    append!(logπ, mpartial.m.logπ)

    γ = [copy(m.γ[i]) for i in keepcomps]
    append!(γ, mpartial.γ)
    SoftMixtureFit(Mixture(components, logπ), m.X, γ, m.λ, -Inf, m.tmp1, m.tmp2)
end

function merge_partial{T,C}(m::HardMixtureFit{T,C}, mpartial::PartialHardMixtureFit{T,C})
    original_components = mpartial.original_components
    keepcomps = deleteat!(collect(1:length(m.m.components)), original_components)
    components = [deepcopy(m.m.components[i])::eltype(m.m.components) for i in keepcomps]
    append!(components, mpartial.m.components)
    logπ = m.m.logπ[keepcomps]
    append!(logπ, mpartial.m.logπ)

    old_assignments = m.assignments
    partial_assignments = mpartial.assignments
    assignments = similar(m.assignments)
    keepcomps_map = zeros(Int, length(m.m.components))
    keepcomps_map[keepcomps] = 1:length(keepcomps)
    for i = 1:length(assignments)
        v = old_assignments[i]
        u = keepcomps_map[v]
        for j = 1:length(original_components)
            u = ifelse(v == original_components[j], length(original_components)+j, u)
        end
        assignments[i] = u
    end
    HardMixtureFit(Mixture(components, logπ), m.X, assignments, m.λ, -Inf, m.tmp1, m.tmp2)
end

adjust_logpdf!(m::Union{SoftMixtureFit,HardMixtureFit,PartialHardMixtureFit}, γ_k, logπ) =
    broadcast!(+, γ_k, γ_k, logπ)
adjust_logpdf!(m::PartialSoftMixtureFit, γ_k, logπ) =
    broadcast!(+, γ_k, γ_k, m.logpdf_adjust, logπ)
adjust_ll(m::Union{SoftMixtureFit,HardMixtureFit,PartialHardMixtureFit}, ll) = ll
adjust_ll(m::PartialSoftMixtureFit, ll) = ll + m.ll_adj

function computeγ!{T}(γ::Vector{Vector{T}}, m::MixtureFit{T})
    components = m.m.components
    logπ = m.m.logπ
    X = m.X
    sumtmp = m.tmp1
    maxtmp = m.tmp2

    fill!(maxtmp, -Inf)
    for k = 1:length(components)
        γ_k = γ[k]
        # Compute log(π_k*p(x_n|θ_k))
        logpdf!(γ_k, components[k], X)
        adjust_logpdf!(m, γ_k, logπ[k])
        for i = 1:size(X, 2)
            maxtmp[i] = max(maxtmp[i], γ_k[i])
        end
    end

    # Now compute softmax over rows of γ to complete Eq. 9.23
    fill!(sumtmp, 0)
    for k = 1:length(components)
        γ_k = γ[k]
        for i = 1:size(X, 2)
            sumtmp[i] += γ_k[i] = exp(γ_k[i] - maxtmp[i])
        end
    end

    # Log likelihood (Eq. 9.28)
    ll = sum(maxtmp)
    for i = 1:length(sumtmp)
        ll += log(sumtmp[i])
    end
    ll = adjust_ll(m, ll)

    # Finish the softmax
    broadcast!(inv, sumtmp, sumtmp)
    for k = 1:length(components)
        broadcast!(*, γ[k], γ[k], sumtmp)
    end
    ll
end
function computeγ{T}(m::MixtureFit{T})
    γ = [zeros(T, size(m.X, 2)) for i = 1:length(m.m.components)]
    computeγ!(γ, m)
    γ
end

function estep!{T}(m::Union{SoftMixtureFit{T},PartialSoftMixtureFit{T}})
    m.ll = computeγ!(m.γ, m)
    m
end

function estep!{T}(m::Union{HardMixtureFit{T},PartialHardMixtureFit{T}})
    components = m.m.components
    logπ = m.m.logπ
    pdftmp = m.tmp1
    maxtmp = m.tmp2
    fill!(maxtmp, -Inf)
    assignments = m.assignments
    for k = 1:length(components)
        # TODO very inefficient for partial!
        logpdf!(pdftmp, components[k], m.X)
        for i = 1:length(pdftmp)
            assignments[i] == 0 && continue
            l = pdftmp[i] + logπ[k]
            improved = l > maxtmp[i]
            maxtmp[i] = ifelse(improved, l, maxtmp[i])
            assignments[i] = ifelse(improved, k, assignments[i])
        end
    end

    ll = zero(T)
    for i = 1:length(assignments)
        ll += ifelse(assignments[i] == 0, zero(T), maxtmp[i])
    end
    m.ll = ll
    m
end

function regularize_and_factorize!(c::FullNormal, λ::Real)
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

function estimate!(c::FullNormal, x::AbstractMatrix{Float64}, w::AbstractVector{Float64}, λ::Real)
    m = size(x, 1)
    n = size(x, 2)

    inv_sw = 1.0 / sum(w)
    mu = c.μ
    Base.LinAlg.BLAS.gemv!('N', inv_sw, x, w, 0.0, mu)

    z = Array(Float64, m, n)
    for j = 1:n
        cj = sqrt(w[j])
        for i = 1:m
            @inbounds z[i,j] = (x[i,j] - mu[i]) * cj
        end
    end
    LinAlg.copytri!(Base.LinAlg.BLAS.syrk!('U', 'N', inv_sw, z, 0.0, c.Σ.mat), 'U')
    regularize_and_factorize!(c, λ)
end

function estimate!(c::FullNormal, x::AbstractMatrix{Float64}, λ::Real)
    n = size(x, 2)

    mu = mean!(c.μ, x)
    z = x .- mu

    Σmat = c.Σ.mat
    Base.LinAlg.copytri!(Base.LinAlg.BLAS.syrk!('U', 'N', 1/n, z, 0.0, c.Σ.mat), 'U')
    regularize_and_factorize!(c, λ)
end

function regularize_and_factorize!(c::DiagNormal, λ::Real)
    diag = c.Σ.diag
    invdiag = c.Σ.inv_diag
    for i = 1:length(diag)
        invdiag[i] = inv(diag[i] += λ)
    end
    c
end

function estimate!(c::DiagNormal, x::AbstractMatrix{Float64}, w::AbstractVector{Float64}, λ::Real)
    m = size(x, 1)
    n = size(x, 2)
    wt = weights(w)
    mu = mean!(c.μ, x, wt, 2)
    diag = Base.varm!(c.Σ.diag, x, mu, wt, 2)
    regularize_and_factorize!(c, λ)
end

function estimate!(c::DiagNormal, x::AbstractMatrix{Float64}, λ::Real)
    n = size(x, 2)
    mu = mean!(c.μ, x)
    diag = Base.varm!(c.Σ.diag, x, mu)
    regularize_and_factorize!(c, λ)
end

function mstep!{T,C}(m::Union{SoftMixtureFit{T,C},PartialSoftMixtureFit{T,C}})
    γ = m.γ
    logπ = m.m.logπ
    components = m.m.components
    for k = 1:length(components)
        # Eq. 9.24 and 9.25
        c = components[k]
        estimate!(c, m.X, γ[k], m.λ)
        # Eq. 9.26
        logπ[k] = log(mean(γ[k]))
    end
    m
end

function mstep!{T,C}(m::Union{HardMixtureFit{T,C},PartialHardMixtureFit{T,C}})
    logπ = m.m.logπ
    components = m.m.components
    for k = 1:length(components)
        # Eq. 9.24 and 9.25
        c = components[k]
        incomponent = m.assignments .== k
        # TODO inefficient
        estimate!(c, m.X[:, incomponent], m.λ)
        # Eq. 9.26
        logπ[k] = log(mean(incomponent))
    end
    m
end

function logsumexp(a, b)
    u = max(a, b)
    v = min(a, b)
    u + log1p(exp(v-u))
end

function nparameters(d::FullNormal)
    n = length(d.μ)
    div(n*(n-1), 2)+n
end
nparameters(d::DiagNormal) = length(d.μ)*6

function score{T,C<:Union{MvNormal,MvTDist}}(m::MixtureFit{T,C})
    k = length(m.m.components)
    N = nparameters(m.m.components[1])
    n = size(m.X, 2)

    penalty = N/2*(k*log(n/12)+sum(m.m.logπ[m.m.logπ .> 1/n])) + k/2*log(n/12) + k*(N+1)/2
    @assert penalty > 0
    m.ll - penalty
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
function merge!(out::DiagNormal, c1::DiagNormal, logπ1::Real, c2::DiagNormal, logπ2::Real)
    π1 = exp(logπ1)
    π2 = exp(logπ2)

    # Merge μ and Σ
    for m = 1:length(c1.μ)
        out.μ[m] = (π1*c1.μ[m] + π2*c2.μ[m])/(π1+π2)
        out.Σ.diag[m] = (π1*c1.Σ.diag[m] + π2*c2.Σ.diag[m])/(π1+π2)
        out.Σ.inv_diag[m] = 1/out.Σ.diag[m]
    end

    # New log mixing coefficient
    logsumexp(logπ1, logπ2)
end

function merge_priority(m::MixtureFit)
    γ = isa(m, SoftMixtureFit) ? m.γ : computeγ(m)
    norms = [norm(x) for x in γ]
    ncandidates = div(length(γ)*(length(γ) - 1), 2)
    priority = zeros(ncandidates)
    candidates = Array(Tuple{Int,Int}, ncandidates)
    n = 1
    for i = 1:length(γ), j = i+1:length(γ)
        candidates[n] = (i, j)
        priority[n] = dot(γ[i], γ[j])/(norms[i]*norms[j])
        n += 1
    end
    (priority, candidates)
end

function merge{T}(m::SoftMixtureFit{T}; tol::T=1e-3, maxiter::Int=200, verbose::Symbol=:none)
    ncomponents = length(m.m.components)
    ncomponents == 1 && return m

    # Find merge candidates
    (priority, candidates) = merge_priority(m)

    # Try some of the best candidates
    maxcandidates = min(ceil(Int, ncomponents*sqrt(ncomponents)/2), length(priority))
    p = sortperm(priority)
    mpartial = PartialMixtureFit(m, 1)
    merged = mpartial.m.components[1]
    origscore = score(m)
    for icandidate = length(p):-1:length(p)-maxcandidates+1
        i, j = candidates[p[icandidate]]

        # Do the merge
        verbose_op(verbose) && println("Attempting to merge components $i and $j...")
        mpartial.m.logπ[1] = merge!(mpartial.m.components[1], m.m.components[i], m.m.logπ[i], m.m.components[j], m.m.logπ[j])

        # There's only one component, so I don't think anything changes with full EM?
        setup_partial!(mpartial, m, [i, j])
        estep!(mpartial)

        # Merge the partial EM results into a new model
        mnew = merge_partial(m, mpartial)
        em!(mnew; tol=tol, maxiter=maxiter, verbose=verbose)

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

function merge{T}(m::HardMixtureFit{T}; tol::T=1e-3, maxiter::Int=200, verbose::Symbol=:none)
    ncomponents = length(m.m.components)
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
        mnew = deepcopy(m)
        assignmap = zeros(Int, ncomponents)
        assignmap[[1:j-1; j+1:ncomponents]] = 1:ncomponents-1
        assignmap[j] = i
        mnew.assignments = [assignmap[x] for x in m.assignments]
        em!(mnew; tol=tol, maxiter=maxiter, verbose=verbose)
        drop_degenerate!(mnew)

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

function drop_degenerate!(m::HardMixtureFit)
    assigned = fill(false, length(m.m.components))
    assignments = m.assignments
    for i = 1:length(assignments)
        assigned[assignments[i]] = true
    end

    assigned_ind = find(assigned)
    assignmap = zeros(UInt8, length(m.m.components))
    assignmap[assigned_ind] = 1:length(assigned_ind)
    m.m.components = m.m.components[assigned_ind]
    m.m.logπ = m.m.logπ[assigned_ind]
    for i = 1:length(assignments)
        assignments[i] = assignmap[assignments[i]]
    end
    m
end

drop_degenerate!(m::SoftMixtureFit) = m

"""
    split!(out1, out2, c, logπ)

Split component c into out1 and out2 and return new log mixing coefficient
"""
function split!{C<:Union{MvTDist,FullNormal}}(out1::C, out2::C, c::C, logπ::Real)
    # Split μ
    ε = scale!(c.Σ.chol.factors'randn(c.Σ.dim), 0.2)
    broadcast!(+, out1.μ, c.μ, ε)
    broadcast!(-, out2.μ, c.μ, ε)

    # Split Σ
    q = logdet(c.Σ)/c.Σ.dim
    s = exp(q)
    for Σ in (out1.Σ, out2.Σ)
        mat = Σ.mat
        chol = Σ.chol
        fill!(mat, 0)
        fill!(chol.factors, 0)
        mat[diagind(mat)] = s
        chol.factors[diagind(chol.factors)] = sqrt(s)
    end

    # New log mixing coefficient
    logπ - log(2)
end
function split!(out1::DiagNormal, out2::DiagNormal, c::DiagNormal, logπ::Real)
    # Split μ
    ε = scale!(c.Σ.inv_diag.*randn(c.Σ.dim), 0.5)
    broadcast!(+, out1.μ, c.μ, ε)
    broadcast!(-, out2.μ, c.μ, ε)

    # Split Σ
    q = sum(log(c.Σ.diag))/c.Σ.dim/2
    s = exp(q)
    sinv = inv(s)
    for Σ in (out1.Σ, out2.Σ)
        copy!(Σ.diag, s)
        copy!(Σ.inv_diag, sinv)
    end

    # New log mixing coefficient
    logπ - log(2)
end

function split_priority{T}(m::MixtureFit{T})
    γ = isa(m, SoftMixtureFit) ? m.γ : computeγ(m)
    components = m.m.components
    priority = zeros(length(components))
    fk = zeros(T, size(m.X, 2))
    logpk = zeros(T, size(m.X, 2))
    for k = 1:length(components)
        γ_k = γ[k]
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

function split{T}(m::MixtureFit{T}; tol::T=1e-3, maxiter::Int=200, verbose::Symbol=:none)
    # Find split candidates
    priority = split_priority(m)

    # Try each candidate
    p = sortperm(priority)
    mpartial = PartialMixtureFit(m, 2)
    split1, split2 = mpartial.m.components
    origscore = score(m)
    for icandidate = length(p):-1:1
        candidate = p[icandidate]

        # Do the split
        verbose_op(verbose) && println("Attempting to split component $candidate...")
        mpartial.m.logπ[:] = split!(split1, split2, m.m.components[candidate], m.m.logπ[candidate])

        # Perform EM
        setup_partial!(mpartial, m, [candidate])
        estep!(mpartial)
        em!(mpartial)

        # Merge the partial EM results into a new model
        mnew = merge_partial(m, mpartial)
        estep!(mnew)
        em!(mnew; tol=tol, maxiter=maxiter, verbose=verbose)
        drop_degenerate!(mnew)

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

function em!{T}(m::MixtureFit{T}; tol::T=1e-3, maxiter::Int=200, verbose::Symbol=:none)
    niter = 0
    m.ll = -Inf
    while niter < maxiter
        oldll = m.ll
        mstep!(m)
        estep!(m)
        verbose_iter(verbose) && println("    $niter $(m.ll)")
        !isfinite(m.ll) && error("log likelihood is not finite")
        abs(m.ll - oldll) < tol && return m
        niter += 1
    end
    warn("EM did not converge")
    m
end

function fsmem!{T}(m::MixtureFit{T}; tol::T=1e-3, maxiter::Int=200, verbose::Symbol=:none)
    verbose_op(verbose) && println("Performing initial EM...")
    em!(m, tol=tol, maxiter=maxiter, verbose=verbose)
    while true
        mouter = m
        isa(m, HardMixtureFit) && drop_degenerate!(m)
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

function assignments{T}(m::SoftMixtureFit{T})
    out = zeros(Int, size(m.X, 2))
    maxes = fill(T(-Inf), size(m.X, 2))
    for k = 1:length(m.γ)
        γ_k = m.γ[k]
        for i = 1:length(out)
            if γ_k[i] > maxes[i]
                out[i] = k
                maxes[i] = γ_k[i]
            end
        end
    end
    out
end

assignments{T}(m::HardMixtureFit{T}) = m.assignments

end # module
