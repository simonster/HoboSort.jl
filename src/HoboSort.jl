module HoboSort
include("Extract.jl")
include("TMixtures2.jl")
include("HoboPlot.jl")
include("cholutils.jl")
using Base.Dates, Base.LinAlg, Distributions, PDMats, .Extract, .Mixtures, .HoboPlot

immutable SortParams
    init_period::Second
    inspect_period::Second
    history_period::Second
    min_separation_periods::Int
    split_score_improvement::Float64
    merge_score_improvement::Float64
    min_cluster_size::Int
    initial_clusters::Int
    verbose::Bool
    plot::Bool
    movie_file::UTF8String
end
SortParams(; init_period::TimePeriod=Minute(10), inspect_period::TimePeriod=Minute(5),
           history_period::TimePeriod=Minute(10), min_separation_periods::Int=6,
           merge_score_improvement::Real=0, split_score_improvement::Real=0,
           min_cluster_size::Integer=180, initial_clusters::Integer=10,
           verbose::Bool=false, plot::Bool=false, movie_file::AbstractString="") =
	SortParams(init_period, inspect_period, history_period, min_separation_periods,
               split_score_improvement, merge_score_improvement,
               min_cluster_size, initial_clusters, verbose, plot, movie_file)

function fit_tdist{T}(X::AbstractMatrix{T}, ν;
                      maxiter::Int=100, tol=1e-3, μ_init::Vector{T}=vec(mean(X, 2)),
                      Σ_init::Union{PDMat{T},Void}=nothing)
    μ = copy(μ_init)
    Xmμ = X .- μ
    Σ::PDMat{T,Matrix{T}} = Σ_init === nothing ? PDMat(LinAlg.copytri!(Base.LinAlg.BLAS.syrk('U', 'N', 1/size(X, 2), Xmμ), 'U')) : deepcopy(Σ_init)
    mahal = η = zeros(eltype(X), size(X, 2))
    dist = MvTDist(ν, μ, Σ)
    ll = -Inf
    for t = 1:maxiter
        oldll = ll
        ll = 0.0

        # E step
        broadcast!(-, Xmμ, X, μ)
        mahal = invquad!(mahal, Σ, Xmμ)
        shdfhdim, v = Distributions.mvtdist_consts(dist)
        for i = 1:size(X, 2)
            ll += v - shdfhdim * log1p(mahal[i] / ν)
            η[i] = (ν + size(X, 1))/(ν + mahal[i])
        end

        # M step
        Mixtures.weighted_mean_cov!(μ, Σ.mat, Xmμ, X, η)
        copy!(Σ.chol.factors, Σ.mat)
        LinAlg.chol!(Σ.chol.factors, Val{:U})

        abs(ll - oldll) < tol && return (dist, η, ll)
    end
    warning("T distribution fit did not converge")
    (dist, η, ll)
end

type Cluster{T,S<:AbstractMatrix}
    active::Bool
    first_appeared::Int
    split_from::Int
    wf::S
    η::Vector{T}
    sumη::T
    sum::Vector{T}
    comoments::Matrix{T}
    comoment_chol::Cholesky{T,Matrix{T}}
    indexes::Vector{Int}
    dist::MvTDist
    scratch::Vector{T}
end

function Cluster{T}(ν::T, wf::AbstractMatrix{T}, indexes::Vector{Int}, first_appeared::Int, split_from::Int)
	@assert issorted(indexes)
    n = length(indexes)
    cluster_wf = wf[:, indexes]

    dist, η = fit_tdist(cluster_wf, ν)
    sumη = sum(η)

    Cluster(true, first_appeared, split_from, wf, η, sumη, dist.μ*sumη, dist.Σ.mat*sumη,
            Cholesky{T,Matrix{T}}(dist.Σ.chol.factors*sqrt(sumη), 'U'),
            indexes, dist, zeros(T, size(wf, 1)))
end

"""
    refit!(cluster::Cluster)

Refit distribution based on current parameters, returning log likelihood
"""
function refit!{T}(cluster::Cluster{T})
    cluster_wf = cluster.wf[:, cluster.indexes]
    dist, η, ll = fit_tdist(cluster_wf, cluster.dist.df; μ_init=cluster.dist.μ, Σ_init=cluster.dist.Σ)
    sumη = sum(η)
    cluster.η = η
    cluster.sumη = sumη
    cluster.sum = dist.μ*sumη
    cluster.comoments = dist.Σ.mat*sumη
    cluster.comoment_chol = Cholesky{T,Matrix{T}}(dist.Σ.chol.factors*sqrt(sumη), 'U')
    cluster.dist = dist
    ll
end

"""
    updatedist!(cluster::Cluster)

Update distribution parameters based on our forms, which are more convenient for online updates
"""
function updatedist!(cluster)
    n = cluster.sumη
    broadcast!(*, cluster.dist.μ, cluster.sum, 1/n)
    broadcast!(*, cluster.dist.Σ.mat, cluster.comoments, 1/n)
    broadcast!(*, cluster.dist.Σ.chol.factors, cluster.comoment_chol.factors, 1/sqrt(n))
    # mu = sum(cluster.wf[:, cluster.indexes].*cluster.η', 2)/cluster.sumη
    # demeaned = (cluster.wf[:, cluster.indexes] .- mu)
    # @assert isapprox(cluster.dist.μ, mu)
    # @assert isapprox(cluster.dist.Σ.mat, (demeaned.*cluster.η')*demeaned'/cluster.sumη)
    # @assert isapprox(cluster.dist.Σ.chol[:U], cholfact(cluster.dist.Σ.mat)[:U])

    return cluster
end

"""
    addspike!(cluster::Cluster, spike::AbstractVector)

Add spike to cluster
"""
function addspike!{T}(cluster::Cluster{T}, spike::AbstractVector, index::Int)
    scratch = cluster.scratch
    broadcast!(-, scratch, spike, cluster.dist.μ)
    mahal = invquad(cluster.dist.Σ, scratch)
    shdfhdim, v = Distributions.mvtdist_consts(cluster.dist)
    η = (cluster.dist.df + size(cluster.wf, 1))/(cluster.dist.df + mahal)

    oldsumη = cluster.sumη
    sumη = cluster.sumη += η
    sum_ = cluster.sum
    push!(cluster.indexes, index)
    push!(cluster.η, η)

    # Update sum and mean
    factor = sqrt(η*oldsumη/sumη)
    for i = 1:size(spike, 1)
        scratch[i] *= factor
        sum_[i] += η*spike[i]
    end

    # Update comoments
    LinAlg.copytri!(BLAS.syr!('U', one(T), scratch, cluster.comoments), 'U')
    lowrankupdate!(cluster.comoment_chol, scratch)

    # Update everything else
    updatedist!(cluster)
end

"""
    prune!(cluster::Cluster, wf::AbstractMatrix{T}, before::Real)

Remove old spikes from cluster
"""
function prune!{T}(cluster::Cluster{T}, before::Int)
    wf = cluster.wf
    sum_ = cluster.sum
    scratch = cluster.scratch
    indexes = cluster.indexes
    η = cluster.η
    for iprune = 1:searchsortedlast(indexes, before)
        n = length(indexes)

        # Make sure cluster is still full rank
        n > size(wf, 1)+1 || (cluster.active = false; return cluster) 

        spike = sub(wf, :, shift!(indexes))
        oldsumη = cluster.sumη
        spikeη = shift!(η)
        sumη = cluster.sumη -= spikeη
        factor = sqrt(spikeη*sumη/oldsumη)

        # Update sum and mean
        for i = 1:size(wf, 1)
            sum_[i] -= spikeη*spike[i]
            scratch[i] = (spike[i] - sum_[i]/sumη)*factor
        end

        # Update comoments
        LinAlg.copytri!(BLAS.syr!('U', -one(T), scratch, cluster.comoments), 'U')
        lowrankdowndate!(cluster.comoment_chol, scratch)
    end
    @assert minimum(indexes) > before

    # Update everything else
    updatedist!(cluster)
end

using PyPlot
function spikesort{T}(wf::AbstractMatrix{T}, times::Vector, params::SortParams)
    λ = 1e-6
    size(wf, 2) == length(times) || throw(ArgumentError("second dimension of wf must match times"))
    verbose = params.verbose
    local assignments_::Vector{Int}

    pca_in = similar(wf)
    rg = 1:searchsortedlast(times, first(times)+Int(params.history_period))
    s = sum(sub(wf, :, rg), 2)
    smax = maxabs(s)
    clf()
    n = length(rg)
    for j = rg, k = 1:size(wf, 1)
        pca_in[k, j] = (wf[k, j]*n-s[k])/smax
    end
    cur_first = 1
    for j = last(rg)+1:size(wf, 2)
        last_index = searchsortedlast(times, times[j]-Int(params.history_period))
        while cur_first < last_index
            for k = 1:size(wf, 1)
                s[k] -= wf[k, cur_first]
            end
            cur_first += 1
        end
        n = j-cur_first+1
        for k = 1:size(wf, 1)
            s[k] += wf[k, j]
        end
        smax = maxabs(s)
        for k = 1:size(wf, 1)
            pca_in[k, j] = (wf[k, j]*n-s[k])/smax
        end
    end
    C = Base.covzm(pca_in, vardim=2)
    vals, vecs = eig(Symmetric(C))
    reverse!(vals)
    vecs = flipdim(vecs, 2)
    nkeep = searchsortedfirst(cumsum(vals./sum(vals)), 0.9)
    verbose && println("Using $nkeep PCs")
    keepvecs = vecs[:, 1:nkeep]
    cluster_space::Matrix{T} = keepvecs'*pca_in
    # Drop spikes that are more than 10 SDs away from the mean along any dimension
    # Chebyshev's inequality guarantees this drops no more than 1% of spikes, but
    # in practice it will be far less.
    sd = std(cluster_space, 2)
    mu = mean(cluster_space, 2)
    dropped = Set()
    for j = 1:size(cluster_space, 2), i = 1:size(cluster_space, 1)
        if abs(cluster_space[i, j] - mu[i]) > 10*sd[i]
            cluster_space[i, j] = mu[i]
            push!(dropped, j)
        end
    end
    verbose && println("Dropping $(length(dropped)) spikes >10 SD away")
    broadcast!(-, cluster_space, cluster_space, minimum(cluster_space, 2))
    broadcast!(/, cluster_space, cluster_space, maximum(cluster_space, 2))

    # Initialize using free split-merge GMM
    verbose && println("Initializing Clusters...")
    init_rg = 1:searchsortedlast(times, first(times)+Int(params.init_period))
    init_wft = cluster_space[:, init_rg]
    m = TMixture(params.initial_clusters, init_wft; λ=λ, use_uniform=true)::TMixture{T,Matrix{T}}
    m = fsmem!(m, maxiter=500)::TMixture{T,Matrix{T}}

    # Drop clusters that are too small
    assignments_ = assignments(m)
    init_nclusters = length(m.components)
    ν = m.components[1].df
    verbose && println("Using $init_nclusters clusters")
    counts = hist(assignments_, 0:init_nclusters)[2]
    valid = counts .>= params.min_cluster_size
    if !all(valid)
        # Drop low-count clusters
        remap = zeros(Int, init_nclusters)
        remap[valid] = 1:sum(valid)
        assignments_ = remap[assignments_]
        init_nclusters = sum(valid)
        verbose && println("Dropping $(sum(!valid)) clusters with fewer than $(params.min_cluster_size) spikes")
    end
    counts = hist(assignments_, 0:init_nclusters)[2]
    verbose && println("Cluster counts: $counts")

    # Compute sum and comoment for each cluster
    clusters = Array(Cluster{T,typeof(cluster_space)}, init_nclusters)
    for icluster = 1:length(clusters)
        indexes = find(assignments_ .== icluster)
        clusters[icluster] = Cluster(ν, cluster_space, indexes, 1, -1)
    end

    noise_indexes = find(assignments_ .== 0)+1
    resize!(assignments_, size(wf, 2))

    if params.plot
        hp = HoboPlotter(params.movie_file)
        plotclusters(hp, wf[:, init_rg], sub(assignments_, init_rg), Int(params.history_period))
    end


    iinspect = 0
    next_inspect = zero(eltype(times))
    prune_index = 0
    for ispike = last(init_rg)+1:size(wf, 2)
        if ispike in dropped
            assignments_[ispike] = 0
            continue
        end

        # TODO fix Distributions so this can use SubArrays
        cur_wf = cluster_space[:, ispike]
        cur_time = times[ispike]
        last_prune_index = prune_index
        prune_index = searchsortedlast(times, cur_time - Int(params.history_period))

        # See if we need to split/merge clusters
        if cur_time >= next_inspect
            println("Inspecting...")
            iinspect += 1
            rg = last_prune_index+1:ispike-1
            rg_wf = wf[:, rg]
            rg_wft = cluster_space[:, rg]

            active_clusters = Int[]
            for icluster = 1:length(clusters)
                if clusters[icluster].active
                    push!(active_clusters, icluster)
                end
            end
            dists = MvTDist[clusters[icluster].dist for icluster in active_clusters]
            logπ = Float64[length(clusters[icluster].indexes) for icluster in active_clusters]
            nspikes = sum(logπ)
            for i = 1:length(logπ)
                logπ[i] = log(logπ[i]/nspikes)
            end
            refit_rg = rg[assignments_[rg] .!= 0]
            m = TMixture(dists, cluster_space[:, refit_rg]; logπ=logπ, λ=λ, fix_ν=true, use_uniform=false)
            em!(m)
            for icluster = 1:length(active_clusters)
                cluster = clusters[active_clusters[icluster]]
                cluster.indexes = refit_rg[assignments(m) .== icluster]
                refit!(cluster)
            end

            if params.plot
                plotclusters(hp, rg_wf, sub(assignments_, rg), Int(params.history_period))
            end

            # Try splitting clusters
            changed = false
            more_iter = true
            while more_iter
                more_iter = false
                for icluster = 1:length(clusters)
                    oldclust = clusters[icluster]
                    indexes = oldclust.indexes
                    !oldclust.active && continue
                    length(indexes) <= params.min_cluster_size && continue
                    wft = rg_wft[:, findin(rg, indexes)]

                    ll = refit!(oldclust)
                    mtest = em!(TMixture(typeof(oldclust.dist)[deepcopy(oldclust.dist)], wft; λ=λ, fix_ν=true, use_uniform=false))
                    msplit = split(mtest)

                    if msplit !== mtest && (score(msplit) - score(mtest)) > params.split_score_improvement
                        assign = assignments(msplit)
                        assign1 = assign .== 1
                        assign2 = assign .== 2
                        if sum(assign1) >= params.min_cluster_size && sum(assign2) >= params.min_cluster_size
                            changed = more_iter = true
                            if maxabs(msplit.components[1].μ) < maxabs(msplit.components[2].μ)
                                splitclust = Cluster(ν, cluster_space, indexes[assign1], iinspect, oldclust.split_from)
                                newclust = Cluster(ν, cluster_space, indexes[assign2], iinspect, icluster)
                            else
                                splitclust = Cluster(ν, cluster_space, indexes[assign2], iinspect, oldclust.split_from)
                                newclust = Cluster(ν, cluster_space, indexes[assign1], iinspect, icluster)
                            end
                            clusters[icluster] = splitclust
                            push!(clusters, newclust)
                            assignments_[newclust.indexes] = length(clusters)
                            if verbose
                                println("Splitting cluster $icluster")
                                println("    new cluster $(length(clusters)) has $(length(newclust.indexes)) waveforms")
                                println("    old cluster $(icluster) has $(length(splitclust.indexes)) waveforms")
                                println("    score $(score(mtest)) -> $(score(msplit))")
                            end
                        end
                    end
                end

                # Try merging clusters
                for iclust1 = 1:length(clusters)
                    clust1 = clusters[iclust1]
                    !clust1.active && continue
                    for iclust2 = iclust1+1:length(clusters)
                        clust2 = clusters[iclust2]
                        !clust2.active && continue

                        ind1 = findin(rg, clust1.indexes)
                        ind2 = findin(rg, clust2.indexes)
                        rgindexes = [ind1; ind2]
                        wft = rg_wft[:, rgindexes]

                        # mix = Mixture([MvTDist(10., clust1.μ, deepcopy(clust1.Σ)),
                        #                MvTDist(10., clust2.μ, deepcopy(clust2.Σ))])
                        # γ_1 = zeros(length(rgindexes))
                        # γ_2 = zeros(length(rgindexes))
                        # γ_1[1:length(clust1.indexes)] = 1
                        # γ_2[length(clust1.indexes)+1:end] = 1
                        # mtest = em!(SoftMixtureFit(mix, wft; λ=0.01, γ=Vector{Float64}[γ_1, γ_2]))
                        # mtest = em!(HardMixtureFit(mix, wft; λ=λ,
                        #                       assignments=[fill(UInt8(1), length(clust1.indexes)); fill(UInt8(2), length(clust2.indexes))]))
                        mtest = em!(TMixture(typeof(clust1.dist)[deepcopy(clust1.dist), deepcopy(clust2.dist)], wft;
                                    λ=λ, logπ=[log(length(ind1)/length(rgindexes)), log(length(ind2)/length(rgindexes))], fix_ν=true, use_uniform=false))
                        mmerged = merge(mtest)

                        if mmerged !== mtest && (score(mmerged) - score(mtest)) > params.merge_score_improvement
                            changed = more_iter = true
                            newclust = Cluster(ν, cluster_space, sort!([clust1.indexes; clust2.indexes]), iinspect, -1)
                            if maxabs(clust1.dist.μ) < maxabs(clust2.dist.μ)
                                clusters[iclust1] = newclust
                                clust2.active = false
                                inewclust = iclust1
                                ideadclust = iclust2
                            else
                                clusters[iclust2] = newclust
                                clust1.active = false
                                inewclust = iclust2
                                ideadclust = iclust1
                            end
                            assignments_[clusters[ideadclust].indexes] = inewclust
                            if verbose
                                println("Merging cluster $ideadclust into $inewclust")
                                println("    cluster $iclust1 (split from $(clust1.split_from)) had $(length(clust1.indexes)) waveforms")
                                println("    cluster $iclust2 (split from $(clust2.split_from)) had $(length(clust2.indexes)) waveforms")
                                println("    score $(score(mtest)) -> $(score(mmerged))")
                            end
                            if clust1.split_from == iclust2 || clust2.split_from == iclust1
                                if clust1.split_from == iclust2
                                    split_periods = iinspect - clust1.first_appeared
                                    ioriginal = iclust2
                                    isplit = iclust1
                                else
                                    split_periods = iinspect - clust2.first_appeared
                                    ioriginal = iclust1
                                    isplit = iclust2
                                end
                                println("    split for $split_periods periods")
                                if split_periods < params.min_separation_periods
                                    println("    min number of separation periods is $(params.min_separation_periods), so re-merging")
                                    assignments_[assignments_ .== isplit] = ioriginal
                                end
                            end
                            break;
                        end
                    end
                end
            end

            next_inspect = cur_time+Int(params.inspect_period)
        end

        # Find best cluster
        deleteat!(noise_indexes, 1:searchsortedlast(noise_indexes, prune_index))
        best_cluster = 0
        nspikes = ispike - last_prune_index - 1
        best_logpdf = log((length(noise_indexes)+1)/(nspikes+1))
        for icluster = 1:length(clusters)
            cluster = clusters[icluster]
            !cluster.active && continue
            logpdf_ = logpdf(cluster.dist, cur_wf) + log(length(cluster.indexes)/(nspikes+1))
            prune!(cluster, prune_index)
            if logpdf_ > best_logpdf
                best_cluster = icluster
                best_logpdf = logpdf_
            end
        end

        # Add to cluster
        assignments_[ispike] = best_cluster
        if best_cluster == 0
            push!(noise_indexes, ispike)
        else
            addspike!(clusters[best_cluster], cur_wf, ispike)
        end
    end

    params.plot && finish!(hp)
    assignments_
end

function plot_assignments(wf, times, assignments_, time_step::TimePeriod=Minute(10); movie_file="")
    hp = HoboPlotter(movie_file)
    time_step_seconds = Int(Second(time_step))
    for t = 0:time_step_seconds:last(times)
        rg = searchsortedfirst(times, t):searchsortedlast(times, t+time_step_seconds)
        plotclusters(hp, wf[:, rg], sub(assignments_, rg), times[last(rg)]-times[first(rg)])
    end
end

end # module
