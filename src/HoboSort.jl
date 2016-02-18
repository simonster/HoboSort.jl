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
    min_cluster_size::Int
    initial_clusters::Int
    verbose::Bool
    plot::Bool
    movie_file::UTF8String
end
SortParams(; init_period::TimePeriod=Minute(10), inspect_period::TimePeriod=Minute(5),
           history_period::TimePeriod=Minute(10), min_cluster_size=120, initial_clusters::Integer=10,
           verbose::Bool=false, plot::Bool=false, movie_file::AbstractString="") =
	SortParams(init_period, inspect_period, history_period, min_cluster_size, initial_clusters,
		       verbose, plot, movie_file)

type Cluster{T,S<:AbstractMatrix}
    active::Bool
    wf::S
    sum::Vector{T}
    comoments::Matrix{T}
    comoment_chol::Cholesky{T,Matrix{T}}
    indexes::Vector{Int}
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    chol::Cholesky{T,Matrix{Float64}}
    dist::FullNormal
    scratch::Vector{T}
end

function Cluster{T}(wf::AbstractMatrix{T}, indexes::Vector{Int})
	@assert issorted(indexes)
    n = length(indexes)
    cluster_wf = wf[:, indexes]

    sum_ = vec(sum(cluster_wf, 2))
    μ = sum_/n

    broadcast!(-, cluster_wf, cluster_wf, μ)
    # cluster_wf is now mean-subtracted
    comoments = cluster_wf*cluster_wf'
    comoment_chol = cholfact(comoments)

    Σ = comoments/n
    chol = Cholesky{T,Matrix{T}}(comoment_chol.factors/sqrt(n), 'U')
    dist = MvNormal(μ, PDMat(Σ, chol))
    # @assert isapprox(Σ, cov(wf[:, indexes], vardim=2, corrected=false))
    # @assert isapprox(chol[:U], cholfact(Σ)[:U])

    Cluster(true, wf, sum_, comoments, comoment_chol, indexes, μ, Σ, chol, dist, similar(sum_))
end

"""
    updatedist!(cluster::Cluster)

Update distribution parameters based on our forms, which are more convenient for online updates
"""
function updatedist!(cluster)
    n = length(cluster.indexes)
    broadcast!(*, cluster.μ, cluster.sum, 1/n)
    broadcast!(*, cluster.Σ, cluster.comoments, 1/n)
    broadcast!(*, cluster.chol.factors, cluster.comoment_chol.factors, 1/sqrt(n))
    # @assert isapprox(cluster.μ, vec(mean(cluster.wf[:, cluster.indexes], 2)))
    # @assert isapprox(cluster.Σ, cov(cluster.wf[:, cluster.indexes], vardim=2, corrected=false))
    # @assert isapprox(cluster.chol[:U], cholfact(cluster.Σ)[:U])
    return cluster
end

"""
    addspike!(cluster::Cluster, spike::AbstractVector)

Add spike to cluster
"""
function addspike!{T}(cluster::Cluster{T}, spike::AbstractVector, index::Int)
    scratch = cluster.scratch
    sum_ = cluster.sum
    push!(cluster.indexes, index)
    n = length(cluster.indexes)

    # Update sum and mean
    factor = sqrt((n-1)/n)
    for i = 1:size(spike, 1)
        scratch[i] = (spike[i] - sum_[i]/(n-1))*factor
        sum_[i] += spike[i]
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
    for iprune = 1:searchsortedlast(indexes, before)
        n = length(indexes)

        # Make sure cluster is still full rank
        n > size(wf, 1)+1 || (cluster.active = false; return cluster) 

        spike = sub(wf, :, shift!(indexes))
        factor = sqrt((n-1)/n)

        # Update sum and mean
        for i = 1:size(wf, 1)
            sum_[i] -= spike[i]
            scratch[i] = (spike[i] - sum_[i]/(n-1))*factor
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
    local means::Matrix{T}
    local covars::Array{T,3}

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
    m = TMixture(params.initial_clusters, init_wft; λ=λ)
    m = fsmem!(m, verbose=:iter, maxiter=500)

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
    clusters = Array(Cluster{T}, init_nclusters)
    for icluster = 1:length(clusters)
        indexes = find(assignments_ .== icluster)
        clusters[icluster] = Cluster(cluster_space, indexes)
    end

    resize!(assignments_, size(wf, 2))

    if params.plot
        hp = HoboPlotter(params.movie_file)
        plotclusters(hp, init_rg, wf[:, init_rg], clusters, assignments_, Int(params.history_period))
    end

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
            rg = last_prune_index+1:ispike-1
            rg_wf = wf[:, rg]
            rg_wft = cluster_space[:, rg]

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

                    # This is stupid, since it's a one component GMM
                    # mix = Mixture([MvTDist(10., oldclust.μ, deepcopy(oldclust.Σ))])
                    # mtest = em!(SoftMixtureFit(mix, wft; λ=0.01, γ=Vector{Float64}[ones(length(indexes))]))
                    mtest = em!(TMixture(2, wft; λ=λ, ν=ν, fix_ν=true))
                    msplit = split(mtest)

                    if msplit !== mtest
                        assign = assignments(msplit)
                        assign1 = assign .== 1
                        assign2 = assign .== 2
                        if sum(assign1) >= params.min_cluster_size && sum(assign2) >= params.min_cluster_size
                            changed = more_iter = true
                            splitclust = Cluster(cluster_space, indexes[assign1])
                            newclust = Cluster(cluster_space, indexes[assign2])
                            if maxabs(newclust.μ) < maxabs(splitclust.μ)
                                splitclust, newclust = newclust, splitclust
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

                        rgindexes = [findin(rg, clust1.indexes); findin(rg, clust2.indexes)]
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
                        mtest = em!(TMixture(2, wft; λ=λ, ν=ν, fix_ν=true, centers=[clust1.μ clust2.μ]))
                        mmerged = merge(mtest)

                        if mmerged !== mtest
                            changed = more_iter = true
                            newclust = Cluster(cluster_space, sort!([clust1.indexes; clust2.indexes]))
                            if maxabs(clust1.μ) < maxabs(clust2.μ)
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
                                println("    cluster $iclust1 had $(length(clust1.indexes)) waveforms")
                                println("    cluster $iclust2 had $(length(clust2.indexes)) waveforms")
                                println("    score $(score(mtest)) -> $(score(mmerged))")
                            end
                            break;
                        end
                    end
                end
            end

            if params.plot
            	plotclusters(hp, rg, rg_wf, clusters, assignments_, Int(params.history_period))
            end

            next_inspect = cur_time+Int(params.inspect_period)
        end

        # Find best cluster
        best_cluster = 0
        best_logpdf = -Inf
        nspikes = ispike - last_prune_index - 1
        for icluster = 1:length(clusters)
            cluster = clusters[icluster]
            !cluster.active && continue
            logpdf_ = logpdf(cluster.dist, cur_wf) + log(length(cluster.indexes)/nspikes)
            prune!(cluster, prune_index)
            if logpdf_ > best_logpdf
                best_cluster = icluster
                best_logpdf = logpdf_
            end
        end

        # Add to cluster
        assignments_[ispike] = best_cluster
        if best_cluster != 0
            addspike!(clusters[best_cluster], cur_wf, ispike)
        end
    end

    params.plot && finish!(hp)
    assignments_
end

end # module
