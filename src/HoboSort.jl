module HoboSort
include("Extract.jl")
include("Mixtures.jl")
include("cholutils.jl")
using Base.Dates, Base.LinAlg, Distributions, PDMats, MultivariateStats, PyPlot, .Extract, .Mixtures

immutable SortParams
    init_period::Second
    inspect_period::Second
    history_period::Second
    min_cluster_size::Int
    max_clusters::Int
    verbose::Bool
    plot::Bool
end
SortParams(; init_period::TimePeriod=Minute(10), inspect_period::TimePeriod=Minute(5),
           history_period::TimePeriod=Minute(10), min_cluster_size=120, max_clusters::Integer=10,
           verbose::Bool=false, plot::Bool=false) = SortParams(init_period, inspect_period,
                                                               history_period, min_cluster_size,
                                                               max_clusters, verbose, plot)

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

    # Update everything else
    updatedist!(cluster)
end

function cluster_space(x)
    # x = convert(Matrix{eltype(x)}, x)
    # pca = fit(PCA, x, pratio=0.9, maxoutdim=3)
    # pcs = transform(pca, x)
    # pcs'
    x
end

function spikesort{T}(wf::AbstractMatrix{T}, times::Vector, params::SortParams)
    size(wf, 2) == length(times) || throw(ArgumentError("second dimension of wf must match times"))
    verbose = params.verbose
    local assignments_::Vector{Int}
    local means::Matrix{T}
    local covars::Array{T,3}

    # Initialize using a GMM
    verbose && println("Initializing Clusters...")
    init_rg = 1:searchsortedlast(times, first(times)+Int(params.init_period))
    init_wf = sub(wf, :, init_rg)
    init_wft = cluster_space(init_wf)
    m = @time fsmem!(init_wft, verbose=:op)
    # return m.m.components
    assignments_ = assignments(m)
    init_nclusters = length(m.m.components)
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
        clusters[icluster] = Cluster(wf, indexes)
    end

    resize!(assignments_, size(wf, 2))

    next_inspect = zero(eltype(times))
    prune_index = 1
    for ispike = last(init_rg)+1:size(wf, 2)
        # TODO fix Distributions so this can use SubArrays
        cur_wf = wf[:, ispike]
        cur_time = times[ispike]
        last_prune_index = prune_index
        prune_index = searchsortedlast(times, cur_time - Int(params.history_period))

        # See if we need to split/merge clusters
        if cur_time >= next_inspect
            println("Inspecting...")
            rg = last_prune_index:ispike
            rg_wft = cluster_space(wf[:, rg])

            # Try splitting clusters
            changed = false
            more_iter = true
            while more_iter
                more_iter = false
                for icluster = 1:length(clusters)
                    oldclust = clusters[icluster]
                    indexes = oldclust.indexes
                    !oldclust.active && continue
                    wft = rg_wft[:, findin(rg, indexes)]

                    # This is stupid, since it's a one component GMM
                    # mtest = em!(SoftMixtureFit(Mixture(size(wft, 1), 1), wft; λ=0.01, γ=Vector{Float64}[ones(length(indexes))]))
                    mtest = em!(HardMixtureFit(Mixture(size(wft, 1), 1), wft; λ=0.01, assignments=ones(UInt8, length(indexes))))
                    msplit = split(mtest, verbose=:op)

                    if msplit !== mtest
                        assign = assignments(msplit)
                        assign1 = assign .== 1
                        assign2 = assign .== 2
                        if sum(assign1) >= params.min_cluster_size && sum(assign2) >= params.min_cluster_size
                            changed = more_iter = true
                            splitclust = Cluster(wf, indexes[assign1])
                            newclust = Cluster(wf, indexes[assign2])
                            if sumabs2(newclust.μ - oldclust.μ) < sumabs2(splitclust.μ - oldclust.μ)
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

                        # γ_1 = zeros(length(rgindexes))
                        # γ_2 = zeros(length(rgindexes))
                        # γ_1[1:length(clust1.indexes)] = 1
                        # γ_2[length(clust1.indexes)+1:end] = 1
                        # mtest = em!(SoftMixtureFit(Mixture(size(wft, 1), 2), wft; λ=0.01, γ=Vector{Float64}[γ_1, γ_2]))
                        mtest = em!(HardMixtureFit(Mixture(size(wft, 1), 2), wft; λ=0.01,
                                              assignments=[fill(UInt8(1), length(clust1.indexes)); fill(UInt8(2), length(clust2.indexes))]))
                        mmerged = merge(mtest, verbose=:op)

                        if mmerged !== mtest
                            changed = more_iter = true
                            newclust = Cluster(wf, [clust1.indexes; clust2.indexes])
                            if sumabs2(newclust.μ - clust1.μ) < sumabs2(newclust.μ - clust2.μ)
                                clusters[iclust1] = newclust
                                clusters[iclust2].active = false
                                inewclust = iclust1
                                ideadclust = iclust2
                            else
                                clusters[iclust2] = newclust
                                clusters[iclust1].active = false
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
                colors = "bgrcmykbgrcmykbgrcmykbgrcmyk"
                active_indices = find(x->x.active, clusters)
                active_clusters = clusters[active_indices]
                nwf = sum(x->length(x.indexes), active_clusters)
                tmpwf = wf[:, prune_index:ispike-1]
                pca = fit(PCA, tmpwf, pratio=1.0, maxoutdim=2)
                pcs = transform(pca, tmpwf)

                clf()
                bax = subplot2grid((3, length(active_indices)), (0, 0), colspan=length(active_indices))
                for iclust = 1:length(active_clusters)
                    inds = find(assignments_[prune_index:ispike-1] .== active_indices[iclust])
                    sca(bax)
                    scatter(pcs[1, inds], pcs[2, inds], 1, marker=".", color=string(colors[iclust]))
                    subplot2grid((3, length(active_indices)), (1, iclust-1))
                    plot(tmpwf[:, sample(inds, 1000)], color=string(colors[iclust]))
                    title("Cluster $(active_indices[iclust])")
                    subplot2grid((3, length(active_indices)), (2, iclust-1))
                    imshow(active_clusters[iclust].Σ, interpolation="nearest")
                    colorbar()
                end
                draw()
            end

            next_inspect = cur_time+Int(params.inspect_period)
        end

        # Find best cluster
        best_cluster = 0
        best_logpdf = -Inf
        nspikes = ispike - last_prune_index
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
        addspike!(clusters[best_cluster], cur_wf, ispike)
    end
    assignments_
end

end # module
