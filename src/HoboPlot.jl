module HoboPlot
using PyPlot, PyCall, MultivariateStats, Interpolations
@pyimport matplotlib.colors as mpl_colors
@pyimport matplotlib.animation as mpl_animation
export HoboPlotter, plotclusters, finish!

function gca_size_pixels()
    bbox = gca()[:get_window_extent]()[:transformed](gcf()[:dpi_scale_trans][:inverted]())
    dpi = gcf()[:dpi]
    (ceil(Int, bbox[:width]*dpi)::Int, ceil(Int, bbox[:height]*dpi)::Int)
end

function densitymap(X, sz::Tuple{Int,Int}=gca_size_pixels(); kws...)
    width = sz[1]
    height = sz[2]

    interpolated = interpolate(X, BSpline(Linear()), OnCell())
    counts = zeros(Int, height, width)
    minval, maxval = extrema(X)
    if minval > 20 || maxval > 20
        minval = trunc(minval/10)*10-10
        maxval = trunc(maxval/10)*10+10
    else
        minval = trunc(minval)-1
        maxval = trunc(maxval)+1
    end
    ls = linspace(1, size(X, 1), width)
    for j = 1:size(X, 2)
        v = round(Int, (interpolated[1, j]-minval)/(maxval-minval)*(height-1)+1)
        counts[v, 1] += 1
        for i = 2:width
            vlast = v
            v = round(Int, (interpolated[ls[i], j]-minval)/(maxval-minval)*(height-1)+1)
            for k = min(v, vlast + sign(v - vlast)):max(v, vlast + sign(v - vlast))
                counts[k, i] += 1
            end
        end
    end
    logcounts = log10(counts)
    imshow(logcounts./maximum(logcounts), interpolation="nearest", aspect="auto",
           extent=(0.5, size(X, 1)+0.5, minval-0.5, maxval+0.5), origin="lower"; kws...)
end

function linearcmap(rgb::NTuple{3})
    cdict = Dict("red"   => ((0.0, 1.0, 1.0),
                             (1.0, rgb[1], rgb[1])),
                 "green" => ((0.0, 1.0, 1.0),
                             (1.0, rgb[2], rgb[2])),
                 "blue"  => ((0.0, 1.0, 1.0),
                             (1.0, rgb[3], rgb[3])))
    mpl_colors.LinearSegmentedColormap("linearcmap", cdict)
end

immutable HoboPlotter
    movie_writer
    function HoboPlotter(movie_file)
        figure(figsize=(14, 8))
        if movie_file != ""
            writer = mpl_animation.FFMpegWriter(fps=2, bitrate=10000, extra_args=["-vcodec", "png"])
            writer[:setup](gcf(), movie_file, gcf()[:dpi])
        else
            writer = nothing
        end
        new(writer)
    end
end

function plotclusters(hp::HoboPlotter, rg, rg_wf, clusters, assignments_, time)
    cluster_colors = "bgrcmykbgrcmykbgrcmykbgrcmyk"
    colordict = mpl_colors.ColorConverter[:colors]
    active_indices = find(x->x.active, clusters)
    active_clusters = clusters[active_indices]
    nwf = sum(x->length(x.indexes), active_clusters)
    pca = fit(PCA, rg_wf, pratio=1.0, maxoutdim=2)
    proj = projection(pca)
    for i = 1:size(proj, 2)
        proj_i = proj[:, i]
        if abs(maximum(proj_i)) > abs(minimum(proj_i))
            proj[:, i] = scale!(proj_i, -1.0)
        end
    end
    pcs = transform(pca, rg_wf)

    clf()
    bax = subplot2grid((3, length(active_indices)), (0, 0), colspan=length(active_indices))
    for iclust = 1:length(active_clusters)
        cluster_color = string(cluster_colors[iclust])
        inds = find(sub(assignments_, rg) .== active_indices[iclust])
        sca(bax)
        scatter(pcs[1, inds], pcs[2, inds], 1, marker=".", color=cluster_color)
        subplot2grid((3, length(active_indices)), (1, iclust-1))
        densitymap(sub(rg_wf, :, inds), cmap=linearcmap(colordict[cluster_color]))
        hz = @sprintf("%0.1f", length(inds)/time)
        title("Cluster $(active_indices[iclust]) ($hz Hz)")
        subplot2grid((3, length(active_indices)), (2, iclust-1))
        imshow(active_clusters[iclust].Σ, interpolation="nearest")
        colorbar()
    end
    draw()
    if hp.movie_writer !== nothing
        hp.movie_writer[:grab_frame]()
    end
end

finish!(hp::HoboPlotter) = hp.movie_writer != nothing && hp.movie_writer[:finish]()
end