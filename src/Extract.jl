module Extract
using Interpolations, DSP, AxisAlgorithms, SpecialMatrices
export PositiveThreshold, NegativeThreshold, DoubleThreshold, extract_wf, extract_wf_teo, noise_autocovariance

immutable PositiveThreshold{T<:Real}
    thr::T
end
apply_threshold(t::PositiveThreshold, x) = x >= t.thr
find_alignment_point(t::PositiveThreshold, v::AbstractVector) = indmax(v)

immutable NegativeThreshold{T<:Real}
    thr::T
    NegativeThreshold(thr) = new(-abs(thr))
end
NegativeThreshold(thr::Real) = NegativeThreshold{typeof(thr)}(thr)
apply_threshold(t::NegativeThreshold, x) = x <= t.thr
find_alignment_point(t::NegativeThreshold, v::AbstractVector) = indmin(v)

immutable DoubleThreshold{T<:Real}
    posthr::T
    negthr::T
    DoubleThreshold(thr) = new(abs(thr), -abs(thr))
    DoubleThreshold(posthr, negthr) = new(abs(posthr), -abs(negthr))
end
DoubleThreshold(posthr::Real, negthr::Real) = DoubleThreshold{Base.promote_typeof(posthr, negthr)}(posthr, negthr)
DoubleThreshold(thr::Real) = DoubleThreshold(thr, thr)
apply_threshold(t::DoubleThreshold, x) = (x >= t.posthr) | (x <= t.negthr)

function find_closest_extremum(x, istart, dt, firstallowed, lastallowed)
    irextremum = istart+zero(dt)
    sgn = sign(x[istart-dt] - x[istart])
    while irextremum < lastallowed && sign(x[irextremum] - x[irextremum+dt]) == sgn
        irextremum += dt
    end
    ilextremum = istart+zero(dt)
    sgn = sign(x[istart+dt] - x[istart])
    while ilextremum > firstallowed && sign(x[ilextremum] - x[ilextremum-dt]) == sgn
        ilextremum -= dt
    end
    (istart - ilextremum) < (irextremum - istart) ? ilextremum : irextremum
end


immutable FastInterpolationUpdater{T,TCoefs,S}
    interp::Interpolations.BSplineInterpolation{T,1,TCoefs,BSpline{Cubic{Flat}},OnGrid,1}
    prefiltering_system::S
    tmp::TCoefs
end

function FastInterpolationUpdater(x)
    interp = interpolate(x, BSpline(Cubic(Flat())), OnGrid())
    prefiltering_system = Interpolations.prefiltering_system(Interpolations.tweight(x), Interpolations.tcoef(x), length(interp.coefs), Cubic{Flat}, OnGrid)
    FastInterpolationUpdater(interp, prefiltering_system, similar(interp.coefs))
end

function update_interpolation!(updater::FastInterpolationUpdater, x, rg)
    interp = updater.interp
    copy!(interp.coefs, 2:length(rg)+1, x, rg)
    interp.coefs[1] = interp.coefs[end] = 0
    W, b = updater.prefiltering_system
    R1 = R2 = CartesianRange(())
    dest = src = interp.coefs
    Interpolations._A_ldiv_B_md!(dest, W.A, src, R1, R2, b)
    tmp = updater.tmp
    fill!(tmp, 0)
    tmp[1] = W.Cp[1,1]*src[3]+W.Cp[1,2]*src[end-2]
    tmp[end] = W.Cp[2,1]*src[3]+W.Cp[2, 2]*src[end-2]
    Interpolations._A_ldiv_B_md!(tmp, W.A, tmp, R1, R2)
    AxisAlgorithms.sub!(dest, tmp)
    interp
end

function extract_wf(x, thr_type::Union{PositiveThreshold,NegativeThreshold}, pre, post, align; upsample::Int=1, refractory::Int=post)
    nx = length(x)
    i = pre+1
    nwf = 0
    @inbounds while i <= nx
        if apply_threshold(thr_type, x[i])
            nwf += 1
            i += refractory - 1
        end
        i += 1
    end
    tmp = zeros(eltype(x), upsample*(pre+post)+1)
    updater = FastInterpolationUpdater(zeros(eltype(x), 2*pre+2*post+1))
    interpolated = updater.interp
    wf = zeros(eltype(x), pre+post+1, nwf)
    times = zeros(Int, nwf)
    oldnwf = nwf # Might detect fewer waveforms due to alignment
    nwf = 0
    i = pre+1
    while i <= nx-post
        if apply_threshold(thr_type, x[i])
            nwf += 1

            oldi = -1
            if align
                oldi = i
                for iter = 1:10
                    if upsample != 1
                        interp_start = i-pre-post
                        interp_end = i+pre+post
                        update_interpolation!(updater, x, interp_start:interp_end)
                        for isample = 1:upsample*(pre+post+1)
                            tmp[isample] = interpolated[post+1+(isample-1)/upsample]
                        end
                        pt = interp_start+post+1+(find_alignment_point(thr_type, tmp)-1)/upsample
                        for iwf = 1:pre+post+1
                            wf[iwf, nwf] = interpolated[pt+iwf-1-pre]
                        end
                        i = round(Int, interp_start+pt-1)
                    else
                        copy!(tmp, 1, x, i-pre, pre+post+1)
                        i += find_alignment_point(thr_type, tmp) - 1
                    end

                    if i == oldi || i < max(initiali - pre, lastspike + refractory)
                        i = oldi
                        break
                    end
                end
            end

            if !align || upsample == 1
                for iwf = 1:pre+post+1
                    wf[iwf, nwf] = x[i+iwf-1-pre]
                end
            end

            times[nwf] = i
            i += refractory - 1
            while i <= nx-post && apply_threshold(thr_type, x[i])
                i += 1
            end
        end
        i += 1
    end
    if nwf < oldnwf
        wf = wf[:, 1:nwf]
        resize!(times, nwf)
    end
    wf, times
end

function extract_wf(x, thr_type::DoubleThreshold, pre, post, align; upsample::Int=1, refractory::Int=post)
    nx = length(x)
    i = pre+1
    nwf = 0
    @inbounds while i <= nx
        if apply_threshold(thr_type, x[i])
            nwf += 1
            i += refractory - 1
        end
        i += 1
    end
    updater = FastInterpolationUpdater(zeros(eltype(x), 2*pre+2*post+1))
    wf = zeros(eltype(x), pre+post+1, nwf)
    times = zeros(Int, nwf)
    grad = zeros(eltype(x), 1)
    oldnwf = nwf # Might detect fewer waveforms due to alignment
    nwf = 0
    lastspike = 0
    i = pre+1
    interpolated = updater.interp
    extrema = Tuple{Float64,eltype(x)}[]
    while i <= nx-post
        if apply_threshold(thr_type, x[i])
            nwf += 1

            if align
                initiali = i
                pt = 1/one(upsample)
                # Find the local extremum that corresponds to the highest TEO
                best = -Inf
                secondbest = -Inf
                for iter = 1:10 
                    oldi = i
                    interp_start = i-pre-post
                    interp_end = i+pre+post
                    update_interpolation!(updater, x, interp_start:interp_end)
                    empty!(extrema)
                    npos = 0
                    nneg = 0
                    for iext = post+1:1/upsample:2*post+pre+1
                        v = interpolated[iext]
                        !apply_threshold(thr_type, v) && continue
                        gradient!(grad, interpolated, iext-1/(2*upsample))
                        g1 = grad[1]
                        gradient!(grad, interpolated, iext+1/(2*upsample))
                        g2 = grad[1]
                        if sign(g1) != sign(g2)
                            push!(extrema, (iext, v))
                            if v > 0
                                npos += 1
                            else
                                nneg += 1
                            end
                        end
                    end
                    if npos == 1 && nneg > 1
                        # If multiple extrema on one side but only one on another, use the one
                        for (iext, v) in extrema
                            if v > 0
                                pt = iext
                                break
                            end
                        end
                    elseif nneg >= 1
                        # Use largest negative threshold crossing
                        mext = 0.0
                        for (iext, v) in extrema
                            if v < mext
                                pt = iext
                                mext = v
                            end
                        end
                    else
                        # Use the first positive crossing
                        pt = extrema[1][1]
                    end

                    i = round(Int, interp_start+pt-1)
                    if i == oldi || i < max(initiali - pre, lastspike + refractory)
                        i = oldi
                        break
                    end
                end

                for iwf = 1:pre+post+1
                    wf[iwf, nwf] = interpolated[pt+iwf-1-pre]
                end
            else
                for iwf = 1:pre+post+1
                    wf[iwf, nwf] = x[i+iwf-1-pre]
                end
            end

            lastspike = i
            times[nwf] = i
            i += refractory
            while i <= nx-post && apply_threshold(thr_type, x[i])
                i += 1
            end
        end
        i += 1
    end
    if nwf < oldnwf
        wf = wf[:, 1:nwf]
        resize!(times, nwf)
    end
    wf, #=wfteo, extrema,=# times
end

# Find segments of noise suitably far from a spike
function noise_autocovariance(x, thr_type, n, gap)
    acov = zeros(eltype(x), n)
    acovn = zeros(Int, n)
    i = 1
    nx = length(x)
    @inbounds while i <= nx
        j = i
        while j < nx && !apply_threshold(thr_type, x[j+1])
            j += 1
        end
        j -= gap
        for lag = 0:min(n-1, j-i)
            acov[lag+1] += dot(x, i:j-lag, x, i+lag:j)
            acovn[lag+1] += (j-i+1-lag)
        end
        i = j + 2*gap + 1
    end
    acov./acovn
end

whiten(fwf, acov) = sqrtm(full(Toeplitz([reverse(acov[2:end]); acov])))\fwf

end # module
