module Extract
using Interpolations, DSP
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
function find_alignment_point(t::DoubleThreshold, v::AbstractVector)
    # Go up to the first zero crossing
    if v[1] < 0
        minind = 1
        minval = v[1]
        i = 2
        while i <= length(v)
            if v[i] <= minval
                minind = i
                minval = v[i]
            elseif v[i] > 0
                break
            end
            i += 1
        end
        i += 1
        i > length(v) && return minind
        maxind = i
        maxval = v[i]
        while i <= length(v)
            if v[i] >= maxval
                maxind = i
                maxval = v[i]
            elseif v[i] < 0
                break
            end
            i += 1
        end
    else
        maxind = 1
        maxval = v[1]
        i = 2
        while i <= length(v)
            if v[i] >= maxval
                maxind = i
                maxval = v[i]
            elseif v[i] < 0
                break
            end
            i += 1
        end
        i += 1
        i > length(v) && return maxind
        minind = i
        minval = v[i]
        while i <= length(v)
            if v[i] <= minval
                minind = i
                minval = v[i]
            elseif v[i] > 0
                break
            end
            i += 1
        end
    end
    if maxval > abs(minval)
        maxind
    else
        minind
    end
end

function extract_wf(x, thr_type, pre, post, align; upsample::Int=1, aligndur::Int=post, refractory::Int=post)
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
    tmp = zeros(eltype(x), upsample*aligndur+1)
    if upsample != 1
        interpolated = interpolate(x, BSpline(Cubic(Flat())), OnGrid())
    else
        interpolated = interpolate(zeros(eltype(x), 2), BSpline(Cubic(Flat())), OnGrid())
    end
    wf = zeros(eltype(x), pre+post+1, nwf)
    times = zeros(Int, nwf)
    oldnwf = nwf # Might detect fewer waveforms due to alignment
    nwf = 0
    i = pre+1
    while i <= nx-post
        if apply_threshold(thr_type, x[i])
            nwf += 1

            if align
                if upsample != 1
                    for isample = 1:upsample*aligndur+1
                        tmp[isample] = interpolated[i+(isample-1)/upsample]
                    end
                    pt = i+(find_alignment_point(thr_type, tmp)-1)/upsample
                    for iwf = 1:pre+post+1
                        wf[iwf, nwf] = interpolated[pt+iwf-1-pre]
                    end
                    i = ceil(Int, pt)
                else
                    copy!(tmp, 1, x, i, aligndur+1)
                    i += ceil(Int, find_alignment_point(thr_type, tmp)) - 1
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

            # Do not allow a lower point than the alignment point in
            # the waveform, since that is a sign that this is doubled
            lowerpt = false
            for iwf = 1:pre+post+1
                lowerpt = lowerpt || (apply_threshold(thr_type, wf[iwf, nwf]) && abs(wf[iwf, nwf]) > abs(wf[pre+1, nwf]))
            end
            if lowerpt
                nwf -= 1
                continue
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

function compute_mteo(x, ks)
    tmp = zeros(eltype(x), length(x)+2*maximum(ks))
    out = zeros(eltype(x), length(x))
    @inbounds for ik = 1:length(ks)
        k = ks[ik]
        # Compute TEO
        @simd for i = k+1:length(x)-k
            tmp[i] = abs2(x[i]) - x[i-k]*x[i+k]
        end

        # Apply Hamming window filter
        h = hamming(4*k+1)
        h /= sqrt(3*sumabs2(h)+sum(h)^2) # Eq. (13)
        filt!(tmp, h, tmp)

        # Compute output
        @simd for i = 1:length(x)
            out[i] = max(out[i], tmp[i+2k])
        end
    end
    out
end

function find_closest_extremum(x, istart, dt)
    irextremum = istart+zero(dt)
    rextremum = abs(x[istart])
    while irextremum < length(x) && abs(x[irextremum+dt]) > rextremum && sign(x[irextremum+dt]) == sign(rextremum)
        irextremum += dt
        rextremum = abs(x[irextremum])
    end
    ilextremum = istart+zero(dt)
    lextremum = abs(x[istart])
    while ilextremum > 1 && abs(x[ilextremum-dt]) > lextremum && sign(x[ilextremum-dt]) == sign(lextremum)
        ilextremum -= dt
        lextremum = abs(x[ilextremum])
    end
    pt = lextremum > rextremum ? ilextremum : irextremum
end

# function find_closest_extremum(x, istart, dt)
#     irextremum = istart+zero(dt)
#     sgn = sign(x[istart-dt] - x[istart])
#     while irextremum < length(x) && sign(x[irextremum] - x[irextremum+dt]) == sgn
#         irextremum += dt
#     end
#     ilextremum = istart+zero(dt)
#     sgn = sign(x[istart+dt] - x[istart])
#     while ilextremum > 1 && sign(x[ilextremum] - x[ilextremum-dt]) == sgn
#         ilextremum -= dt
#     end
#     pt = (istart - ilextremum) < (irextremum - istart) ? ilextremum : irextremum
# end

function extract_wf_teo(x, thr_type, pre, post, align; upsample::Int=1, aligndur::Int=post, refractory::Int=post)
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
    mteo = compute_mteo(x, (1, 4, 7, 10, 13, 16))
    tmp = zeros(eltype(x), upsample*aligndur+1)
    if upsample != 1
        interpolated = interpolate(x, BSpline(Cubic(Flat())), OnGrid())
    else
        interpolated = interpolate(zeros(eltype(x), 2), BSpline(Cubic(Flat())), OnGrid())
    end
    wf = zeros(eltype(x), pre+post+1, nwf)
    times = zeros(Int, nwf)
    oldnwf = nwf # Might detect fewer waveforms due to alignment
    nwf = 0
    i = pre+1
    while i <= nx-post
        if apply_threshold(thr_type, x[i])
            nwf += 1

            if align
                # First find (first) peak of TEO
                imaxteo = i
                while imaxteo < i+aligndur && mteo[imaxteo+1] > mteo[imaxteo]
                    imaxteo += 1
                end

                # Now find closest local extremum
                if upsample != 1
                    pt = find_closest_extremum(interpolated, imaxteo, 1/upsample)
                    for iwf = 1:pre+post+1
                        wf[iwf, nwf] = interpolated[pt+iwf-1-pre]
                    end
                    i = ceil(Int, pt)
                else
                    i = find_closest_extremum(x, imaxteo, 1)
                end
            end

            if !align || upsample == 1
                for iwf = 1:pre+post+1
                    wf[iwf, nwf] = mteo[i+iwf-1-pre]
                end
            end

            times[nwf] = i
            i += refractory - 1
            while i <= nx-post && apply_threshold(thr_type, x[i])
                i += 1
            end

            # Do not allow a lower point than the alignment point in
            # the waveform, since that is a sign that this is doubled
            lowerpt = false
            for iwf = 1:pre+post+1
                lowerpt = lowerpt || (apply_threshold(thr_type, wf[iwf, nwf]) && abs(wf[iwf, nwf]) > abs(wf[pre+1, nwf]))
            end
            if lowerpt
                nwf -= 1
                continue
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

end # module
