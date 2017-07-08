module MapWindow

using DataStructures, TiledIteration
using ..ImageFiltering: BorderSpecAny, Pad, Fill, borderinstance, _interior, padindex, imfilter
using Base: Indices, tail

export mapwindow

"""
    mapwindow(f, img, window, [border="replicate"]) -> imgf

Apply `f` to sliding windows of `img`, with window size or indices
specified by `window`. For example, `mapwindow(median!, img, window)`
returns an `Array` of values similar to `img` (median-filtered, of
course), whereas `mapwindow(extrema, img, window)` returns an `Array`
of `(min,max)` tuples over a window of size `window` centered on each
point of `img`.

The function `f` receives a buffer `buf` for the window of data
surrounding the current point. If `window` is specified as a
Dims-tuple (tuple-of-integers), then all the integers must be odd and
the window is centered around the current image point. For example, if
`window=(3,3)`, then `f` will receive an Array `buf` corresponding to
offsets `(-1:1, -1:1)` from the `imgf[i,j]` for which this is
currently being computed. Alternatively, `window` can be a tuple of
AbstractUnitRanges, in which case the specified ranges are used for
`buf`; this allows you to use asymmetric windows if needed.

`border` specifies how the edges of `img` should be handled; see
`imfilter` for details.

For functions that can only take `AbstractVector` inputs, you might have to
first specialize `default_shape`:

```julia
f = v->quantile(v, 0.75)
ImageFiltering.MapWindow.default_shape(::typeof(f)) = vec
```

and then `mapwindow(f, img, (m,n))` should filter at the 75th quantile.

See also: [`imfilter`](@ref).
"""
function mapwindow(f, img::AbstractArray, window::Dims, args...; kwargs...)
    all(isodd(w) for w in window) || error("entries in window must be odd, got $window")
    halfsize = map(w->w>>1, window)
    mapwindow(f, img, map(h->-h:h, halfsize), args...; kwargs...)
end
function mapwindow(f, img::AbstractVector, window::Integer, args...; kwargs...)
    isodd(window) || error("window must be odd, got $window")
    h = window>>1
    mapwindow(f, img, (-h:h,), args...; kwargs...)
end

mapwindow(f, img::AbstractArray, window::Indices; kwargs...) =
    mapwindow(f, img, window, "replicate"; kwargs...)
mapwindow(f, img::AbstractVector, window::AbstractUnitRange; kwargs...) =
    mapwindow(f, img, (window,); kwargs...)

function mapwindow(f, img::AbstractArray, window::Indices, border::AbstractString;
                   kwargs...)
    mapwindow(f, img, window, borderinstance(border); kwargs...)
end
function mapwindow(f, img::AbstractVector, window::AbstractUnitRange, border::AbstractString;
                   kwargs...)
    mapwindow(f, img, (window,), border; kwargs...)
end

mapwindow(f, img, window::AbstractArray, args...; kwargs...) = mapwindow(f, img, (window...,), args...; kwargs...)



function mapwindow{T,N}(f,
                        img::AbstractArray{T,N},
                        window::Indices{N},
                        border::BorderSpecAny;
                        callmode=:copy!)
    if(uses_histogram(f))
        median_filter(replace_function(f), img, window, border, default_shape(f); callmode=callmode)
    else
        _mapwindow(replace_function(f), img, window, border, default_shape(f); callmode=callmode)
    end 

end
function _mapwindow{T,N}(f,
                         img::AbstractArray{T,N},
                         window::Indices{N},
                         border::BorderSpecAny,
                         shape=default_shape(f);
                         callmode=:copy!)
    inds = indices(img)
    inner = _interior(inds, window)
    if callmode == :copy!
        buf = Array{T}(map(length, window))
        bufrs = shape(buf)
        Rbuf = CartesianRange(size(buf))
        offset = CartesianIndex(map(w->first(w)-1, window))
        # To allocate the output, we have to evaluate f once
        Rinner = CartesianRange(inner)
        if !isempty(Rinner)
            Rwin = CartesianRange(map(+, window, first(Rinner).I))
            copy!(buf, Rbuf, img, Rwin)
            out = similar(img, typeof(f(bufrs)))
            # Handle the interior
            for I in Rinner
                Rwin = CartesianRange(map(+, window, I.I))
                copy!(buf, Rbuf, img, Rwin)
                out[I] = f(bufrs)
            end
        else
            copy_win!(buf, img, first(CartesianRange(inds)), border, offset)
            out = similar(img, typeof(f(bufrs)))
        end
        # Now pick up the edge points we skipped over above
        for I in EdgeIterator(inds, inner)
            copy_win!(buf, img, I, border, offset)
            out[I] = f(bufrs)
        end
    else
        # TODO: implement :view
        error("callmode $callmode not supported")
    end
    out
end

function dept_iter{T,N}(f,img::AbstractArray{T,N},buf,bufrs,Rbuf,offset,Rwin,out,level,depth,indice,window::Indices{N},
                         border::BorderSpecAny,
                         shape=default_shape(f);callmode=:copy!)
    inds = indices(img)

    inner = _interior(inds, window)
    Rinner=CartesianRange(inner)
    buf = Array{T}(map(length, window))
    bufrs = shape(buf)
    Rbuf = CartesianRange(size(buf))
    offset = CartesianIndex(map(w->first(w)-1, window))

    if(level==depth)

#        println(img[:,:,indice])
#        println( CartesianRange(img[:,:,indice]))

        inds = indices(img)
        inner = _interior(inds, window)
        if callmode == :copy!
            Rinner = CartesianRange(inner)            
            mode =0
            m_histogram=zeros(Int64,(256,))

            if !isempty(Rinner)
                R2=last(Rinner).I[2]
                L1=first(Rinner).I[1]
                R1=first(Rinner).I[2]
                L2=last(Rinner).I[1]

                prev_mode= 0
                # Handle the interior
                m_histogram=zeros(Int64,(256,))
                for i=L1:L2
                    if (i-L1)%2==0
                        temp1=(i,R1)
                        temp2=(i,R2)
                        id1=(temp1...,indice...)
                        id2=(temp2...,indice...)
                        RinnerN= CartesianRange(CartesianIndex(id1), CartesianIndex(id2))
                        #println(RinnerN)
                        for I in RinnerN
                            #println(I)
                            Rwin = CartesianRange(map(+, window, I.I))
                            #println(Rwin)
                            copy!(buf, Rbuf, img, Rwin)
                            if (I[2]== R2 && prev_mode==1)
                                out[I] = f(bufrs,m_histogram,3,window)
                                prev_mode=3
                                continue
                            elseif (I[2]==R1 && prev_mode==2) 
                                out[I] = f(bufrs,m_histogram,4,window)
                                prev_mode=4
                                continue
                            elseif prev_mode == 3
                                out[I] = f(bufrs,m_histogram,5,window)
                                prev_mode = 2
                                continue
                            elseif prev_mode == 4

                                out[I] = f(bufrs,m_histogram,6,window)
                                prev_mode = 1
                                continue 
                            elseif prev_mode == 0
                                out[I] = f(bufrs,m_histogram,0,window)
                                prev_mode=1
                                continue
                            elseif prev_mode == 1
                                out[I] = f(bufrs,m_histogram,1,window)
                                prev_mode=1
                                continue
                            elseif prev_mode ==2
                                out[I] = f(bufrs,m_histogram,2,window)
                                prev_mode=2
                                continue
                            end
                        end

                    else

                        for k= R2:-1:R1

                            I=CartesianIndex(tuple((i,k)...,indice...))
                            Rwin = CartesianRange(map(+, window, I.I))
                            copy!(buf, Rbuf, img, Rwin)
                            if (I[2]== R2 && prev_mode==1)
                                out[I] = f(bufrs,m_histogram,3,window)
                                prev_mode=3
                                continue
                            elseif (I[2]==R1 && prev_mode==2) 
                                out[I] = f(bufrs,m_histogram,4,window)
                                prev_mode=4
                                continue
                            elseif prev_mode == 3
                                out[I] = f(bufrs,m_histogram,5,window)
                                prev_mode = 2
                                continue
                            elseif prev_mode == 4
                                out[I] = f(bufrs,m_histogram,6,window)
                                prev_mode = 1
                                continue 
                            elseif prev_mode == 0
                                out[I] = f(bufrs,m_histogram,0,window)
                                prev_mode=2
                                continue
                            elseif prev_mode == 1
                                out[I] = f(bufrs,m_histogram,1,window)
                                prev_mode=1
                                continue
                            elseif prev_mode == 2
                                out[I] = f(bufrs,m_histogram,2,window)
                                prev_mode=2
                                continue
                            end
                        end

                    end
                
                end
            else
                copy_win!(buf, img, first(CartesianRange(inds)), border, offset)
                out = similar(img, typeof(f(bufrs,m_histogram,mode,window)))
            end
            # Now pick up the edge points we skipped over above
            for I in EdgeIterator(inds, inner)
                # Handle the edge points with mode 0
                #mode =0
                #m_histogram=zeros(Int64,(256,))
                #copy_win!(buf, img, I, border, offset)
                #out[I] = f(bufrs,m_histogram,mode,window)
                out[I]=0.0
            end
            
        else
                # TODO: implement :view
            error("callmode $callmode not supported")
        end

        return
    else
        for i = 2:size(img,ndims(img)-level)-1
            #println(i)
            indice[ndims(img)-level-2]=i
            dept_iter(f,img,buf,bufrs,Rbuf,offset,Rwin,out,level+1,depth,indice,window,border,shape)

        end
    end
end


function iterate_points{T,N}(f,img::AbstractArray{T,N},window::Indices{N},
                         border::BorderSpecAny,
                         shape=default_shape(f);callmode=:copy!)
    
    depth= ndims(img)-2
    indice=zeros(Int64,(depth,))
    inds = indices(img)
    mode = 0
    m_histogram=zeros(Int64,(256,))


    inner = _interior(inds, window)
    Rinner=CartesianRange(inner)
    buf = Array{T}(map(length, window))
    bufrs = shape(buf)
    Rbuf = CartesianRange(size(buf))
    offset = CartesianIndex(map(w->first(w)-1, window))
            # To allocate the output, we have to evaluate f once
            
            # Initialise the mode to zero and histogram consisting of 255 bins to zeros


    Rwin = CartesianRange(map(+, window, first(Rinner).I))
    copy!(buf, Rbuf, img, Rwin)
    #out = similar(img, typeof(f(bufrs,m_histogram,mode,window)))
    out=similar(img)
    dept_iter(f,img,buf,bufrs,Rbuf,offset,Rwin,out,0,depth,indice,window,border,shape)
    return out

end




function median_filter{T,N}(f,
                         img::AbstractArray{T,N},
                         window::Indices{N},
                         border::BorderSpecAny,
                         shape=default_shape(f);
                         callmode=:copy!)
    inds = indices(img)
    inner = _interior(inds, window)
    if callmode == :copy!
        Rinner = CartesianRange(inner)
        buf = Array{T}(map(length, window))
        bufrs = shape(buf)
        Rbuf = CartesianRange(size(buf))
        offset = CartesianIndex(map(w->first(w)-1, window))
        # To allocate the output, we have to evaluate f once
        R2=last(Rinner).I[2]
        L1=first(Rinner).I[1]
        R1=first(Rinner).I[2]
        L2=last(Rinner).I[1]
        skipper= (R2-R1+1)*(L2-L1+1)
        # Initialise the mode to zero and histogram consisting of 255 bins to zeros
        counter=0
        mode =0
        m_histogram=zeros(Int64,(256,))
        if !isempty(Rinner)
            out = similar(img, typeof(f(bufrs,m_histogram,mode,window)))
            for I in Rinner
                if(counter%skipper==0)
                    flag=0
                    if(ndims(img)==2)
                        flag=1
                    end
                    indice=I.I[3:end]
                    #println(indice)
                    prev_mode= 0
                    # Handle the interior
                    m_histogram=zeros(Int64,(256,))
                    for i=L1:L2
                        if (i-L1)%2==0
                            temp1=(i,R1)
                            temp2=(i,R2)

                            if(flag==0)
                                id1=(temp1...,indice...)
                                id2=(temp2...,indice...)
                            else
                                id1=temp1
                                id2=temp2
                            end
                            RinnerN= CartesianRange(CartesianIndex(id1), CartesianIndex(id2))
                            #println(RinnerN)
                            for I in RinnerN
                                #println(I)
                                Rwin = CartesianRange(map(+, window, I.I))
                                
                                #println(Rwin)
                                copy!(buf, Rbuf, img, Rwin)
                                if (I[2]== R2 && prev_mode==1)
                                    out[I] = f(bufrs,m_histogram,3,window)
                                    prev_mode=3
                                    continue
                                elseif (I[2]==R1 && prev_mode==2) 
                                    out[I] = f(bufrs,m_histogram,4,window)
                                    prev_mode=4
                                    continue
                                elseif prev_mode == 3
                                    out[I] = f(bufrs,m_histogram,5,window)
                                    prev_mode = 2
                                    continue
                                elseif prev_mode == 4

                                    out[I] = f(bufrs,m_histogram,6,window)
                                    prev_mode = 1
                                    continue 
                                elseif prev_mode == 0
                                    out[I] = f(bufrs,m_histogram,0,window)
                                    prev_mode=1
                                    continue
                                elseif prev_mode == 1
                                    out[I] = f(bufrs,m_histogram,1,window)
                                    prev_mode=1
                                    continue
                                elseif prev_mode ==2
                                    out[I] = f(bufrs,m_histogram,2,window)
                                    prev_mode=2
                                    continue
                                end
                            end

                        else

                            for k= R2:-1:R1
                                if flag==1
                                    I=CartesianIndex((i,k))
                                else
                                    I=CartesianIndex(tuple((i,k)...,indice...))
                                end
                                Rwin = CartesianRange(map(+, window, I.I))
                                copy!(buf, Rbuf, img, Rwin)
                                if (I[2]== R2 && prev_mode==1)
                                    out[I] = f(bufrs,m_histogram,3,window)
                                    prev_mode=3
                                    continue
                                elseif (I[2]==R1 && prev_mode==2) 
                                    out[I] = f(bufrs,m_histogram,4,window)
                                    prev_mode=4
                                    continue
                                elseif prev_mode == 3
                                    out[I] = f(bufrs,m_histogram,5,window)
                                    prev_mode = 2
                                    continue
                                elseif prev_mode == 4
                                    out[I] = f(bufrs,m_histogram,6,window)
                                    prev_mode = 1
                                    continue 
                                elseif prev_mode == 0
                                    out[I] = f(bufrs,m_histogram,0,window)
                                    prev_mode=2
                                    continue
                                elseif prev_mode == 1
                                    out[I] = f(bufrs,m_histogram,1,window)
                                    prev_mode=1
                                    continue
                                elseif prev_mode == 2
                                    out[I] = f(bufrs,m_histogram,2,window)
                                    prev_mode=2
                                    continue
                                end
                            end

                        end
                    end

                end                        
                counter+=1
            end   
        
        else
            copy_win!(buf, img, first(CartesianRange(inds)), border, offset)
            out = similar(img, typeof(f(bufrs,m_histogram,mode,window)))
        end
        # Now pick up the edge points we skipped over above
        for I in EdgeIterator(inds, inner)
            # Handle the edge points with mode 0
            mode =0
            m_histogram=zeros(Int64,(256,))
            copy_win!(buf, img, I, border, offset)
            out[I] = f(bufrs,m_histogram,mode,window)
            #out[I]=0.0
        end
            
    else
                # TODO: implement :view
            error("callmode $callmode not supported")
    end
    out
end

function uses_histogram(::typeof(median!))
    true
end

function uses_histogram(::Any)
    false
end
    
# For copying along the edge of the image
function copy_win!{T,N}(buf::AbstractArray{T,N}, img, I, border::Pad, offset)
    win_inds = map(+, indices(buf), (I+offset).I)
    win_img_inds = map(intersect, indices(img), win_inds)
    padinds = map((inner,outer)->padindex(border, inner, outer), win_img_inds, win_inds)
    docopy!(buf, img, padinds)
    buf
end
docopy!(buf, img, padinds::NTuple{1}) = buf[:] = view(img, padinds[1])
docopy!(buf, img, padinds::NTuple{2}) = buf[:,:] = view(img, padinds[1], padinds[2])
docopy!(buf, img, padinds::NTuple{3}) = buf[:,:,:] = view(img, padinds[1], padinds[2], padinds[3])
@inline function docopy!{N}(buf, img, padinds::NTuple{N})
    @show N
    colons = ntuple(d->Colon(), Val{N})
    buf[colons...] = view(img, padinds...)
end

function copy_win!{T,N}(buf::AbstractArray{T,N}, img, I, border::Fill, offset)
    R = CartesianRange(indices(img))
    Ioff = I+offset
    for J in CartesianRange(indices(buf))
        K = Ioff+J
        buf[J] = K ∈ R ? img[K] : convert(eltype(img), border.value)
    end
    buf
end



### Optimizations for particular window-functions

mapwindow(::typeof(extrema), A::AbstractArray, window::Dims) = extrema_filter(A, window)
mapwindow(::typeof(extrema), A::AbstractVector, window::Integer) = extrema_filter(A, (window,))

# Max-min filter

# This is an implementation of the Lemire max-min filter
# http://arxiv.org/abs/cs.DS/0610046

# Monotonic wedge
immutable Wedge{T}
    L::CircularDeque{T}
    U::CircularDeque{T}
end
(::Type{Wedge{T}}){T}(n::Integer) = Wedge(CircularDeque{T}(n), CircularDeque{T}(n))

function Base.push!(W::Wedge, i::Integer)
    push!(W.L, i)
    push!(W.U, i)
    W
end

function addtoback!(W::Wedge, A, i, J)
    mn, mx = A[i, J]
    @inbounds while !isempty(W.L) && mn < A[back(W.L), J][1]
        pop!(W.L)
    end
    @inbounds while !isempty(W.U) && mx > A[back(W.U), J][2]
        pop!(W.U)
    end
    push!(W.L, i)
    push!(W.U, i)
    W
end

function Base.empty!(W::Wedge)
    empty!(W.L)
    empty!(W.U)
    W
end

@inline function getextrema(A, W::Wedge, J)
    (A[front(W.L), J][1], A[front(W.U), J][2])
end

"""
    extrema_filter(A, window) --> Array{(min,max)}

Calculate the running min/max over a window of width `window[d]` along
dimension `d`, centered on the current point. The returned array has
the same indices as the input `A`.
"""
function extrema_filter{T,N}(A::AbstractArray{T,N}, window::NTuple{N,Integer})
    _extrema_filter!([(a,a) for a in A], window...)
end
extrema_filter(A::AbstractArray, window::AbstractArray) = extrema_filter(A, (window...,))
extrema_filter(A::AbstractArray, window) = error("`window` must have the same number of entries as dimensions of `A`")

extrema_filter{T,N}(A::AbstractArray{T,N}, window::Integer) = extrema_filter(A, ntuple(d->window, Val{N}))

function _extrema_filter!(A::Array, w1, w...)
    if w1 > 1
        a = first(A)
        cache = ntuple(i->a, w1>>1)
        _extrema_filter1!(A, w1, cache)
    end
    _extrema_filter!(permutedims(A, [2:ndims(A);1]), w...)
end
_extrema_filter!(A::Array) = A

# Extrema-filtering along "columns" (dimension 1). This implements Lemire
# Algorithm 1, with the following modifications:
#   - multidimensional array support by looping over trailing dimensions
#   - working with min/max pairs rather than plain values, to
#     facilitate multidimensional processing
#   - output for all points of the array, handling the edges as max-min
#     over halfwindow on either side
function _extrema_filter1!{T}(A::AbstractArray{Tuple{T,T}}, window::Int, cache)
    # Initialise the internal wedges
    # U[1], L[1] are the location of the global (within the window) maximum and minimum
    # U[2], L[2] are the maximum and minimum over (U1, end] and (L1, end], respectively
    W = Wedge{Int}(window+1)
    tmp = Array{Tuple{T,T}}(window)
    c = z = first(cache)

    inds = indices(A)
    inds1 = inds[1]
    halfwindow = window>>1
    iw = min(last(inds1), first(inds1)+window-1)
    for J in CartesianRange(tail(inds))
        empty!(W)
        # Leading edge. We can't overwrite any values yet in A because
        # we'll need them again in later computations.
        for i = first(inds1):iw
            addtoback!(W, A, i, J)
            c, cache = cyclecache(cache, getextrema(A, W, J))
        end
        # Process the rest of the "column"
        for i = iw+1:last(inds1)
            A[i-window, J] = c
            if i == window+front(W.U)
                shift!(W.U)
            end
            if i == window+front(W.L)
                shift!(W.L)
            end
            addtoback!(W, A, i, J)
            c, cache = cyclecache(cache, getextrema(A, W, J))
        end
        for i = last(inds1)-window+1:last(inds1)-1
            if i >= first(inds1)
                A[i, J] = c
            end
            if i == front(W.U)
                shift!(W.U)
            end
            if i == front(W.L)
                shift!(W.L)
            end
            c, cache = cyclecache(cache, getextrema(A, W, J))
        end
        A[last(inds1), J] = c
    end
    A
end

# This is slightly faster than a circular buffer
@inline cyclecache(b, x) = b[1], (Base.tail(b)..., x)

function median_find(m_histogram,window_size)
    tempsum = 0
    m_index=-1
    for i = 1:256
        tempsum+= m_histogram[i]
        if tempsum>= window_size/2
            m_index=i-1
            break
        end
    end
    return convert(Float64,m_index)/255
end


function update_median(v_blob,m_histogram,mode,window,recursion_depth,indice,level)
    v = v_blob[:,:,indice]

    width=window[2].stop-window[2].start+1
    height= window[1].stop-window[1].start+1    
    v_reshape= reshape(v,(height,width))
    #v_reshape=v
    p=zeros(eltype(v_reshape), (size(v_reshape,2),size(v_reshape,1)))
    transpose!(p,v_reshape)
    v= reshape(p,(size(v,1)*size(v,2),))
    m_index=-1
    inds = indices(v,1)
    if mode == 0
        for i = first(inds):last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
    elseif mode == 1
        for i =  width:width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
    elseif mode == 2
        for i =  first(inds):width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
    elseif mode == 3 
        for i =  width:width:last(inds)            
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
    elseif mode ==4
        for i =  first(inds):width:last(inds)            
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
    elseif mode ==5
        for i =  last(inds)-width+1:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
    elseif mode ==6
        for i =  last(inds)-width+1:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
    end
end


function clear_histogram(v_blob,m_histogram,mode,window,recursion_depth,indice,level)
    v = v_blob[:,:,indice]
    width=window[2].stop-window[2].start+1
    height= window[1].stop-window[1].start+1    
    v_reshape= reshape(v,(height,width))
    p=zeros(eltype(v_reshape), (size(v_reshape,2),size(v_reshape,1)))
    transpose!(p,v_reshape)
#    v= reshape(p,size(v))
    v= reshape(p,(size(v,1)*size(v,2),))

    inds = indices(v,1)
    if mode == 0
        for i = first(inds):width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
    elseif mode == 1
        for i = first(inds):width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
    elseif mode == 2
        for i = width:width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
    elseif mode == 3 
        for i =  first(inds):first(inds)+width-1            
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
    elseif mode ==4
        for i =  first(inds):first(inds)+width-1
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
    elseif mode ==5
        for i =  width:width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
    elseif mode ==6
        for i =  first(inds):width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
    end

end



function depth_looper_delete(v_blob,m_histogram,mode,window,recursion_depth,indice,level)
    if level == recursion_depth
        clear_histogram(v_blob,m_histogram,mode,window,recursion_depth,indice,level)
    else
        for i = 1:size(v_blob,ndims(v_blob)-level)
            indice[ndims(v_blob)-level-2]=i
            depth_looper_delete(v_blob,m_histogram,mode,window,recursion_depth,indice,level+1)
        end
    end
end

function depth_looper_update(v_blob,m_histogram,mode,window,recursion_depth,indice,level)
    if level == recursion_depth
        update_median(v_blob,m_histogram,mode,window,recursion_depth,indice,level)
    else
        for i = 1:size(v_blob,ndims(v_blob)-level)
            indice[ndims(v_blob)-level-2]=i
            depth_looper_update(v_blob,m_histogram,mode,window,recursion_depth,indice,level+1)
        end
    end
end

replace_function(f) = f

replace_function(::typeof(median!)) = function(v,m_histogram,mode,window)
    # reshape the value of the window so that they are horizontal major which allows to do horizontal traversal
    
    window_size=size(v,1)
    dims = map(x->x.stop-x.start+1,window)
    startp= ones(Int64,(size(dims,1),))
    v_blob=reshape(v,dims)
    blob_range=CartesianRange(CartesianIndex(tuple(startp...)),CartesianIndex(tuple(dims...)))
#=    recursion_depth = size(window,1)-2
    indice = zeros(Int64,(recursion_depth,))
    #println(recursion_depth)
#    println("v_blob: ",size(v_blob))
    depth_looper_update(v_blob,m_histogram,mode,window,recursion_depth,indice,0)
    current_median=median_find(m_histogram,window_size)
    depth_looper_delete(v_blob,m_histogram,mode,window,recursion_depth,indice,0)
    println(current_median)
    current_median
=#
    skipper= size(v_blob,1)*size(v_blob,2)
    counter=0
    for I in blob_range
        if(counter%skipper==0)
            indice=I.I[3:end]
            if(ndims(v_blob)==2)
                v=v_blob
            else
                v=v_blob[:,:,indice...]
            end
        m_index=-1
        inds = indices(v,1)
        width=window[2].stop-window[2].start+1
        height= window[1].stop-window[1].start+1    
        v_reshape= reshape(v,(height,width))
        #v_reshape=v
        p=zeros(eltype(v_reshape), (size(v_reshape,2),size(v_reshape,1)))
        transpose!(p,v_reshape)
        v= reshape(p,(size(v,1)*size(v,2),))
        m_index=-1
        inds = indices(v,1)
            if mode == 0
                for i = first(inds):last(inds)
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]+=1
                end
            elseif mode == 1
                for i =  width:width:last(inds)
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]+=1
                end
            elseif mode == 2
                for i =  first(inds):width:last(inds)
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]+=1
                end
            elseif mode == 3 
                for i =  width:width:last(inds)            
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]+=1
                end
            elseif mode ==4
                for i =  first(inds):width:last(inds)            
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]+=1
                end
            elseif mode ==5
                for i =  last(inds)-width+1:last(inds)
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]+=1
                end
            elseif mode ==6
                for i =  last(inds)-width+1:last(inds)
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]+=1
                end
            end
        end
    end
# Find the median using histogram
    tempsum = 0
    m_index=-1
    for i = 1:256
        tempsum+= m_histogram[i]
        if tempsum>= window_size/2
            m_index=i-1
            break
        end
    end

#Clear histogram
counter=0
    for I in blob_range
        if(counter%skipper==0)
            indice=I.I[3:end]
            if(ndims(v_blob)==2)
                v=v_blob
            else
                v=v_blob[:,:,indice...]
            end
        width=window[2].stop-window[2].start+1
        height= window[1].stop-window[1].start+1    
        v_reshape= reshape(v,(height,width))
        p=zeros(eltype(v_reshape), (size(v_reshape,2),size(v_reshape,1)))
        transpose!(p,v_reshape)
        #    v= reshape(p,size(v))
        v= reshape(p,(size(v,1)*size(v,2),))

        inds = indices(v,1)
            if mode == 0
                for i = first(inds):width:last(inds)
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]-=1
                end
            elseif mode == 1
                for i = first(inds):width:last(inds)
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]-=1
                end
            elseif mode == 2
                for i = width:width:last(inds)
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]-=1
                end
            elseif mode == 3 
                for i =  first(inds):first(inds)+width-1            
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]-=1
                end
            elseif mode ==4
                for i =  first(inds):first(inds)+width-1
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]-=1
                end
            elseif mode ==5
                for i =  width:width:last(inds)
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]-=1
                end
            elseif mode ==6
                for i =  first(inds):width:last(inds)
                    id= trunc(Int64,(v[i]*255))+1
                    m_histogram[id]-=1
                end
            end
        end
    end



    return convert(Float64,m_index)/255



#=

    width=window[2].stop-window[2].start+1
    height= window[1].stop-window[1].start+1    
    v_reshape= reshape(v,(height,width))
    p=zeros(eltype(v_reshape), (size(v_reshape,2),size(v_reshape,1)))
    transpose!(p,v_reshape)
    v= reshape(p,size(v))
    width=3
    height=3
    m_index=-1
    inds = indices(v,1)
    if mode == 0
        for i = first(inds):last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
        tempsum = 0
        for i = 1:256
            tempsum+= m_histogram[i]
            if tempsum>= last(inds)/2
                m_index=i-1
                break
            end
        end
        for i = first(inds):width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
        return convert(Float64,m_index)/255
    elseif mode == 1
        for i =  width:width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
        tempsum = 0
        for i = 1:256
            tempsum+= m_histogram[i]
            if tempsum>= last(inds)/2
                m_index=i-1
                break
            end
        end
        for i = first(inds):width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
        return convert(Float64,m_index)/255
    elseif mode == 2
        for i =  first(inds):width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
        tempsum = 0
        for i = 1:256
            tempsum+= m_histogram[i]
            if tempsum>= last(inds)/2
                m_index=i-1
                break
            end
        end
        for i = width:width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
        return convert(Float64,m_index)/255
    elseif mode == 3 
        for i =  width:width:last(inds)            
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
        tempsum = 0
        for i = 1:256
            tempsum+= m_histogram[i]
            if tempsum>= last(inds)/2
                m_index=i-1
                break
            end
        end
        for i =  first(inds):first(inds)+width-1            
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
        return convert(Float64,m_index)/255
    elseif mode ==4
        for i =  first(inds):width:last(inds)            
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
        tempsum = 0
        for i = 1:256
            tempsum+= m_histogram[i]
            if tempsum>= last(inds)/2
                m_index=i-1
                break
            end
        end
        for i =  first(inds):first(inds)+width-1
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
        return convert(Float64,m_index)/255
    elseif mode ==5
        for i =  last(inds)-width+1:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
        tempsum = 0
        for i = 1:256
            tempsum+= m_histogram[i]
            if m_histogram[i]<0
                println("stop:",mode)
            end
            if tempsum>= last(inds)/2
                m_index=i-1
                break
            end
        end
        for i =  width:width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
        return convert(Float64,m_index)/255
    elseif mode ==6
        for i =  last(inds)-width+1:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]+=1
        end
        tempsum = 0
        for i = 1:256
            tempsum+= m_histogram[i]
            if tempsum>= last(inds)/2
                m_index=i-1
                break
            end
        end
        for i =  first(inds):width:last(inds)
            id= trunc(Int64,(v[i]*255))+1
            m_histogram[id]-=1
        end
        return convert(Float64,m_index)/255
    end
=#    
end

default_shape(::Any) = identity
default_shape(::typeof(median!)) = vec

end