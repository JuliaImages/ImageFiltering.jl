# Min max filter

# This is a port of the Lemire min max filter as implemented by Bruno Luong
# http://arxiv.org/abs/cs.DS/0610046
# http://lemire.me/
# http://www.mathworks.com/matlabcentral/fileexchange/24705-min-max-filter

type Wedge{A <: AbstractArray}
    buffer::A
    size::Int
    n::Int
    first::Int
    last::Int
    mxn::Int
end


for N = 2:4
    @eval begin
    function extrema_filter{T <: Number}(A::Array{T, $N}, window::Array{Int, 1})

        maxval_temp = copy(A); minval_temp = copy(A)

        for dim = 1:$N

            # For all but the last dimension
            @nloops $(N-1) i maxval_temp begin

                # Create index for full array (fa) length
                @nexprs $(N)   j->(fa_{j} = 1:size(maxval_temp)[j])
                @nexprs $(N-1) j->(fa_{j} = i_j)

                # Create index for short array (sa) length
                @nexprs $(N)   j->(sa_{j} = 1:size(maxval_temp)[j] - window[dim] + 1)
                @nexprs $(N-1) j->(sa_{j} = i_j)

                # Filter the last dimension
                (@nref $N minval_temp sa) = min_filter(vec( @nref $N minval_temp fa), window[dim])
                (@nref $N maxval_temp sa) = max_filter(vec( @nref $N maxval_temp fa), window[dim])

            end

            # Circular shift the dimensions
            maxval_temp = permutedims(maxval_temp, mod(collect(1:$N), $N)+1)
            minval_temp = permutedims(minval_temp, mod(collect(1:$N), $N)+1)

        end

        # The dimensions to extract
        @nexprs $N j->(a_{j} = 1:size(A, j)-window[j]+1)

        # Extract set dimensions
        maxval_out = @nref $N maxval_temp a
        minval_out = @nref $N minval_temp a

        return minval_out, maxval_out
    end
    end
end


function extrema_filter{T <: Number}(a::AbstractArray{T}, window::Int)

    n = length(a)

    # Initialise the output variables
    # This is the running minimum and maximum over the specified window length
    minval = zeros(T, 1, n-window+1)
    maxval = zeros(T, 1, n-window+1)

    # Initialise the internal wedges
    # U[1], L[1] are the location of the global maximum and minimum
    # U[2], L[2] are the maximum and minimum over (U1, inf)
    L = Wedge(zeros(Int,1,window+1), window+1, 0, 1, 0, 0)          # Min
    U = Wedge(zeros(Int,1,window+1), window+1, 0, 1, 0, 0)

    for i = 2:n
        if i > window
            if !wedgeisempty(U)
                maxval[i-window] = a[getfirst(U)]
            else
                maxval[i-window] = a[i-1]
            end
            if !wedgeisempty(L)
                minval[i-window] = a[getfirst(L)]
            else
                minval[i-window] = a[i-1]
            end
        end # window

        if a[i] > a[i-1]
            pushback!(L, i-1)
            if i==window+getfirst(L); L=popfront(L); end
            while !wedgeisempty(U)
                if a[i] <= a[getlast(U)]
                    if i == window+getfirst(U); U = popfront(U); end
                    break
                end
                U = popback(U)
            end

        else

            pushback!(U, i-1)
            if i==window+getfirst(U); U=popfront(U); end

            while !wedgeisempty(L)
                if a[i] >= a[getlast(L)]
                    if i == window+getfirst(L); L = popfront(L); end
                    break
                end
                L = popback(L)
            end

        end  # a>a-1

    end # for i

    i = n+1
    if !wedgeisempty(U)
        maxval[i-window] = a[getfirst(U)]
    else
        maxval[i-window] = a[i-1]
    end

    if !wedgeisempty(L)
        minval[i-window] = a[getfirst(L)]
    else
        minval[i-window] = a[i-1]
    end

    return minval, maxval
end


function min_filter(a::AbstractArray, window::Int)

    minval, maxval = extrema_filter(a, window)

    return minval
end


function max_filter(a::AbstractArray, window::Int)

    minval, maxval = extrema_filter(a, window)

    return maxval
end


function wedgeisempty(X::Wedge)
    X.n <= 0
end

function pushback!(X::Wedge, v)
    X.last += 1
    if X.last > X.size
        X.last = 1
    end
    X.buffer[X.last] = v
    X.n = X.n+1
    X.mxn = max(X.mxn, X.n)
end

function getfirst(X::Wedge)
    X.buffer[X.first]
end

function getlast(X::Wedge)
    X.buffer[X.last]
end

function popfront(X::Wedge)
    X.n = X.n-1
    X.first = mod(X.first, X.size) + 1
    return X
end

function popback(X::Wedge)
    X.n = X.n-1
    X.last = mod(X.last-2, X.size) + 1
    return X
end
