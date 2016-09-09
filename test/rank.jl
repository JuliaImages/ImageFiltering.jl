    context("Extrema_filter") do
        # 2d case
        A = zeros(5,5)
        A[2,2] = 0.8
        A[4,4] = 0.6
        minval, maxval = extrema_filter(A, 2)
        matching = [vec(A[2:end]) .!= vec(maxval); false]
        @fact A[reshape(matching, size(A))] --> [0.8, 0.6]
        # 3d case
        A = zeros(5,5,5)
        A[2,2,2] = 0.7
        A[4,4,3] = 0.5
        minval, maxval = extrema_filter(A, 2)
        matching = [vec(A[2:end]) .!= vec(maxval); false]
        @fact A[reshape(matching, size(A))] --> [0.7, 0.5]
        # 4d case
        A = zeros(5,5,5,5)
        A[2,2,2,2] = 0.7
        A[4,4,3,1] = 0.4
        A[3,4,3,2] = 0.5
        minval, maxval = extrema_filter(A, 2)
        matching = [vec(A[2:end]) .!= vec(maxval); false]
        @fact A[reshape(matching, size(A))] --> [0.4,0.7,0.5]
        x, y, z, t = ind2sub(size(A), find(A .== 0.4))
        @fact x[1] --> 4
        @fact y[1] --> 4
        @fact z[1] --> 3
        @fact t[1] --> 1
        # 2d case
        A = rand(5,5)/10
        A[2,2] = 0.8
        A[4,4] = 0.6
        minval, maxval = extrema_filter(A, [2, 2])
        matching = falses(A)
        matching[2:end, 2:end] = maxval .== A[2:end, 2:end]
        @fact sort(A[matching])[end-1:end] --> [0.6, 0.8]
        # 3d case
        A = rand(5,5,5)/10
        A[2,2,2] = 0.7
        A[4,4,2] = 0.4
        A[2,2,4] = 0.5
        minval, maxval = extrema_filter(A, [2, 2, 2])
        matching = falses(A)
        matching[2:end, 2:end, 2:end] = maxval .== A[2:end, 2:end, 2:end]
        @fact sort(A[matching])[end-2:end] --> [0.4, 0.5, 0.7]
        # 4d case
        A = rand(5,5,5,5)/10
        A[2,2,2,2] = 0.7
        A[4,4,2,3] = 0.4
        A[2,2,4,3] = 0.5
        minval, maxval = extrema_filter(A, [2, 2, 2, 2])
        matching = falses(A)
        matching[2:end, 2:end, 2:end, 2:end] = maxval .== A[2:end, 2:end, 2:end, 2:end]
        @fact sort(A[matching])[end-2:end] --> [0.4, 0.5, 0.7]
    end
