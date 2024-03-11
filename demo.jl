using ImageFiltering, FFTW, LinearAlgebra, Profile, Random
using ComputationalResources

FFTW.set_num_threads(parse(Int, get(ENV, "FFTW_NUM_THREADS", "1")))
BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_NUM_THREADS", string(Threads.nthreads() ÷ 2))))

function benchmark_new(mats)
    kernel = ImageFiltering.factorkernel(Kernel.LoG(1))
    Threads.@threads for mat in mats
        frame_filtered = deepcopy(mat[:, :, 1])
        r_cached = CPU1(ImageFiltering.planned_fft(frame_filtered, kernel))
        for i in axes(mat, 3)
            frame = @view mat[:, :, i]
            imfilter!(r_cached, frame_filtered, frame, kernel)
        end
        return
    end
end
function benchmark_old(mats)
    kernel = ImageFiltering.factorkernel(Kernel.LoG(1))
    Threads.@threads for mat in mats
        frame_filtered = deepcopy(mat[:, :, 1])
        r_noncached = CPU1(Algorithm.FFT())
        for i in axes(mat, 3)
            frame = @view mat[:, :, i]
            imfilter!(r_noncached, frame_filtered, frame, kernel)
        end
        return
    end
end

function test(mats)
    kernel = ImageFiltering.factorkernel(Kernel.LoG(1))
    for mat in mats
        f1 = deepcopy(mat[:, :, 1])
        r_cached = CPU1(ImageFiltering.planned_fft(f1, kernel))
        f2 = deepcopy(mat[:, :, 1])
        r_noncached = CPU1(Algorithm.FFT())
        for i in axes(mat, 3)
            imfilter!(r_noncached, f2, deepcopy(mat[:, :, i]), kernel)
            imfilter!(r_cached, f1, deepcopy(mat[:, :, i]), kernel)
            @show f1[1:4] f2[1:4]
            f1 ≈ f2 || error("f1 !≈ f2")
        end
        return
    end
end

function run()
    Random.seed!(1)
    nmats = 10
    mats = [rand(Float64, rand(80:100), rand(80:100), rand(2000:3000)) for _ in 1:nmats]

    benchmark_new(mats)
    for _ in 1:3
        @time "warm run of benchmark_new(mats)" benchmark_new(mats)
    end

    benchmark_old(mats)
    for _ in 1:3
        @time "warm run of benchmark_old(mats)" benchmark_old(mats)
    end

    test(mats)
end

run()
