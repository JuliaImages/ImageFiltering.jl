# Deprecated

"""
    gabor(size_x,size_y,σ,θ,λ,γ,ψ) -> (k_real,k_complex)

Returns a 2 Dimensional Complex Gabor kernel contained in a tuple where

  - `size_x`, `size_y` denote the size of the kernel
  - `σ` denotes the standard deviation of the Gaussian envelope
  - `θ` represents the orientation of the normal to the parallel stripes of a Gabor function
  - `λ` represents the wavelength of the sinusoidal factor
  - `γ` is the spatial aspect ratio, and specifies the ellipticity of the support of the Gabor function
  - `ψ` is the phase offset

# Citation

N. Petkov and P. Kruizinga, “Computational models of visual neurons specialised in the detection of periodic and aperiodic oriented visual stimuli: bar and grating cells,” Biological Cybernetics, vol. 76, no. 2, pp. 83–96, Feb. 1997. doi.org/10.1007/s004220050323
"""
function gabor(size_x::Integer, size_y::Integer, σ::Real, θ::Real, λ::Real, γ::Real, ψ::Real)
    Base.depwarn("use `Kernel.Gabor` instead.", :gabor)
    σx = σ
    σy = σ/γ
    nstds = 3
    c = cos(θ)
    s = sin(θ)

    validate_gabor(σ,λ,γ)

    if(size_x > 0)
        xmax = floor(Int64,size_x/2)
    else
        @warn "The input parameter size_x should be positive. Using size_x = 6 * σx + 1 (Default value)"
        xmax = round(Int64,max(abs(nstds*σx*c),abs(nstds*σy*s),1))
    end

    if(size_y > 0)
        ymax = floor(Int64,size_y/2)
    else
        @warn "The input parameter size_y should be positive. Using size_y = 6 * σy + 1 (Default value)"
        ymax = round(Int64,max(abs(nstds*σx*s),abs(nstds*σy*c),1))
    end

    xmin = -xmax
    ymin = -ymax

    x = [j for i in xmin:xmax,j in ymin:ymax]
    y = [i for i in xmin:xmax,j in ymin:ymax]
    xr = x*c + y*s
    yr = -x*s + y*c

    kernel_real = (exp.(-0.5*(((xr.*xr)/σx^2) + ((yr.*yr)/σy^2))).*cos.(2*(π/λ)*xr .+ ψ))
    kernel_imag = (exp.(-0.5*(((xr.*xr)/σx^2) + ((yr.*yr)/σy^2))).*sin.(2*(π/λ)*xr .+ ψ))

    kernel = (kernel_real,kernel_imag)
    return kernel
end

function validate_gabor(σ::Real,λ::Real,γ::Real)
    if !(σ>0 && λ>0 && γ>0)
        throw(ArgumentError("The parameters σ, λ and γ must be positive numbers."))
    end
end
