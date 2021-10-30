using Documenter, DemoCards
using ImageFiltering
using OffsetArrays
# loading Plots before generating demos (might be helpful to avoid segfaults)
using Plots
ENV["GKSwstype"] = "nul" # needed for the GR backend on headless servers
gr()

demos, demos_cb, demos_assets = makedemos("demos")

assets = []
isnothing(demos_assets) || (push!(assets, demos_assets))
format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                         assets = assets)

makedocs(
    modules  = [ImageFiltering, OffsetArrays, Kernel, KernelFactors, ImageFiltering.MapWindow],
    format   = format,
    sitename = "ImageFiltering",
    pages    = [
        "index.md",
        demos,
        "Function reference" => "function_reference.md"
    ],
    # Note(johnnychen94): doctest is moved as part of unit test for two reasons:
    # - we want to run doctest for various Julia versions on various platforms.
    # - `doctest=true` for `makedocs` will trigger not only doctest of ImageFiltering but also
    #   that of other packages listed in `modules`. It usually throws warnings that don't
    #   belong to the scope of this package.
    doctest = false
)

demos_cb()

deploydocs(
    repo   = "github.com/JuliaImages/ImageFiltering.jl.git",
    # NOTE: remove the comment if the PR requires a push preview
    # push_preview = true
)
