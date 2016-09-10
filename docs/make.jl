using Documenter, ImagesFiltering

makedocs(modules  = [ImagesFiltering, Kernel, KernelFactors],
         format   = Documenter.Formats.HTML,
         sitename = "ImagesFiltering",
         pages    = ["index.md", "Function reference" => "function_reference.md"])

deploydocs(repo   = "github.com/JuliaImages/ImagesFiltering.jl.git")
         # deps   = Deps.pip("mkdocs", "python-markdown-math"),
