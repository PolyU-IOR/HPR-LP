using Documenter
using HPRLP

makedocs(
    sitename = "HPRLP.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://PolyU-IOR.github.io/HPR-LP",
        assets = String[],
    ),
    modules = [HPRLP],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Input Methods" => [
                "Overview" => "guide/input_overview.md",
                "Direct API" => "guide/direct_api.md",
                "JuMP Integration" => "guide/jump_integration.md",
                "MPS Files" => "guide/mps_files.md",
            ],
            "Parameters" => "guide/parameters.md",
            "Output & Results" => "guide/output_results.md",
        ],
        "API Reference" => "api.md",
        "Examples" => "examples.md",
    ],
    doctest = false,  # Disable doctests to speed up build
    checkdocs = :none,  # Don't check for missing docstrings
)

deploydocs(
    repo = "github.com/PolyU-IOR/HPR-LP.git",
    devbranch = "main",
    push_preview = true,
    forcepush = true,
)
