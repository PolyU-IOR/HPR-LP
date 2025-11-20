# HPRLP Documentation

This directory contains the documentation for HPRLP.jl built using [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl).

## Building the Documentation Locally

To build the documentation on your local machine:

```bash
# From the root directory of HPR-LP
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The generated documentation will be in `docs/build/`.

To view it:
```bash
# Linux/Mac
open docs/build/index.html

# Or use Python's simple HTTP server
cd docs/build
python3 -m http.server 8000
# Then open http://localhost:8000 in your browser
```

## Documentation Structure

- `src/index.md` - Homepage and overview
- `src/getting_started.md` - Installation and first steps
- `src/guide/` - User guides for different interfaces
  - `mps_files.md` - Solving MPS files
  - `direct_api.md` - Using the direct matrix API
  - `jump_integration.md` - JuMP integration
- `src/api.md` - API reference
- `src/examples.md` - Complete examples

## Automatic Deployment

Documentation is automatically built and deployed to GitHub Pages when you:
- Push to `main` or `master` branch
- Create a new tag/release

The deployed documentation will be available at:
https://PolyU-IOR.github.io/HPR-LP/

## Adding New Pages

1. Create a new `.md` file in `src/`
2. Add it to the `pages` list in `make.jl`
3. Rebuild the documentation

## Writing Documentation

Documenter.jl supports:
- Markdown syntax
- LaTeX math with `$` and `$$`
- Julia code blocks with syntax highlighting
- Cross-references
- Docstring integration with `@docs`

For more information, see the [Documenter.jl documentation](https://documenter.juliadocs.org/).
