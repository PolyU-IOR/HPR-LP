# LP folding by color refinement. This is intentionally separate from presolve:
# it aggregates symmetric row/column, solves the quotient LP, then unfolds the
# reduced solution back to the original dimensions.

include("folding/types.jl")
include("folding/context.jl")
include("folding/kernels.jl")
include("folding/refinement.jl")
include("folding/reduce.jl")
include("folding/map.jl")
include("folding/api.jl")
