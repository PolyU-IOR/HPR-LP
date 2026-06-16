# LP folding by color refinement. This is intentionally separate from presolve:
# it aggregates symmetric row/column, solves the quotient LP, then unfolds the
# reduced solution back to the original dimensions.

const MAX_FOLDING_REDUCE_SIZE = 0.8

# CPU map returned after folding. `row_scale`/`col_scale` are inverse color sizes
# used when expanding reduced dual/slack vectors.
struct FoldingMap
    original_nrow::Int
    original_ncol::Int
    row_color_id::Vector{Int}
    col_color_id::Vector{Int}
    row_scale::Vector{Float64}
    col_scale::Vector{Float64}
end

# Device-side map kept until the final color ids and scales are copied to CPU.
struct DeviceFoldingMap
    original_nrow::Int
    original_ncol::Int
    row_color_id::CuVector{Int32}
    col_color_id::CuVector{Int32}
    row_scale::CuVector{Float64}
    col_scale::CuVector{Float64}
end

folding_active(map::FoldingMap) =
    map.original_nrow != length(unique(map.row_color_id)) ||
    map.original_ncol != length(unique(map.col_color_id))

folding_active(map::DeviceFoldingMap) =
    map.original_nrow != length(unique(Array(map.row_color_id))) ||
    map.original_ncol != length(unique(Array(map.col_color_id)))

const VALID_FOLDING_MODES = ("GPU", "NONE")

function normalize_folding_mode(mode)
    mode_name = if mode isa Bool
        mode ? "GPU" : "NONE"
    else
        uppercase(String(mode))
    end
    mode_name in VALID_FOLDING_MODES || throw(ArgumentError(
        "Unsupported folding mode $(mode). Expected one of GPU, NONE."))
    return mode_name
end

function set_folding_mode!(params::HPRLP_parameters, mode)
    params.folding = normalize_folding_mode(mode)
    return params
end

folding_enabled(params::HPRLP_parameters) = normalize_folding_mode(params.folding) != "NONE"
