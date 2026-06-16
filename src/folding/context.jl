@inline function _fold_requested(params::Union{Nothing,HPRLP_parameters})
    return params !== nothing && normalize_folding_mode(params.folding) == "GPU"
end

function _device_ready(params::HPRLP_parameters)
    if !CUDA.functional()
        return false, "CUDA is not functional"
    end
    num_devices = length(CUDA.devices())
    if params.device_number < 0 || params.device_number >= num_devices
        return false, "invalid GPU device number $(params.device_number), valid range is [0, $(num_devices - 1)]"
    end
    return true, ""
end

# Context reuse keeps the GPU model/workspace ownership explicit at the API
# boundary. Workspace buffers are resized only when the LP shape changes.
mutable struct FoldingContext
    device_number::Int
    device_model::Any
    workspace::Any
end

FoldingContext(device_number::Int) = FoldingContext(device_number, nothing, nothing)

function _ensure_context!(
    ctx::Union{Nothing,FoldingContext},
    model::LP_info_gpu,
    device_number::Int,
)
    if ctx === nothing || ctx.device_number != device_number
        ctx = FoldingContext(device_number)
    end
    CUDA.device!(device_number)
    ctx.device_model = model
    ctx.workspace = _ensure_workspace!(ctx.workspace, size(model.A, 1), size(model.A, 2))
    return ctx
end
