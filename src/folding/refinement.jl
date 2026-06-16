@inline function _reduction_too_small(num_row_color::Int, num_col_color::Int, num_row::Int, num_col::Int)
    return num_row_color > MAX_FOLDING_REDUCE_SIZE * num_row &&
           num_col_color > MAX_FOLDING_REDUCE_SIZE * num_col
end

# Initialize colors from static row/column data before graph-based refinement.
# Row colors start from bounds; column colors start from cost and variable bounds.
function _init_color!(
    device_model::LP_info_gpu,
    workspace::FoldingWorkspace,
    tolerance::Float64,
)
    workspace.num_row_color = 1
    workspace.num_col_color = 1
    row_color_sig = workspace.row_color_sig
    col_color_sig = workspace.col_color_sig
    fill!(workspace.row_color_id, Int32(0))
    fill!(workspace.col_color_id, Int32(0))
    _, workspace.num_row_color = _update_color_id!(
        workspace.row_color_id,
        workspace.row_color_start,
        workspace.row_perm,
        workspace.row_color,
        device_model.AL,
        tolerance,
    )
    _, workspace.num_row_color = _update_color_id!(
        workspace.row_color_id,
        workspace.row_color_start,
        workspace.row_perm,
        workspace.row_color,
        device_model.AU,
        tolerance,
    )
    _, workspace.num_col_color = _update_color_id!(
        workspace.col_color_id,
        workspace.col_color_start,
        workspace.col_perm,
        workspace.col_color,
        device_model.c,
        tolerance,
    )
    _, workspace.num_col_color = _update_color_id!(
        workspace.col_color_id,
        workspace.col_color_start,
        workspace.col_perm,
        workspace.col_color,
        device_model.l,
        tolerance,
    )
    _, workspace.num_col_color = _update_color_id!(
        workspace.col_color_id,
        workspace.col_color_start,
        workspace.col_perm,
        workspace.col_color,
        device_model.u,
        tolerance,
    )
    _write_color_sig!(
        row_color_sig,
        workspace.row_color_id,
        workspace.num_row_color,
        _color_seed(0, UInt64(2)),
    )
    _write_color_sig!(
        col_color_sig,
        workspace.col_color_id,
        workspace.num_col_color,
        _color_seed(0, UInt64(5)),
    )
    return nothing
end

function refine_color!(
    num_row::Int,
    num_col::Int,
    device_model::LP_info_gpu,
    workspace::FoldingWorkspace,
    tolerance::Float64,
    verbose::Bool,
)
    _init_color!(
        device_model,
        workspace,
        tolerance,
    )

    if _reduction_too_small(workspace.num_row_color, workspace.num_col_color, num_row, num_col)
        verbose && println("Not enough folding reduction, reduced size $(workspace.num_row_color) row, $(workspace.num_col_color) col.")
        return workspace.row_color_id, workspace.col_color_id, workspace.num_row_color, workspace.num_col_color, false
    end

    max_round = max(1, num_row + num_col)
    round = 0

    while round < max_round
        round += 1
        # The signature buffers double as projection workspace: each mul! writes
        # the neighbor feature that _refine_color! immediately consumes.
        mul!(workspace.col_color_sig, device_model.AT, workspace.row_color_sig)
        workspace.num_col_color, col_changed = _refine_color!(
            workspace.col_color_sig,
            workspace.col_color_id,
            workspace.col_color_start,
            workspace.col_perm,
            workspace.col_color,
            workspace.num_col_color,
            tolerance,
            _color_seed(round, UInt64(1)),
        )

        mul!(workspace.row_color_sig, device_model.A, workspace.col_color_sig)
        workspace.num_row_color, row_changed = _refine_color!(
            workspace.row_color_sig,
            workspace.row_color_id,
            workspace.row_color_start,
            workspace.row_perm,
            workspace.row_color,
            workspace.num_row_color,
            tolerance,
            _color_seed(round, UInt64(2)),
        )

        if _reduction_too_small(workspace.num_row_color, workspace.num_col_color, num_row, num_col)
            verbose && println("Not enough folding reduction, reduced size $(workspace.num_row_color) row, $(workspace.num_col_color) col.")
            return workspace.row_color_id, workspace.col_color_id, workspace.num_row_color, workspace.num_col_color, false
        end

        if !(col_changed || row_changed)
            return workspace.row_color_id, workspace.col_color_id, workspace.num_row_color, workspace.num_col_color, true
        end
    end

    verbose && println("Hybrid signature color refinement reached round limit ($(max_round)); stopping.")
    return workspace.row_color_id, workspace.col_color_id, workspace.num_row_color, workspace.num_col_color, true
end
