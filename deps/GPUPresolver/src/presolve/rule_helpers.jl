"""
Shared helpers for Layer-2 presolve rules.
"""

function append_fixed_col_postsolve_records!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    pparams::PresolveParams,
    fixed_mask::CuVector{UInt8},
    fixed_val::CuVector{Float64},
    fixed_obj::CuVector{Float64},
    keep_row_mask::CuVector{UInt8},
)
    pparams.record_postsolve_tape || return nothing

    _, fixed_cols, _ = build_maps_from_mask(fixed_mask)
    fixed_vals_sel = gather_by_red2org(fixed_val, fixed_cols)
    fixed_obj_sel = gather_by_red2org(fixed_obj, fixed_cols)
    fixed_tape_gpu = build_fixed_col_records_gpu(
        fixed_cols,
        fixed_vals_sel,
        fixed_obj_sel,
        lp.AT.rowPtr,
        lp.AT.colVal,
        lp.AT.nzVal,
        keep_row_mask;
        use_keep_row_mask=true,
        dual_mode=POSTSOLVE_DUAL_MINIMAL,
    )
    append_postsolve_tape!(plan.tape_gpu, fixed_tape_gpu)
    if pparams.record_postsolve_tape_cpu
        append_postsolve_tape!(plan.tape, PostsolveTape(fixed_tape_gpu))
    end
    return nothing
end

function append_fixed_col_inf_postsolve_records!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    pparams::PresolveParams,
    fixed_mask::CuVector{UInt8},
    infinite_fix_mask::CuVector{UInt8},
    fixed_val::CuVector{Float64},
    keep_row_mask::CuVector{UInt8},
    l::CuVector{Float64},
    u::CuVector{Float64},
    AL::CuVector{Float64},
    AU::CuVector{Float64},
)
    pparams.record_postsolve_tape || return nothing

    inf_mask = UInt8.((fixed_mask .!= UInt8(0)) .& (infinite_fix_mask .!= UInt8(0)))
    _, inf_cols, inf_count = build_maps_from_mask(inf_mask)
    Int(inf_count) == 0 && return nothing

    fixed_vals_sel = gather_by_red2org(fixed_val, inf_cols)
    fixed_tape_gpu = build_fixed_col_inf_records_gpu(
        inf_cols,
        fixed_vals_sel,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        lp.AT.rowPtr,
        lp.AT.colVal,
        keep_row_mask,
        l,
        u,
        AL,
        AU;
        dual_mode=POSTSOLVE_DUAL_EXACT,
    )
    append_postsolve_tape!(plan.tape_gpu, fixed_tape_gpu)
    if pparams.record_postsolve_tape_cpu
        append_postsolve_tape!(plan.tape, PostsolveTape(fixed_tape_gpu))
    end
    return nothing
end
