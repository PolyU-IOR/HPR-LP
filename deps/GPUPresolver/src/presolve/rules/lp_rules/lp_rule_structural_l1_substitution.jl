"""
Layer-2 rule: structural L1 substitution.

Current scope:
- LP only
- strict structural certification only
- GPU screening plus selected GPU direct rewrites, with host fallback for unsupported patterns
- real elimination is enabled only when postsolve tape recording is off

The rewrite reuses a subset of original row/column slots so it fits the current
presolve rebuild path without introducing brand-new indices into the tape-aware
record mappings.
"""

const STRUCTURAL_L1_SUB_TOL = 1.0e-12

@inline _sls_finite_abs(v::Float64) = isfinite(v) ? abs(v) : 0.0

@inline function _sls_same(a::Float64, b::Float64; tol::Float64=STRUCTURAL_L1_SUB_TOL)
    return isfinite(a) && isfinite(b) && abs(a - b) <= tol
end

function _sls_row_entries(AT::SparseMatrixCSC{Float64,Int32}, row::Int)
    lo = AT.colptr[row]
    hi = AT.colptr[row + 1] - 1
    if hi < lo
        return Int[], Float64[]
    end
    return Int.(AT.rowval[lo:hi]), collect(AT.nzval[lo:hi])
end

function _sls_row_is_equality(AL::Vector{Float64}, AU::Vector{Float64}, row::Int;
    rhs::Union{Nothing,Float64}=nothing,
    tol::Float64=STRUCTURAL_L1_SUB_TOL)
    isfinite(AL[row]) && isfinite(AU[row]) || return false
    _sls_same(AL[row], AU[row]; tol=tol) || return false
    rhs === nothing && return true
    return _sls_same(AL[row], rhs; tol=tol)
end

function _sls_row_is_lower_only(AL::Vector{Float64}, AU::Vector{Float64}, row::Int;
    rhs::Union{Nothing,Float64}=nothing,
    tol::Float64=STRUCTURAL_L1_SUB_TOL)
    isfinite(AL[row]) && isinf(AU[row]) || return false
    rhs === nothing && return true
    return _sls_same(AL[row], rhs; tol=tol)
end

function _sls_row_is_upper_only(AL::Vector{Float64}, AU::Vector{Float64}, row::Int;
    rhs::Union{Nothing,Float64}=nothing,
    tol::Float64=STRUCTURAL_L1_SUB_TOL)
    isinf(AL[row]) && isfinite(AU[row]) || return false
    rhs === nothing && return true
    return _sls_same(AU[row], rhs; tol=tol)
end

function _sls_two_col_coeff(cols::Vector{Int}, vals::Vector{Float64}, c1::Int, c2::Int)
    length(cols) == 2 || return nothing
    d = Dict(cols[k] => vals[k] for k in 1:2)
    haskey(d, c1) && haskey(d, c2) || return nothing
    return d[c1], d[c2]
end

function _sls_normalized_zero_lower_row(
    AL::Vector{Float64},
    AU::Vector{Float64},
    AT::SparseMatrixCSC{Float64,Int32},
    row::Int,
)
    cols, vals = _sls_row_entries(AT, row)
    if _sls_row_is_lower_only(AL, AU, row; rhs=0.0)
        return cols, vals
    elseif _sls_row_is_upper_only(AL, AU, row; rhs=0.0)
        return cols, collect(-vals)
    end
    error("Row $(row) is neither lower-only nor upper-only with zero rhs.")
end

function _sls_add_entry!(
    row_entries::Dict{Int,Float64},
    col::Int,
    val::Float64,
)
    abs(val) <= STRUCTURAL_L1_SUB_TOL && return nothing
    row_entries[col] = get(row_entries, col, 0.0) + val
    if abs(row_entries[col]) <= STRUCTURAL_L1_SUB_TOL
        delete!(row_entries, col)
    end
    return nothing
end

function _sls_finalize_sparse(
    row_maps::Vector{Dict{Int,Float64}},
    m::Int,
    n::Int,
)
    I = Int32[]
    J = Int32[]
    V = Float64[]
    for row in 1:m
        entries = row_maps[row]
        isempty(entries) && continue
        for col in sort!(collect(keys(entries)))
            val = entries[col]
            abs(val) <= STRUCTURAL_L1_SUB_TOL && continue
            push!(I, Int32(row))
            push!(J, Int32(col))
            push!(V, val)
        end
    end
    return sparse(I, J, V, m, n)
end

function _sls_detect_l1_split_3row(
    A::SparseMatrixCSC{Float64,Int32},
    AT::SparseMatrixCSC{Float64,Int32},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
)
    m, _ = size(A)
    m % 3 == 0 || error("Expected a pure 3-row block structure.")
    nblocks = div(m, 3)
    colnnz = diff(A.colptr)
    shared_cols = Int.(findall(==(Int32(nblocks)), colnnz))
    isempty(shared_cols) && error("No shared columns detected.")
    shared_pos = Dict{Int,Int}(c => k for (k, c) in enumerate(shared_cols))

    t_cols = Vector{Int}(undef, nblocks)
    e_cols = Vector{Int}(undef, nblocks)

    for i in 1:nblocks
        eq = 3i - 2
        r1 = 3i - 1
        r2 = 3i
        _sls_row_is_equality(AL, AU, eq) || error("Dense row $(eq) is not an equality.")

        cols_eq, vals_eq = _sls_row_entries(AT, eq)
        local_eq = [(c, v) for (c, v) in zip(cols_eq, vals_eq) if !haskey(shared_pos, c)]
        length(local_eq) == 1 || error("Dense row $(eq) does not have exactly one local residual column.")
        e_cols[i] = local_eq[1][1]
        abs(local_eq[1][2]) > STRUCTURAL_L1_SUB_TOL || error("Residual coefficient is zero.")
        isinf(l[e_cols[i]]) && isinf(u[e_cols[i]]) || error("Residual column is not free.")

        cols_a, vals_a = _sls_normalized_zero_lower_row(AL, AU, AT, r1)
        cols_b, vals_b = _sls_normalized_zero_lower_row(AL, AU, AT, r2)
        length(cols_a) == 2 && length(cols_b) == 2 || error("Epigraph rows are not two-column rows.")
        e_cols[i] in cols_a && e_cols[i] in cols_b || error("Epigraph rows do not contain the residual column.")
        cand_a = only([c for c in cols_a if c != e_cols[i]])
        cand_b = only([c for c in cols_b if c != e_cols[i]])
        cand_a == cand_b || error("Epigraph rows use different t columns.")
        t_cols[i] = cand_a
        abs(l[t_cols[i]]) <= STRUCTURAL_L1_SUB_TOL && isinf(u[t_cols[i]]) ||
            error("Epigraph t column has unsupported bounds.")

        e_a, t_a = _sls_two_col_coeff(cols_a, vals_a, e_cols[i], t_cols[i])
        e_b, t_b = _sls_two_col_coeff(cols_b, vals_b, e_cols[i], t_cols[i])
        t_a > 0.0 && t_b > 0.0 || error("Epigraph t coefficients must be positive.")
        signs = sort([e_a / t_a, e_b / t_b])
        abs(signs[1] + 1.0) <= STRUCTURAL_L1_SUB_TOL &&
            abs(signs[2] - 1.0) <= STRUCTURAL_L1_SUB_TOL ||
            error("Rows do not encode t >= |e|.")
    end

    return (t_cols=t_cols, e_cols=e_cols)
end

function _sls_apply_l1_split_3row(
    A::SparseMatrixCSC{Float64,Int32},
    AT::SparseMatrixCSC{Float64,Int32},
    c::Vector{Float64},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
)
    det = _sls_detect_l1_split_3row(A, AT, AL, AU, l, u)
    m, n = size(A)
    keep_row = trues(m)
    keep_col = trues(n)
    new_c = copy(c)
    new_l = copy(l)
    new_u = copy(u)
    splits = StructuralL1SplitRecovery[]

    for i in eachindex(det.t_cols)
        eq = 3i - 2
        r1 = 3i - 1
        r2 = 3i
        t_col = det.t_cols[i]
        e_col = det.e_cols[i]

        keep_row[r1] = false
        keep_row[r2] = false

        new_c[t_col] = c[t_col] + c[e_col]
        new_c[e_col] = c[t_col] - c[e_col]
        new_l[t_col] = 0.0
        new_l[e_col] = 0.0
        new_u[t_col] = Inf
        new_u[e_col] = Inf
        push!(splits, StructuralL1SplitRecovery(Int32(t_col), Int32(e_col), 1.0))
    end

    return (
        pattern=:l1_split_3row,
        A_new=nothing,
        keep_row=keep_row,
        keep_col=keep_col,
        c_new=new_c,
        AL_new=copy(AL),
        AU_new=copy(AU),
        l_new=new_l,
        u_new=new_u,
        removed_rows=count(!, keep_row),
        removed_cols=0,
        gpu_rewrite=_sls_l1_split_gpu_rewrite_meta(det),
        primal_recovery=StructuralL1PrimalRecoveryStep(
            :l1_split_3row,
            splits,
            StructuralOuterPairRecovery[],
            StructuralLinkedSlackRecovery[],
            StructuralMaxSlackRecovery[],
        ),
    )
end

function _sls_detect_outer_equal_pairs(
    AT::SparseMatrixCSC{Float64,Int32},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    m::Int,
)
    pairs = Tuple{Int,Int}[]
    row = 1
    while row + 2 <= m
        cols1, vals1 = _sls_row_entries(AT, row)
        cols2, vals2 = _sls_row_entries(AT, row + 1)
        cols3, vals3 = _sls_row_entries(AT, row + 2)
        length(cols1) == 2 || break
        length(cols2) == 2 || break
        length(cols3) == 2 || break
        sort(cols1) == sort(cols2) == sort(cols3) || break
        c1, c2 = sort(cols1)
        v1 = _sls_two_col_coeff(cols1, vals1, c1, c2)
        v2 = _sls_two_col_coeff(cols2, vals2, c1, c2)
        v3 = _sls_two_col_coeff(cols3, vals3, c1, c2)
        v1 == (1.0, -1.0) || break
        v2 == (1.0, 1.0) || break
        v3 == (-1.0, 1.0) || break
        _sls_row_is_lower_only(AL, AU, row; rhs=0.0) || break
        _sls_row_is_lower_only(AL, AU, row + 1; rhs=0.0) || break
        _sls_row_is_lower_only(AL, AU, row + 2; rhs=0.0) || break
        l[c1] == 0.0 && isfinite(u[c1]) || break
        isinf(l[c2]) && isinf(u[c2]) || break
        push!(pairs, (c1, c2))
        row += 3
    end
    return pairs, row
end

function _sls_detect_linked_l1_inner_blocks(
    AT::SparseMatrixCSC{Float64,Int32},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    c::Vector{Float64},
    start_row::Int,
    m::Int,
)
    blocks = NamedTuple[]
    row = start_row
    while row <= m
        row + 7 <= m || error("Trailing rows do not form an 8-row block.")
        _sls_row_is_equality(AL, AU, row) || error("Dense row is not an equality.")
        cols_dense, _ = _sls_row_entries(AT, row)
        length(cols_dense) >= 2 || error("Dense row is malformed.")

        cols_ep1, vals_ep1 = _sls_row_entries(AT, row + 1)
        cols_ep2, vals_ep2 = _sls_row_entries(AT, row + 2)
        _sls_row_is_lower_only(AL, AU, row + 1; rhs=0.0) || error("First epigraph row is not lower-only zero.")
        _sls_row_is_lower_only(AL, AU, row + 2; rhs=0.0) || error("Second epigraph row is not lower-only zero.")
        length(cols_ep1) == 2 && length(cols_ep2) == 2 || error("Epigraph rows are not two-column rows.")
        sort(cols_ep1) == sort(cols_ep2) || error("Epigraph rows use different columns.")
        e_col, q_col = sort(cols_ep1)
        v_ep1 = _sls_two_col_coeff(cols_ep1, vals_ep1, e_col, q_col)
        v_ep2 = _sls_two_col_coeff(cols_ep2, vals_ep2, e_col, q_col)
        v_ep1 == (-1.0, 1.0) || error("First epigraph row does not encode q - e >= 0.")
        v_ep2 == (1.0, 1.0) || error("Second epigraph row does not encode q + e >= 0.")
        e_col in cols_dense || error("Dense row does not contain e.")
        isinf(l[e_col]) && isinf(u[e_col]) || error("e column is not free.")
        l[q_col] == 0.0 && isinf(u[q_col]) || error("q column is not nonnegative free-up.")

        cols_link, vals_link = _sls_row_entries(AT, row + 3)
        _sls_row_is_equality(AL, AU, row + 3; rhs=0.0) || error("Link row is not a zero equality.")
        length(cols_link) == 2 || error("Link row is not two-column.")
        q_pos = findfirst(==(q_col), cols_link)
        q_pos !== nothing || error("Link row does not contain q.")
        s_pos = q_pos == 1 ? 2 : 1
        s_col = cols_link[s_pos]
        alpha = vals_link[q_pos]
        vals_link[s_pos] == -1.0 || error("Link row does not contain -s.")
        alpha > 0.0 || error("Link coefficient alpha must be positive.")
        l[s_col] == 0.0 && isinf(u[s_col]) || error("s column is not nonnegative free-up.")

        local_rows = ntuple(k -> row + 3 + k, 4)
        for rr in local_rows
            _sls_row_is_lower_only(AL, AU, rr) || error("Local row is not lower-only.")
            cols_loc, vals_loc = _sls_row_entries(AT, rr)
            length(cols_loc) == 2 || error("Local row is not two-column.")
            s_here = findfirst(==(s_col), cols_loc)
            s_here !== nothing || error("Local row does not contain s.")
            vals_loc[s_here] == -1.0 || error("Local row does not use -s.")
            x_pos = s_here == 1 ? 2 : 1
            x_col = cols_loc[x_pos]
            vals_loc[x_pos] == 1.0 || error("Local row does not use +x.")
            l[x_col] == 0.0 && isinf(u[x_col]) || error("Local x column is not nonnegative.")
            c[x_col] > STRUCTURAL_L1_SUB_TOL || error("Local x column does not have positive objective cost.")
        end

        push!(blocks, (
            dense_row=row,
            e_col=e_col,
            q_col=q_col,
            s_col=s_col,
            alpha=alpha,
            local_rows=local_rows,
        ))
        row += 8
    end
    return blocks
end

function _sls_apply_outer_pair_linked_l1(
    A::SparseMatrixCSC{Float64,Int32},
    AT::SparseMatrixCSC{Float64,Int32},
    c::Vector{Float64},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
)
    m, n = size(A)
    outer_pairs, start_row = _sls_detect_outer_equal_pairs(AT, AL, AU, l, u, m)
    isempty(outer_pairs) && error("No certified outer equality pairs detected.")
    blocks = _sls_detect_linked_l1_inner_blocks(AT, AL, AU, l, u, c, start_row, m)
    isempty(blocks) && error("No certified inner blocks detected.")
    gpu_rewrite = _sls_outer_pair_linked_l1_gpu_safe(AT, outer_pairs, blocks) ?
        _sls_outer_pair_linked_l1_gpu_rewrite_meta(outer_pairs, blocks, start_row) : nothing

    keep_row = trues(m)
    keep_col = trues(n)
    row_maps = gpu_rewrite === nothing ? [Dict{Int,Float64}() for _ in 1:m] : Dict{Int,Float64}[]
    outer_free_to_bound = Dict(free => bound for (bound, free) in outer_pairs)

    new_c = copy(c)
    new_l = copy(l)
    new_u = copy(u)
    splits = StructuralL1SplitRecovery[]
    pair_recoveries = StructuralOuterPairRecovery[
        StructuralOuterPairRecovery(Int32(bound), Int32(free)) for (bound, free) in outer_pairs
    ]
    linked_slacks = StructuralLinkedSlackRecovery[]

    for (bound, free) in outer_pairs
        keep_col[free] = false
        new_c[bound] += c[free]
    end
    for row in 1:(start_row - 1)
        keep_row[row] = false
    end

    for block in blocks
        q_col = block.q_col
        e_col = block.e_col
        s_col = block.s_col

        keep_row[block.dense_row + 1] = false
        keep_row[block.dense_row + 2] = false
        keep_row[block.dense_row + 3] = false
        keep_col[s_col] = false

        if gpu_rewrite === nothing
            cols_dense, vals_dense = _sls_row_entries(AT, block.dense_row)
            for (col, val) in zip(cols_dense, vals_dense)
                if col == e_col
                    _sls_add_entry!(row_maps[block.dense_row], q_col, val)
                    _sls_add_entry!(row_maps[block.dense_row], e_col, -val)
                else
                    mapped = get(outer_free_to_bound, col, col)
                    mapped == s_col && error("Dense row unexpectedly uses s.")
                    mapped == e_col && col != e_col && error("Dense row uses an eliminated local e column.")
                    _sls_add_entry!(row_maps[block.dense_row], mapped, val)
                end
            end

            for rr in block.local_rows
                cols_loc, vals_loc = _sls_row_entries(AT, rr)
                for (col, val) in zip(cols_loc, vals_loc)
                    if col == s_col
                        _sls_add_entry!(row_maps[rr], q_col, val * block.alpha)
                        _sls_add_entry!(row_maps[rr], e_col, val * block.alpha)
                    else
                        mapped = get(outer_free_to_bound, col, col)
                        mapped == s_col && error("Local row unexpectedly keeps s.")
                        _sls_add_entry!(row_maps[rr], mapped, val)
                    end
                end
            end
        end

        new_c[q_col] = c[e_col] + c[q_col] + block.alpha * c[s_col]
        new_c[e_col] = -c[e_col] + c[q_col] + block.alpha * c[s_col]
        new_l[q_col] = 0.0
        new_l[e_col] = 0.0
        new_u[q_col] = Inf
        new_u[e_col] = Inf
        push!(splits, StructuralL1SplitRecovery(Int32(q_col), Int32(e_col), 1.0))
        push!(linked_slacks, StructuralLinkedSlackRecovery(Int32(s_col), Int32(q_col), block.alpha))
    end

    if gpu_rewrite === nothing
        for row in 1:m
            keep_row[row] || continue
            isempty(row_maps[row]) || continue
            cols, vals = _sls_row_entries(AT, row)
            for (col, val) in zip(cols, vals)
                keep_col[col] || continue
                mapped = get(outer_free_to_bound, col, col)
                keep_col[mapped] || continue
                _sls_add_entry!(row_maps[row], mapped, val)
            end
        end
    end

    return (
        pattern=:outer_pair_linked_l1,
        A_new=gpu_rewrite === nothing ? _sls_finalize_sparse(row_maps, m, n) : nothing,
        keep_row=keep_row,
        keep_col=keep_col,
        c_new=new_c,
        AL_new=copy(AL),
        AU_new=copy(AU),
        l_new=new_l,
        u_new=new_u,
        removed_rows=count(!, keep_row),
        removed_cols=count(!, keep_col),
        gpu_rewrite=gpu_rewrite,
        primal_recovery=StructuralL1PrimalRecoveryStep(
            :outer_pair_linked_l1,
            splits,
            pair_recoveries,
            linked_slacks,
            StructuralMaxSlackRecovery[],
        ),
    )
end

function _sls_row_to_col_value(cols::Vector{Int}, vals::Vector{Float64})
    return Dict(cols[k] => vals[k] for k in eachindex(cols))
end

function _sls_try_l1_orientation(
    l::Vector{Float64},
    u::Vector{Float64},
    c::Vector{Float64},
    cols::Vector{Int},
    vals_a::Vector{Float64},
    vals_b::Vector{Float64},
    t_col::Int,
    e_col::Int,
)
    isinf(l[e_col]) && isinf(u[e_col]) || return nothing
    abs(l[t_col]) <= STRUCTURAL_L1_SUB_TOL && isinf(u[t_col]) || return nothing
    c[t_col] >= -STRUCTURAL_L1_SUB_TOL || return nothing

    da = _sls_row_to_col_value(cols, vals_a)
    db = _sls_row_to_col_value(cols, vals_b)
    t_a = da[t_col]
    t_b = db[t_col]
    e_a = da[e_col]
    e_b = db[e_col]
    t_a > STRUCTURAL_L1_SUB_TOL && t_b > STRUCTURAL_L1_SUB_TOL || return nothing
    rho_a = e_a / t_a
    rho_b = e_b / t_b
    abs(rho_a + rho_b) <= STRUCTURAL_L1_SUB_TOL * max(1.0, abs(rho_a), abs(rho_b)) || return nothing
    abs(rho_a) > STRUCTURAL_L1_SUB_TOL || return nothing
    return abs(rho_a)
end

function _sls_find_graph_l1_candidates(
    A::SparseMatrixCSC{Float64,Int32},
    AT::SparseMatrixCSC{Float64,Int32},
    c::Vector{Float64},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
)
    row_groups = Dict{Tuple{Int,Int},Vector{Tuple{Int,Vector{Int},Vector{Float64}}}}()
    for row in 1:size(A, 1)
        try
            cols, vals = _sls_normalized_zero_lower_row(AL, AU, AT, row)
            length(cols) == 2 || continue
            key = Tuple(sort(cols))
            push!(get!(row_groups, key, Tuple{Int,Vector{Int},Vector{Float64}}[]), (row, cols, vals))
        catch
            continue
        end
    end

    blocks = NamedTuple[]
    epi_rows_by_block = Tuple{Int,Int}[]
    used_t = Set{Int}()
    used_e = Set{Int}()

    for (key, rows) in row_groups
        length(rows) >= 2 || continue
        c1, c2 = key
        for a in 1:(length(rows) - 1), b in (a + 1):length(rows)
            r_a, cols_a, vals_a = rows[a]
            r_b, cols_b, vals_b = rows[b]
            cols = collect(key)
            vals_a_by_col = [_sls_row_to_col_value(cols_a, vals_a)[col] for col in cols]
            vals_b_by_col = [_sls_row_to_col_value(cols_b, vals_b)[col] for col in cols]

            orientation = nothing
            for (t_col, e_col) in ((c1, c2), (c2, c1))
                rho = _sls_try_l1_orientation(l, u, c, cols, vals_a_by_col, vals_b_by_col, t_col, e_col)
                if rho !== nothing
                    orientation = (t_col=t_col, e_col=e_col, rho=rho)
                    break
                end
            end
            orientation === nothing && continue
            t_col = orientation.t_col
            e_col = orientation.e_col
            rho = orientation.rho
            (t_col in used_t || e_col in used_e) && continue

            e_rows = Set(Int.(A.rowval[A.colptr[e_col]:(A.colptr[e_col + 1] - 1)]))
            epi_set = Set([r_a, r_b])
            coupling_rows = setdiff(e_rows, epi_set)
            length(coupling_rows) == 1 || continue
            coupling_row = first(coupling_rows)
            _sls_row_is_equality(AL, AU, coupling_row) || continue

            push!(blocks, (coupling_row=coupling_row, t_col=t_col, e_col=e_col, rho=rho))
            push!(epi_rows_by_block, (r_a, r_b))
            push!(used_t, t_col)
            push!(used_e, e_col)
            break
        end
    end

    return blocks, epi_rows_by_block
end

function _sls_graph_l1_slack_recoveries(
    A::SparseMatrixCSC{Float64,Int32},
    AT::SparseMatrixCSC{Float64,Int32},
    c::Vector{Float64},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    blocks::Vector,
    epi_rows_by_block::Vector{Tuple{Int,Int}},
)
    epi_rows = Set{Int}()
    for pair in epi_rows_by_block
        push!(epi_rows, pair[1])
        push!(epi_rows, pair[2])
    end

    row_to_block_factor = Dict{Int,Tuple{Int,Int,Float64}}()
    block_has_bad_extra_row = falses(length(blocks))

    for (i, block) in enumerate(blocks)
        t_rows = Set(Int.(A.rowval[A.colptr[block.t_col]:(A.colptr[block.t_col + 1] - 1)]))
        extra_rows = setdiff(t_rows, epi_rows)
        for row in extra_rows
            try
                cols, vals = _sls_normalized_zero_lower_row(AL, AU, AT, row)
                length(cols) == 2 || (block_has_bad_extra_row[i] = true; continue)
                d = _sls_row_to_col_value(cols, vals)
                coeff_t = d[block.t_col]
                coeff_t < -STRUCTURAL_L1_SUB_TOL || (block_has_bad_extra_row[i] = true; continue)
                slack_cols = [col for col in cols if col != block.t_col]
                length(slack_cols) == 1 || (block_has_bad_extra_row[i] = true; continue)
                slack_col = slack_cols[1]
                coeff_s = d[slack_col]
                coeff_s > STRUCTURAL_L1_SUB_TOL || (block_has_bad_extra_row[i] = true; continue)
                row_to_block_factor[row] = (i, slack_col, -coeff_t / coeff_s)
            catch
                block_has_bad_extra_row[i] = true
            end
        end
    end

    slack_rows = Dict{Int,Vector{Int}}()
    for (row, (_, slack_col, _)) in row_to_block_factor
        push!(get!(slack_rows, slack_col, Int[]), row)
    end

    removable_slacks = Set{Int}()
    removable_rows = Set{Int}()
    for (slack_col, rows) in slack_rows
        abs(c[slack_col]) <= STRUCTURAL_L1_SUB_TOL || continue
        isinf(u[slack_col]) || continue
        all_slack_rows = Set(Int.(A.rowval[A.colptr[slack_col]:(A.colptr[slack_col + 1] - 1)]))
        all_slack_rows == Set(rows) || continue
        ok = true
        for row in rows
            block_idx, _, _ = row_to_block_factor[row]
            if block_has_bad_extra_row[block_idx]
                ok = false
                break
            end
        end
        ok || continue
        push!(removable_slacks, slack_col)
        union!(removable_rows, rows)
    end

    valid_blocks = trues(length(blocks))
    for (i, block) in enumerate(blocks)
        t_rows = Set(Int.(A.rowval[A.colptr[block.t_col]:(A.colptr[block.t_col + 1] - 1)]))
        extra_rows = setdiff(t_rows, epi_rows)
        isempty(extra_rows) && continue
        all(row -> row in removable_rows, extra_rows) || (valid_blocks[i] = false)
    end

    slack_recoveries = StructuralMaxSlackRecovery[]
    for slack_col in sort!(collect(removable_slacks))
        rows = get(slack_rows, slack_col, Int[])
        t_cols = Int32[]
        factors = Float64[]
        for row in rows
            block_idx, _, factor = row_to_block_factor[row]
            push!(t_cols, Int32(blocks[block_idx].t_col))
            push!(factors, factor)
        end
        push!(slack_recoveries, StructuralMaxSlackRecovery(Int32(slack_col), t_cols, factors))
    end

    return valid_blocks, removable_rows, removable_slacks, slack_recoveries
end

function _sls_apply_graph_l1_substitution(
    A::SparseMatrixCSC{Float64,Int32},
    AT::SparseMatrixCSC{Float64,Int32},
    c::Vector{Float64},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
)
    blocks_all, epi_rows_all = _sls_find_graph_l1_candidates(A, AT, c, AL, AU, l, u)
    isempty(blocks_all) && error("No graph-certified L1 epigraph gadgets detected.")
    valid_blocks, removable_rows, removable_slacks, slack_recoveries =
        _sls_graph_l1_slack_recoveries(A, AT, c, AL, AU, l, u, blocks_all, epi_rows_all)

    blocks = NamedTuple[]
    epi_rows_by_block = Tuple{Int,Int}[]
    for (i, keep) in enumerate(valid_blocks)
        keep || continue
        push!(blocks, blocks_all[i])
        push!(epi_rows_by_block, epi_rows_all[i])
    end
    isempty(blocks) && error("No graph-certified L1 gadgets survived safety checks.")

    m, n = size(A)
    keep_row = trues(m)
    keep_col = trues(n)
    row_maps = [Dict{Int,Float64}() for _ in 1:m]
    block_by_e = Dict(block.e_col => block for block in blocks)
    block_by_t = Dict(block.t_col => block for block in blocks)

    rows_to_remove = Set{Int}()
    for pair in epi_rows_by_block
        push!(rows_to_remove, pair[1])
        push!(rows_to_remove, pair[2])
    end
    union!(rows_to_remove, removable_rows)
    for row in rows_to_remove
        keep_row[row] = false
    end
    for slack_col in removable_slacks
        keep_col[slack_col] = false
    end
    gpu_rewrite = _sls_graph_gpu_rewrite_safe(AT, blocks, keep_row) ?
        _sls_graph_gpu_rewrite_meta(blocks) : nothing

    new_c = copy(c)
    new_l = copy(l)
    new_u = copy(u)
    splits = StructuralL1SplitRecovery[]
    for block in blocks
        inv_rho = 1.0 / block.rho
        new_c[block.t_col] = c[block.t_col] + c[block.e_col] * inv_rho
        new_c[block.e_col] = c[block.t_col] - c[block.e_col] * inv_rho
        new_l[block.t_col] = 0.0
        new_l[block.e_col] = 0.0
        new_u[block.t_col] = Inf
        new_u[block.e_col] = Inf
        push!(splits, StructuralL1SplitRecovery(Int32(block.t_col), Int32(block.e_col), block.rho))
    end

    if gpu_rewrite === nothing
        for row in 1:m
            keep_row[row] || continue
            cols, vals = _sls_row_entries(AT, row)
            for (col, val) in zip(cols, vals)
                if haskey(block_by_e, col)
                    block = block_by_e[col]
                    coeff = val / block.rho
                    _sls_add_entry!(row_maps[row], block.t_col, coeff)
                    _sls_add_entry!(row_maps[row], block.e_col, -coeff)
                elseif haskey(block_by_t, col)
                    block = block_by_t[col]
                    _sls_add_entry!(row_maps[row], block.t_col, val)
                    _sls_add_entry!(row_maps[row], block.e_col, val)
                else
                    keep_col[col] || continue
                    _sls_add_entry!(row_maps[row], col, val)
                end
            end
        end
    end

    return (
        pattern=:graph_l1_substitution,
        A_new=gpu_rewrite === nothing ? _sls_finalize_sparse(row_maps, m, n) : nothing,
        keep_row=keep_row,
        keep_col=keep_col,
        c_new=new_c,
        AL_new=copy(AL),
        AU_new=copy(AU),
        l_new=new_l,
        u_new=new_u,
        removed_rows=count(!, keep_row),
        removed_cols=count(!, keep_col),
        gpu_rewrite=gpu_rewrite,
        primal_recovery=StructuralL1PrimalRecoveryStep(
            :graph_l1_substitution,
            splits,
            StructuralOuterPairRecovery[],
            StructuralLinkedSlackRecovery[],
            slack_recoveries,
        ),
    )
end

function _try_structural_l1_substitution(
    A::SparseMatrixCSC{Float64,Int32},
    AT::SparseMatrixCSC{Float64,Int32},
    c::Vector{Float64},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    pattern::Symbol,
)
    if pattern == :graph_l1_substitution
        return _sls_apply_graph_l1_substitution(A, AT, c, AL, AU, l, u)
    elseif pattern == :outer_pair_linked_l1
        return _sls_apply_outer_pair_linked_l1(A, AT, c, AL, AU, l, u)
    elseif pattern == :l1_split_3row
        return _sls_apply_l1_split_3row(A, AT, c, AL, AU, l, u)
    end
    error("Unsupported structural_l1 pattern $(pattern).")
end

function _sls_target_cols_unique(target_cols::Vector{Int})
    seen = Set{Int}()
    for col in target_cols
        col > 0 || return false
        col in seen && return false
        push!(seen, col)
    end
    return true
end

function _sls_l1_split_gpu_rewrite_meta(det)
    return (
        kind=:l1_split_3row,
        t_cols=Int32.(det.t_cols),
        e_cols=Int32.(det.e_cols),
    )
end

function _sls_outer_pair_linked_l1_gpu_safe(
    AT::SparseMatrixCSC{Float64,Int32},
    outer_pairs,
    blocks,
)
    outer_free_to_bound = Dict(free => bound for (bound, free) in outer_pairs)
    for block in blocks
        cols_dense, _ = _sls_row_entries(AT, block.dense_row)
        targets = Int[]
        for col in cols_dense
            if col == block.e_col
                push!(targets, block.q_col)
                push!(targets, block.e_col)
            else
                push!(targets, get(outer_free_to_bound, col, col))
            end
        end
        _sls_target_cols_unique(targets) || return false

        for rr in block.local_rows
            cols_loc, _ = _sls_row_entries(AT, rr)
            targets = Int[]
            for col in cols_loc
                if col == block.s_col
                    push!(targets, block.q_col)
                    push!(targets, block.e_col)
                else
                    push!(targets, get(outer_free_to_bound, col, col))
                end
            end
            _sls_target_cols_unique(targets) || return false
        end
    end
    return true
end

function _sls_outer_pair_linked_l1_gpu_rewrite_meta(outer_pairs, blocks, start_row::Int)
    return (
        kind=:outer_pair_linked_l1,
        start_row=Int32(start_row),
        bound_cols=Int32[bound for (bound, _) in outer_pairs],
        free_cols=Int32[free for (_, free) in outer_pairs],
        q_cols=Int32[block.q_col for block in blocks],
        e_cols=Int32[block.e_col for block in blocks],
        s_cols=Int32[block.s_col for block in blocks],
        alphas=Float64[block.alpha for block in blocks],
    )
end

function _sls_graph_gpu_rewrite_safe(
    AT::SparseMatrixCSC{Float64,Int32},
    blocks,
    keep_row::AbstractVector{Bool},
)
    for block in blocks
        cols_row, _ = _sls_row_entries(AT, block.coupling_row)
        targets = Int[]
        for col in cols_row
            if col == block.e_col
                push!(targets, block.t_col)
                push!(targets, block.e_col)
            elseif col == block.t_col
                push!(targets, block.t_col)
                push!(targets, block.e_col)
            else
                push!(targets, col)
            end
        end
        _sls_target_cols_unique(targets) || return false
    end
    for row in eachindex(keep_row)
        keep_row[row] || continue
    end
    return true
end

function _sls_graph_gpu_rewrite_meta(blocks)
    return (
        kind=:graph_l1_substitution,
        coupling_rows=Int32[block.coupling_row for block in blocks],
        t_cols=Int32[block.t_col for block in blocks],
        e_cols=Int32[block.e_col for block in blocks],
        rhos=Float64[block.rho for block in blocks],
    )
end

function _kernel_structural_row_metadata!(
    eq_row,
    eq_zero_two_nnz,
    lower_two_nnz,
    zero_lower_two_nnz,
    raw_col1,
    raw_col2,
    raw_val1,
    raw_val2,
    zl_col1,
    zl_col2,
    zl_val1,
    zl_val2,
    rowPtr,
    colVal,
    nzVal,
    AL,
    AU,
    m,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= m
        @inbounds al = AL[row]
        @inbounds au = AU[row]
        @inbounds lo = rowPtr[row]
        @inbounds hi = rowPtr[row + 1] - Int32(1)
        nnz = hi >= lo ? hi - lo + Int32(1) : Int32(0)

        is_eq = isfinite(al) && isfinite(au) && abs(al - au) <= STRUCTURAL_L1_SUB_TOL
        @inbounds eq_row[row] = is_eq ? UInt8(1) : UInt8(0)
        @inbounds eq_zero_two_nnz[row] = UInt8(0)
        @inbounds lower_two_nnz[row] = UInt8(0)
        @inbounds zero_lower_two_nnz[row] = UInt8(0)
        @inbounds raw_col1[row] = Int32(-1)
        @inbounds raw_col2[row] = Int32(-1)
        @inbounds raw_val1[row] = 0.0
        @inbounds raw_val2[row] = 0.0
        @inbounds zl_col1[row] = Int32(-1)
        @inbounds zl_col2[row] = Int32(-1)
        @inbounds zl_val1[row] = 0.0
        @inbounds zl_val2[row] = 0.0

        if nnz == Int32(2)
            @inbounds c1 = colVal[lo]
            @inbounds c2 = colVal[lo + Int32(1)]
            @inbounds v1 = nzVal[lo]
            @inbounds v2 = nzVal[lo + Int32(1)]
            if c2 < c1
                c1, c2 = c2, c1
                v1, v2 = v2, v1
            end

            @inbounds raw_col1[row] = c1
            @inbounds raw_col2[row] = c2
            @inbounds raw_val1[row] = v1
            @inbounds raw_val2[row] = v2

            if is_eq && abs(al) <= STRUCTURAL_L1_SUB_TOL
                @inbounds eq_zero_two_nnz[row] = UInt8(1)
            end

            if isfinite(al) && isinf(au)
                @inbounds lower_two_nnz[row] = UInt8(1)
            end

            is_zero_lower = isfinite(al) && isinf(au) && abs(al) <= STRUCTURAL_L1_SUB_TOL
            is_zero_upper = isinf(al) && isfinite(au) && abs(au) <= STRUCTURAL_L1_SUB_TOL
            if is_zero_lower || is_zero_upper
                scale = is_zero_lower ? 1.0 : -1.0
                @inbounds zero_lower_two_nnz[row] = UInt8(1)
                @inbounds zl_col1[row] = c1
                @inbounds zl_col2[row] = c2
                @inbounds zl_val1[row] = scale * v1
                @inbounds zl_val2[row] = scale * v2
            end
        end
    end
    return
end

function _sls_gpu_row_metadata_gpu(
    A_csr::CuSparseMatrixCSR{Float64,Int32},
    AL::CuVector{Float64},
    AU::CuVector{Float64},
)
    m, _ = size(A_csr)
    eq_row = CUDA.zeros(UInt8, m)
    eq_zero_two_nnz = CUDA.zeros(UInt8, m)
    lower_two_nnz = CUDA.zeros(UInt8, m)
    zero_lower_two_nnz = CUDA.zeros(UInt8, m)
    raw_col1 = CUDA.fill(Int32(-1), m)
    raw_col2 = CUDA.fill(Int32(-1), m)
    raw_val1 = CUDA.zeros(Float64, m)
    raw_val2 = CUDA.zeros(Float64, m)
    zl_col1 = CUDA.fill(Int32(-1), m)
    zl_col2 = CUDA.fill(Int32(-1), m)
    zl_val1 = CUDA.zeros(Float64, m)
    zl_val2 = CUDA.zeros(Float64, m)

    if m > 0
        blocks = cld(m, GPU_PRESOLVE_THREADS)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_structural_row_metadata!(
            eq_row,
            eq_zero_two_nnz,
            lower_two_nnz,
            zero_lower_two_nnz,
            raw_col1,
            raw_col2,
            raw_val1,
            raw_val2,
            zl_col1,
            zl_col2,
            zl_val1,
            zl_val2,
            A_csr.rowPtr,
            A_csr.colVal,
            A_csr.nzVal,
            AL,
            AU,
            Int32(m),
        )
    end

    return (
        eq_row=eq_row,
        eq_zero_two_nnz=eq_zero_two_nnz,
        lower_two_nnz=lower_two_nnz,
        zero_lower_two_nnz=zero_lower_two_nnz,
        raw_col1=raw_col1,
        raw_col2=raw_col2,
        raw_val1=raw_val1,
        raw_val2=raw_val2,
        zl_col1=zl_col1,
        zl_col2=zl_col2,
        zl_val1=zl_val1,
        zl_val2=zl_val2,
    )
end

function _sls_copy_row_metadata(meta_gpu)
    return (
        eq_row=_copy_vector_to_host(meta_gpu.eq_row),
        eq_zero_two_nnz=_copy_vector_to_host(meta_gpu.eq_zero_two_nnz),
        lower_two_nnz=_copy_vector_to_host(meta_gpu.lower_two_nnz),
        zero_lower_two_nnz=_copy_vector_to_host(meta_gpu.zero_lower_two_nnz),
        raw_col1=_copy_vector_to_host(meta_gpu.raw_col1),
        raw_col2=_copy_vector_to_host(meta_gpu.raw_col2),
        raw_val1=_copy_vector_to_host(meta_gpu.raw_val1),
        raw_val2=_copy_vector_to_host(meta_gpu.raw_val2),
        zl_col1=_copy_vector_to_host(meta_gpu.zl_col1),
        zl_col2=_copy_vector_to_host(meta_gpu.zl_col2),
        zl_val1=_copy_vector_to_host(meta_gpu.zl_val1),
        zl_val2=_copy_vector_to_host(meta_gpu.zl_val2),
    )
end

function _sls_gpu_row_metadata(
    A_csr::CuSparseMatrixCSR{Float64,Int32},
    AL::CuVector{Float64},
    AU::CuVector{Float64},
)
    return _sls_copy_row_metadata(_sls_gpu_row_metadata_gpu(A_csr, AL, AU))
end

_sls_meta_is_gpu(meta) = meta !== nothing && meta.eq_row isa CuVector
_sls_host_row_metadata(meta) = _sls_meta_is_gpu(meta) ? _sls_copy_row_metadata(meta) : meta

@inline function _sls_close_host(a::Float64, b::Float64; tol::Float64=STRUCTURAL_L1_SUB_TOL)
    return abs(a - b) <= tol
end

@inline function _sls_row_has_l1_pair(
    c1a::Int32,
    c2a::Int32,
    v1a::Float64,
    v2a::Float64,
    c1b::Int32,
    c2b::Int32,
    v1b::Float64,
    v2b::Float64,
)
    c1a == c1b || return false
    c2a == c2b || return false

    first_is_t =
        v1a > STRUCTURAL_L1_SUB_TOL &&
        _sls_close_host(v1a, v1b) &&
        _sls_close_host(v2a, -v1a) &&
        _sls_close_host(v2b, v1b)
    second_is_t =
        v2a > STRUCTURAL_L1_SUB_TOL &&
        _sls_close_host(v2a, v2b) &&
        _sls_close_host(v1a, -v2a) &&
        _sls_close_host(v1b, v2b)
    return first_is_t || second_is_t
end

@inline function _sls_row_has_outer_pair_signature(
    c1a::Int32,
    c2a::Int32,
    v1a::Float64,
    v2a::Float64,
    c1b::Int32,
    c2b::Int32,
    v1b::Float64,
    v2b::Float64,
    c1c::Int32,
    c2c::Int32,
    v1c::Float64,
    v2c::Float64,
)
    c1a == c1b == c1c || return false
    c2a == c2b == c2c || return false
    return (
        _sls_close_host(v1a, 1.0) && _sls_close_host(v2a, -1.0) &&
        _sls_close_host(v1b, 1.0) && _sls_close_host(v2b, 1.0) &&
        _sls_close_host(v1c, -1.0) && _sls_close_host(v2c, 1.0)
    )
end

@inline function _sls_row_is_signed_two_nnz(
    c1::Int32,
    c2::Int32,
    v1::Float64,
    v2::Float64;
    require_negative::Bool=false,
    require_positive::Bool=false,
)
    c1 > 0 && c2 > 0 || return false
    has_pos = v1 > STRUCTURAL_L1_SUB_TOL || v2 > STRUCTURAL_L1_SUB_TOL
    has_neg = v1 < -STRUCTURAL_L1_SUB_TOL || v2 < -STRUCTURAL_L1_SUB_TOL
    require_positive && !has_pos && return false
    require_negative && !has_neg && return false
    return has_pos && has_neg
end

function _sls_gpu_exact_detect_l1_split_3row(meta, m::Int)
    m >= 3 || return false
    m % 3 == 0 || return false
    nblocks = div(m, 3)
    for i in 1:nblocks
        eq = 3i - 2
        r1 = 3i - 1
        r2 = 3i
        meta.eq_row[eq] != UInt8(1) && return false
        meta.zero_lower_two_nnz[r1] == UInt8(1) || return false
        meta.zero_lower_two_nnz[r2] == UInt8(1) || return false
        _sls_row_has_l1_pair(
            meta.zl_col1[r1], meta.zl_col2[r1], meta.zl_val1[r1], meta.zl_val2[r1],
            meta.zl_col1[r2], meta.zl_col2[r2], meta.zl_val1[r2], meta.zl_val2[r2],
        ) || return false
    end
    return true
end

function _sls_gpu_exact_detect_outer_pair_linked_l1(meta, m::Int)
    m >= 11 || return false
    row = 1
    outer_pairs = 0
    while row + 2 <= m
        meta.zero_lower_two_nnz[row] == UInt8(1) || break
        meta.zero_lower_two_nnz[row + 1] == UInt8(1) || break
        meta.zero_lower_two_nnz[row + 2] == UInt8(1) || break
        _sls_row_has_outer_pair_signature(
            meta.zl_col1[row], meta.zl_col2[row], meta.zl_val1[row], meta.zl_val2[row],
            meta.zl_col1[row + 1], meta.zl_col2[row + 1], meta.zl_val1[row + 1], meta.zl_val2[row + 1],
            meta.zl_col1[row + 2], meta.zl_col2[row + 2], meta.zl_val1[row + 2], meta.zl_val2[row + 2],
        ) || break
        outer_pairs += 1
        row += 3
    end
    outer_pairs > 0 || return false
    row <= m || return false
    (m - row + 1) % 8 == 0 || return false

    while row <= m
        meta.eq_row[row] == UInt8(1) || return false
        meta.zero_lower_two_nnz[row + 1] == UInt8(1) || return false
        meta.zero_lower_two_nnz[row + 2] == UInt8(1) || return false
        _sls_row_has_l1_pair(
            meta.zl_col1[row + 1], meta.zl_col2[row + 1], meta.zl_val1[row + 1], meta.zl_val2[row + 1],
            meta.zl_col1[row + 2], meta.zl_col2[row + 2], meta.zl_val1[row + 2], meta.zl_val2[row + 2],
        ) || return false

        meta.eq_zero_two_nnz[row + 3] == UInt8(1) || return false
        _sls_row_is_signed_two_nnz(
            meta.raw_col1[row + 3], meta.raw_col2[row + 3], meta.raw_val1[row + 3], meta.raw_val2[row + 3];
            require_negative=true,
            require_positive=true,
        ) || return false

        for rr in (row + 4):(row + 7)
            meta.lower_two_nnz[rr] == UInt8(1) || return false
            _sls_row_is_signed_two_nnz(
                meta.raw_col1[rr], meta.raw_col2[rr], meta.raw_val1[rr], meta.raw_val2[rr];
                require_negative=true,
                require_positive=true,
            ) || return false
        end
        row += 8
    end
    return true
end

function _sls_gpu_exact_detect_graph_l1_block4(meta, m::Int)
    m >= 4 || return false
    m % 4 == 0 || return false
    nblocks = div(m, 4)
    for i in 1:nblocks
        eq = 4i - 3
        r1 = 4i - 2
        r2 = 4i - 1
        r3 = 4i
        meta.eq_row[eq] == UInt8(1) || return false
        meta.zero_lower_two_nnz[r1] == UInt8(1) || return false
        meta.zero_lower_two_nnz[r2] == UInt8(1) || return false
        _sls_row_has_l1_pair(
            meta.zl_col1[r1], meta.zl_col2[r1], meta.zl_val1[r1], meta.zl_val2[r1],
            meta.zl_col1[r2], meta.zl_col2[r2], meta.zl_val1[r2], meta.zl_val2[r2],
        ) || return false
        meta.zero_lower_two_nnz[r3] == UInt8(1) || return false
        _sls_row_is_signed_two_nnz(
            meta.zl_col1[r3], meta.zl_col2[r3], meta.zl_val1[r3], meta.zl_val2[r3];
            require_negative=true,
            require_positive=true,
        ) || return false

        shared_with_epi =
            meta.zl_col1[r3] == meta.zl_col1[r1] || meta.zl_col1[r3] == meta.zl_col2[r1] ||
            meta.zl_col2[r3] == meta.zl_col1[r1] || meta.zl_col2[r3] == meta.zl_col2[r1]
        shared_with_epi || return false
    end
    return true
end

function _kernel_structural_l1_split_exact_detect!(
    match_flag,
    eq_row,
    eq_zero_two_nnz,
    lower_two_nnz,
    zero_lower_two_nnz,
    raw_col1,
    raw_col2,
    raw_val1,
    raw_val2,
    zl_col1,
    zl_col2,
    zl_val1,
    zl_val2,
    m::Int32,
)
    if threadIdx().x == 1 && blockIdx().x == 1
        meta = (
            eq_row=eq_row,
            eq_zero_two_nnz=eq_zero_two_nnz,
            lower_two_nnz=lower_two_nnz,
            zero_lower_two_nnz=zero_lower_two_nnz,
            raw_col1=raw_col1,
            raw_col2=raw_col2,
            raw_val1=raw_val1,
            raw_val2=raw_val2,
            zl_col1=zl_col1,
            zl_col2=zl_col2,
            zl_val1=zl_val1,
            zl_val2=zl_val2,
        )
        @inbounds match_flag[1] =
            _sls_gpu_exact_detect_l1_split_3row(meta, Int(m)) ? UInt8(1) : UInt8(0)
    end
    return
end

function _kernel_structural_outer_pair_exact_detect!(
    match_flag,
    eq_row,
    eq_zero_two_nnz,
    lower_two_nnz,
    zero_lower_two_nnz,
    raw_col1,
    raw_col2,
    raw_val1,
    raw_val2,
    zl_col1,
    zl_col2,
    zl_val1,
    zl_val2,
    m::Int32,
)
    if threadIdx().x == 1 && blockIdx().x == 1
        meta = (
            eq_row=eq_row,
            eq_zero_two_nnz=eq_zero_two_nnz,
            lower_two_nnz=lower_two_nnz,
            zero_lower_two_nnz=zero_lower_two_nnz,
            raw_col1=raw_col1,
            raw_col2=raw_col2,
            raw_val1=raw_val1,
            raw_val2=raw_val2,
            zl_col1=zl_col1,
            zl_col2=zl_col2,
            zl_val1=zl_val1,
            zl_val2=zl_val2,
        )
        @inbounds match_flag[1] =
            _sls_gpu_exact_detect_outer_pair_linked_l1(meta, Int(m)) ? UInt8(1) : UInt8(0)
    end
    return
end

function _kernel_structural_graph_exact_detect!(
    match_flag,
    eq_row,
    eq_zero_two_nnz,
    lower_two_nnz,
    zero_lower_two_nnz,
    raw_col1,
    raw_col2,
    raw_val1,
    raw_val2,
    zl_col1,
    zl_col2,
    zl_val1,
    zl_val2,
    m::Int32,
)
    if threadIdx().x == 1 && blockIdx().x == 1
        meta = (
            eq_row=eq_row,
            eq_zero_two_nnz=eq_zero_two_nnz,
            lower_two_nnz=lower_two_nnz,
            zero_lower_two_nnz=zero_lower_two_nnz,
            raw_col1=raw_col1,
            raw_col2=raw_col2,
            raw_val1=raw_val1,
            raw_val2=raw_val2,
            zl_col1=zl_col1,
            zl_col2=zl_col2,
            zl_val1=zl_val1,
            zl_val2=zl_val2,
        )
        @inbounds match_flag[1] =
            _sls_gpu_exact_detect_graph_l1_block4(meta, Int(m)) ? UInt8(1) : UInt8(0)
    end
    return
end

function _sls_gpu_exact_detect_candidate(meta_gpu, m::Int, candidate::Symbol)
    match_flag = CUDA.zeros(UInt8, 1)
    if candidate == :graph_l1_substitution
        @cuda threads=1 blocks=1 _kernel_structural_graph_exact_detect!(
            match_flag,
            meta_gpu.eq_row,
            meta_gpu.eq_zero_two_nnz,
            meta_gpu.lower_two_nnz,
            meta_gpu.zero_lower_two_nnz,
            meta_gpu.raw_col1,
            meta_gpu.raw_col2,
            meta_gpu.raw_val1,
            meta_gpu.raw_val2,
            meta_gpu.zl_col1,
            meta_gpu.zl_col2,
            meta_gpu.zl_val1,
            meta_gpu.zl_val2,
            Int32(m),
        )
    elseif candidate == :outer_pair_linked_l1
        @cuda threads=1 blocks=1 _kernel_structural_outer_pair_exact_detect!(
            match_flag,
            meta_gpu.eq_row,
            meta_gpu.eq_zero_two_nnz,
            meta_gpu.lower_two_nnz,
            meta_gpu.zero_lower_two_nnz,
            meta_gpu.raw_col1,
            meta_gpu.raw_col2,
            meta_gpu.raw_val1,
            meta_gpu.raw_val2,
            meta_gpu.zl_col1,
            meta_gpu.zl_col2,
            meta_gpu.zl_val1,
            meta_gpu.zl_val2,
            Int32(m),
        )
    elseif candidate == :l1_split_3row
        @cuda threads=1 blocks=1 _kernel_structural_l1_split_exact_detect!(
            match_flag,
            meta_gpu.eq_row,
            meta_gpu.eq_zero_two_nnz,
            meta_gpu.lower_two_nnz,
            meta_gpu.zero_lower_two_nnz,
            meta_gpu.raw_col1,
            meta_gpu.raw_col2,
            meta_gpu.raw_val1,
            meta_gpu.raw_val2,
            meta_gpu.zl_col1,
            meta_gpu.zl_col2,
            meta_gpu.zl_val1,
            meta_gpu.zl_val2,
            Int32(m),
        )
    else
        return false
    end

    return CUDA.@allowscalar Int(match_flag[1]) != 0
end

function _sls_gpu_exact_candidates(
    A_csr::CuSparseMatrixCSR{Float64,Int32},
    AL::CuVector{Float64},
    AU::CuVector{Float64},
    pattern::Symbol,
)
    m, _ = size(A_csr)
    meta_gpu = _sls_gpu_row_metadata_gpu(A_csr, AL, AU)
    requested_patterns = if pattern == :auto
        (:graph_l1_substitution, :outer_pair_linked_l1, :l1_split_3row)
    else
        (pattern,)
    end

    candidates = Symbol[]
    errors = String[]
    for candidate in requested_patterns
        matched = _sls_gpu_exact_detect_candidate(meta_gpu, m, candidate)
        if matched
            push!(candidates, candidate)
        else
            push!(errors, "$(candidate): gpu_exact=0")
        end
    end

    isempty(candidates) && return candidates, errors, nothing
    return candidates, errors, meta_gpu
end

function _sls_gpu_extract_l1_pair(
    meta,
    r1::Int,
    r2::Int,
)
    c1 = Int(meta.zl_col1[r1])
    c2 = Int(meta.zl_col2[r1])
    v11 = meta.zl_val1[r1]
    v12 = meta.zl_val2[r1]
    v21 = meta.zl_val1[r2]
    v22 = meta.zl_val2[r2]

    if v11 > STRUCTURAL_L1_SUB_TOL &&
       _sls_close_host(v11, v21) &&
       _sls_close_host(v12, -v11) &&
       _sls_close_host(v22, v21)
        return (t_col=c1, e_col=c2)
    elseif v12 > STRUCTURAL_L1_SUB_TOL &&
           _sls_close_host(v12, v22) &&
           _sls_close_host(v11, -v12) &&
           _sls_close_host(v21, v22)
        return (t_col=c2, e_col=c1)
    end
    return nothing
end

@inline function _sls_extract_l1_pair_device(
    c1::Int32,
    c2::Int32,
    v11::Float64,
    v12::Float64,
    v21::Float64,
    v22::Float64,
)
    if v11 > STRUCTURAL_L1_SUB_TOL &&
       abs(v11 - v21) <= STRUCTURAL_L1_SUB_TOL &&
       abs(v12 + v11) <= STRUCTURAL_L1_SUB_TOL &&
       abs(v22 - v21) <= STRUCTURAL_L1_SUB_TOL
        return c1, c2, UInt8(1)
    elseif v12 > STRUCTURAL_L1_SUB_TOL &&
           abs(v12 - v22) <= STRUCTURAL_L1_SUB_TOL &&
           abs(v11 + v12) <= STRUCTURAL_L1_SUB_TOL &&
           abs(v21 - v22) <= STRUCTURAL_L1_SUB_TOL
        return c2, c1, UInt8(1)
    end
    return Int32(-1), Int32(-1), UInt8(0)
end

@inline function _sls_extract_l1_orientation_device(
    c1::Int32,
    c2::Int32,
    v11::Float64,
    v12::Float64,
    v21::Float64,
    v22::Float64,
)
    t_col, e_col, ok = _sls_extract_l1_pair_device(c1, c2, v11, v12, v21, v22)
    ok == UInt8(1) || return Int32(-1), Int32(-1), 0.0, UInt8(0)

    rho = if t_col == c1 && e_col == c2
        abs(v12 / v11)
    elseif t_col == c2 && e_col == c1
        abs(v11 / v12)
    else
        0.0
    end
    rho > STRUCTURAL_L1_SUB_TOL || return Int32(-1), Int32(-1), 0.0, UInt8(0)
    return t_col, e_col, rho, UInt8(1)
end

function _sls_host_csr_row_entries(
    row_ptr::Vector{Int32},
    col_val::Vector{Int32},
    nz_val::Vector{Float64},
    row::Int,
)
    lo = Int(row_ptr[row])
    hi = Int(row_ptr[row + 1]) - 1
    if hi < lo
        return Int[], Float64[]
    end
    return Int.(col_val[lo:hi]), collect(nz_val[lo:hi])
end

function _sls_host_at_rows_for_col(
    at_row_ptr::Vector{Int32},
    at_col_val::Vector{Int32},
    col::Int,
)
    lo = Int(at_row_ptr[col])
    hi = Int(at_row_ptr[col + 1]) - 1
    if hi < lo
        return Int[]
    end
    return Int.(at_col_val[lo:hi])
end

function _sls_gpu_extract_l1_orientation(
    meta,
    r1::Int,
    r2::Int,
)
    pair = _sls_gpu_extract_l1_pair(meta, r1, r2)
    pair === nothing && return nothing

    c1 = Int(meta.zl_col1[r1])
    c2 = Int(meta.zl_col2[r1])
    v11 = meta.zl_val1[r1]
    v12 = meta.zl_val2[r1]
    t_col = pair.t_col
    e_col = pair.e_col
    rho = if t_col == c1 && e_col == c2
        abs(v12 / v11)
    elseif t_col == c2 && e_col == c1
        abs(v11 / v12)
    else
        return nothing
    end
    rho > STRUCTURAL_L1_SUB_TOL || return nothing
    return (t_col=t_col, e_col=e_col, rho=rho)
end

function _kernel_structural_l1_split_build_plan_data!(
    status_flag,
    keep_row,
    keep_col,
    new_c,
    new_l,
    new_u,
    c,
    l,
    u,
    rowPtr,
    colVal,
    t_cols,
    e_cols,
    residual_bound_as_free_min::Float64,
    nblocks::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > nblocks
        return
    end

    @inbounds begin
        t_col = t_cols[idx]
        e_col = e_cols[idx]
        eq = Int32(3) * idx - Int32(2)
        r1 = eq + Int32(1)
        r2 = eq + Int32(2)

        e_lower = l[Int(e_col)]
        e_upper = u[Int(e_col)]
        e_is_free =
            (isinf(e_lower) && isinf(e_upper)) ||
            (
                isfinite(residual_bound_as_free_min) &&
                e_lower <= -residual_bound_as_free_min &&
                e_upper >= residual_bound_as_free_min
            )
        valid_bounds =
            e_is_free &&
            abs(l[Int(t_col)]) <= STRUCTURAL_L1_SUB_TOL &&
            isinf(u[Int(t_col)])

        found_e = false
        row_start = rowPtr[Int(eq)]
        row_stop = rowPtr[Int(eq) + 1] - Int32(1)
        for p in row_start:row_stop
            if colVal[Int(p)] == e_col
                found_e = true
                break
            end
        end

        if !(valid_bounds && found_e)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        keep_row[Int(r1)] = UInt8(0)
        keep_row[Int(r2)] = UInt8(0)
        new_c[Int(t_col)] = c[Int(t_col)] + c[Int(e_col)]
        new_c[Int(e_col)] = c[Int(t_col)] - c[Int(e_col)]
        new_l[Int(t_col)] = 0.0
        new_l[Int(e_col)] = 0.0
        new_u[Int(t_col)] = Inf
        new_u[Int(e_col)] = Inf
    end
    return
end

function _kernel_structural_l1_split_extract_pairs!(
    status_flag,
    t_cols,
    e_cols,
    eq_row,
    zero_lower_two_nnz,
    zl_col1,
    zl_col2,
    zl_val1,
    zl_val2,
    nblocks::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > nblocks
        return
    end

    @inbounds begin
        eq = Int32(3) * idx - Int32(2)
        r1 = eq + Int32(1)
        r2 = eq + Int32(2)
        if eq_row[Int(eq)] != UInt8(1) ||
           zero_lower_two_nnz[Int(r1)] != UInt8(1) ||
           zero_lower_two_nnz[Int(r2)] != UInt8(1)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        t_col, e_col, ok = _sls_extract_l1_pair_device(
            zl_col1[Int(r1)],
            zl_col2[Int(r1)],
            zl_val1[Int(r1)],
            zl_val2[Int(r1)],
            zl_val1[Int(r2)],
            zl_val2[Int(r2)],
        )
        if ok != UInt8(1)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end
        t_cols[idx] = t_col
        e_cols[idx] = e_col
    end
    return
end

function _sls_l1_split_pairs_from_gpu_meta(meta_gpu, m::Int)
    m % 3 == 0 || return nothing
    nblocks = div(m, 3)
    t_cols = CuVector{Int32}(undef, nblocks)
    e_cols = CuVector{Int32}(undef, nblocks)
    status_flag = CUDA.zeros(Int32, 1)

    if nblocks > 0
        blocks = cld(nblocks, GPU_PRESOLVE_THREADS)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_structural_l1_split_extract_pairs!(
            status_flag,
            t_cols,
            e_cols,
            meta_gpu.eq_row,
            meta_gpu.zero_lower_two_nnz,
            meta_gpu.zl_col1,
            meta_gpu.zl_col2,
            meta_gpu.zl_val1,
            meta_gpu.zl_val2,
            Int32(nblocks),
        )
    end

    status = CUDA.@allowscalar Int(status_flag[1])
    status == 0 || return nothing
    return (
        t_cols=Array(t_cols),
        e_cols=Array(e_cols),
        t_cols_d=t_cols,
        e_cols_d=e_cols,
    )
end

@inline function _sls_outer_map_col_device(free_to_bound, col::Int32)
    mapped = @inbounds free_to_bound[Int(col)]
    return mapped == Int32(-1) ? col : mapped
end

@inline function _sls_outer_targets_overlap(
    a1::Int32,
    a2::Int32,
    acount::Int32,
    b1::Int32,
    b2::Int32,
    bcount::Int32,
)
    a1 == b1 && return true
    bcount == Int32(2) && a1 == b2 && return true
    if acount == Int32(2)
        a2 == b1 && return true
        bcount == Int32(2) && a2 == b2 && return true
    end
    return false
end

function _kernel_structural_outer_pair_apply!(
    status_flag,
    keep_row,
    keep_col,
    new_c,
    c,
    l,
    u,
    bound_cols,
    free_cols,
    pair_count::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > pair_count
        return
    end

    @inbounds begin
        bound_col = bound_cols[idx]
        free_col = free_cols[idx]
        valid_bounds =
            l[Int(bound_col)] == 0.0 &&
            isfinite(u[Int(bound_col)]) &&
            isinf(l[Int(free_col)]) &&
            isinf(u[Int(free_col)])
        if !valid_bounds
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        row1 = Int32(3) * idx - Int32(2)
        keep_row[Int(row1)] = UInt8(0)
        keep_row[Int(row1 + Int32(1))] = UInt8(0)
        keep_row[Int(row1 + Int32(2))] = UInt8(0)
        keep_col[Int(free_col)] = UInt8(0)
        new_c[Int(bound_col)] = c[Int(bound_col)] + c[Int(free_col)]
    end
    return
end

function _kernel_structural_outer_block_apply!(
    status_flag,
    keep_row,
    keep_col,
    new_c,
    new_l,
    new_u,
    c,
    l,
    u,
    rowPtr,
    colVal,
    free_to_bound,
    q_cols,
    e_cols,
    s_cols,
    alphas,
    local_x_cols,
    start_row::Int32,
    block_count::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > block_count
        return
    end

    @inbounds begin
        dense_row = start_row + (idx - Int32(1)) * Int32(8)
        ep1 = dense_row + Int32(1)
        ep2 = dense_row + Int32(2)
        link = dense_row + Int32(3)
        q_col = q_cols[idx]
        e_col = e_cols[idx]
        s_col = s_cols[idx]
        alpha = alphas[idx]

        valid_bounds =
            isinf(l[Int(e_col)]) &&
            isinf(u[Int(e_col)]) &&
            l[Int(q_col)] == 0.0 &&
            isinf(u[Int(q_col)]) &&
            l[Int(s_col)] == 0.0 &&
            isinf(u[Int(s_col)])
        if !valid_bounds
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        for k in Int32(1):Int32(4)
            x_col = local_x_cols[(idx - Int32(1)) * Int32(4) + k]
            if !(l[Int(x_col)] == 0.0 && isinf(u[Int(x_col)]) && c[Int(x_col)] > STRUCTURAL_L1_SUB_TOL)
                CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
                return
            end
        end

        found_e = false
        duplicate_target = false
        row_start = rowPtr[Int(dense_row)]
        row_stop = rowPtr[Int(dense_row) + 1] - Int32(1)
        for p in row_start:row_stop
            col_p = colVal[Int(p)]
            p_count = Int32(1)
            p_t1 = _sls_outer_map_col_device(free_to_bound, col_p)
            p_t2 = Int32(-1)
            if col_p == e_col
                found_e = true
                p_count = Int32(2)
                p_t1 = q_col
                p_t2 = e_col
                p_t1 == p_t2 && (duplicate_target = true)
            end

            for q in (p + Int32(1)):row_stop
                col_q = colVal[Int(q)]
                q_count = Int32(1)
                q_t1 = _sls_outer_map_col_device(free_to_bound, col_q)
                q_t2 = Int32(-1)
                if col_q == e_col
                    q_count = Int32(2)
                    q_t1 = q_col
                    q_t2 = e_col
                    q_t1 == q_t2 && (duplicate_target = true)
                end
                if _sls_outer_targets_overlap(p_t1, p_t2, p_count, q_t1, q_t2, q_count)
                    duplicate_target = true
                end
            end
        end

        if !(found_e && !duplicate_target)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        keep_row[Int(ep1)] = UInt8(0)
        keep_row[Int(ep2)] = UInt8(0)
        keep_row[Int(link)] = UInt8(0)
        keep_col[Int(s_col)] = UInt8(0)
        new_c[Int(q_col)] = c[Int(e_col)] + c[Int(q_col)] + alpha * c[Int(s_col)]
        new_c[Int(e_col)] = -c[Int(e_col)] + c[Int(q_col)] + alpha * c[Int(s_col)]
        new_l[Int(q_col)] = 0.0
        new_l[Int(e_col)] = 0.0
        new_u[Int(q_col)] = Inf
        new_u[Int(e_col)] = Inf
    end
    return
end

@inline function _sls_outer_pair_signature_device(
    c1a::Int32,
    c2a::Int32,
    v1a::Float64,
    v2a::Float64,
    c1b::Int32,
    c2b::Int32,
    v1b::Float64,
    v2b::Float64,
    c1c::Int32,
    c2c::Int32,
    v1c::Float64,
    v2c::Float64,
)
    c1a == c1b == c1c || return false
    c2a == c2b == c2c || return false
    return (
        abs(v1a - 1.0) <= STRUCTURAL_L1_SUB_TOL && abs(v2a + 1.0) <= STRUCTURAL_L1_SUB_TOL &&
        abs(v1b - 1.0) <= STRUCTURAL_L1_SUB_TOL && abs(v2b - 1.0) <= STRUCTURAL_L1_SUB_TOL &&
        abs(v1c + 1.0) <= STRUCTURAL_L1_SUB_TOL && abs(v2c - 1.0) <= STRUCTURAL_L1_SUB_TOL
    )
end

function _kernel_structural_outer_count_pairs!(
    status_flag,
    pair_count,
    start_row_out,
    zero_lower_two_nnz,
    zl_col1,
    zl_col2,
    zl_val1,
    zl_val2,
    m::Int32,
)
    if threadIdx().x == 1 && blockIdx().x == 1
        row = Int32(1)
        count = Int32(0)
        while row + Int32(2) <= m
            ok =
                zero_lower_two_nnz[Int(row)] == UInt8(1) &&
                zero_lower_two_nnz[Int(row + Int32(1))] == UInt8(1) &&
                zero_lower_two_nnz[Int(row + Int32(2))] == UInt8(1) &&
                _sls_outer_pair_signature_device(
                    zl_col1[Int(row)], zl_col2[Int(row)], zl_val1[Int(row)], zl_val2[Int(row)],
                    zl_col1[Int(row + Int32(1))], zl_col2[Int(row + Int32(1))],
                    zl_val1[Int(row + Int32(1))], zl_val2[Int(row + Int32(1))],
                    zl_col1[Int(row + Int32(2))], zl_col2[Int(row + Int32(2))],
                    zl_val1[Int(row + Int32(2))], zl_val2[Int(row + Int32(2))],
                )
            ok || break
            count += Int32(1)
            row += Int32(3)
        end

        pair_count[1] = count
        start_row_out[1] = row
        if count == Int32(0) || row > m || ((m - row + Int32(1)) % Int32(8)) != Int32(0)
            status_flag[1] = Int32(1)
        end
    end
    return
end

function _kernel_structural_outer_extract_pairs!(
    bound_cols,
    free_cols,
    zl_col1,
    zl_col2,
    pair_count::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= pair_count
        row = Int32(3) * idx - Int32(2)
        @inbounds bound_cols[idx] = zl_col1[Int(row)]
        @inbounds free_cols[idx] = zl_col2[Int(row)]
    end
    return
end

function _kernel_structural_outer_build_free_to_bound!(
    free_to_bound,
    bound_cols,
    free_cols,
    pair_count::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= pair_count
        @inbounds free_to_bound[Int(free_cols[idx])] = bound_cols[idx]
    end
    return
end

function _kernel_structural_outer_extract_blocks!(
    status_flag,
    q_cols,
    e_cols,
    s_cols,
    alphas,
    local_x_cols,
    free_to_bound,
    eq_row,
    eq_zero_two_nnz,
    lower_two_nnz,
    zero_lower_two_nnz,
    raw_col1,
    raw_col2,
    raw_val1,
    raw_val2,
    zl_col1,
    zl_col2,
    zl_val1,
    zl_val2,
    start_row::Int32,
    block_count::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > block_count
        return
    end

    @inbounds begin
        dense_row = start_row + (idx - Int32(1)) * Int32(8)
        ep1 = dense_row + Int32(1)
        ep2 = dense_row + Int32(2)
        link = dense_row + Int32(3)

        if eq_row[Int(dense_row)] != UInt8(1) ||
           zero_lower_two_nnz[Int(ep1)] != UInt8(1) ||
           zero_lower_two_nnz[Int(ep2)] != UInt8(1) ||
           eq_zero_two_nnz[Int(link)] != UInt8(1)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        q_col, e_col, ok = _sls_extract_l1_pair_device(
            zl_col1[Int(ep1)],
            zl_col2[Int(ep1)],
            zl_val1[Int(ep1)],
            zl_val2[Int(ep1)],
            zl_val1[Int(ep2)],
            zl_val2[Int(ep2)],
        )
        if ok != UInt8(1)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        link_c1 = raw_col1[Int(link)]
        link_c2 = raw_col2[Int(link)]
        link_v1 = raw_val1[Int(link)]
        link_v2 = raw_val2[Int(link)]
        s_col = Int32(-1)
        alpha = 0.0
        if link_c1 == q_col
            s_col = link_c2
            alpha = link_v1
            abs(link_v2 + 1.0) <= STRUCTURAL_L1_SUB_TOL || (CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1)); return)
        elseif link_c2 == q_col
            s_col = link_c1
            alpha = link_v2
            abs(link_v1 + 1.0) <= STRUCTURAL_L1_SUB_TOL || (CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1)); return)
        else
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end
        alpha > 0.0 || (CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1)); return)
        q_col != e_col || (CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1)); return)

        for k in Int32(1):Int32(4)
            rr = link + k
            lower_two_nnz[Int(rr)] == UInt8(1) || (CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1)); return)
            loc_c1 = raw_col1[Int(rr)]
            loc_c2 = raw_col2[Int(rr)]
            loc_v1 = raw_val1[Int(rr)]
            loc_v2 = raw_val2[Int(rr)]
            x_col = Int32(-1)
            if loc_c1 == s_col
                abs(loc_v1 + 1.0) <= STRUCTURAL_L1_SUB_TOL || (CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1)); return)
                abs(loc_v2 - 1.0) <= STRUCTURAL_L1_SUB_TOL || (CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1)); return)
                x_col = loc_c2
            elseif loc_c2 == s_col
                abs(loc_v2 + 1.0) <= STRUCTURAL_L1_SUB_TOL || (CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1)); return)
                abs(loc_v1 - 1.0) <= STRUCTURAL_L1_SUB_TOL || (CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1)); return)
                x_col = loc_c1
            else
                CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
                return
            end

            mapped_x = _sls_outer_map_col_device(free_to_bound, x_col)
            if mapped_x == q_col || mapped_x == e_col
                CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
                return
            end
            local_x_cols[(idx - Int32(1)) * Int32(4) + k] = x_col
        end

        q_cols[idx] = q_col
        e_cols[idx] = e_col
        s_cols[idx] = s_col
        alphas[idx] = alpha
    end
    return
end

function _sls_outer_pair_linked_l1_from_gpu_meta(meta_gpu, m::Int, n::Int)
    status_flag = CUDA.zeros(Int32, 1)
    pair_count_d = CUDA.zeros(Int32, 1)
    start_row_d = CUDA.zeros(Int32, 1)
    @cuda threads=1 blocks=1 _kernel_structural_outer_count_pairs!(
        status_flag,
        pair_count_d,
        start_row_d,
        meta_gpu.zero_lower_two_nnz,
        meta_gpu.zl_col1,
        meta_gpu.zl_col2,
        meta_gpu.zl_val1,
        meta_gpu.zl_val2,
        Int32(m),
    )

    status = CUDA.@allowscalar Int(status_flag[1])
    status == 0 || return nothing
    pair_count = CUDA.@allowscalar Int(pair_count_d[1])
    start_row = CUDA.@allowscalar Int(start_row_d[1])
    pair_count > 0 || return nothing

    bound_cols_d = CuVector{Int32}(undef, pair_count)
    free_cols_d = CuVector{Int32}(undef, pair_count)
    pair_blocks = cld(pair_count, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=pair_blocks _kernel_structural_outer_extract_pairs!(
        bound_cols_d,
        free_cols_d,
        meta_gpu.zl_col1,
        meta_gpu.zl_col2,
        Int32(pair_count),
    )

    free_to_bound = CUDA.fill(Int32(-1), n)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=pair_blocks _kernel_structural_outer_build_free_to_bound!(
        free_to_bound,
        bound_cols_d,
        free_cols_d,
        Int32(pair_count),
    )

    block_count = div(m - start_row + 1, 8)
    q_cols_d = CuVector{Int32}(undef, block_count)
    e_cols_d = CuVector{Int32}(undef, block_count)
    s_cols_d = CuVector{Int32}(undef, block_count)
    alphas_d = CuVector{Float64}(undef, block_count)
    local_x_cols_d = CuVector{Int32}(undef, 4 * block_count)
    status_flag .= Int32(0)

    if block_count > 0
        block_blocks = cld(block_count, GPU_PRESOLVE_THREADS)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=block_blocks _kernel_structural_outer_extract_blocks!(
            status_flag,
            q_cols_d,
            e_cols_d,
            s_cols_d,
            alphas_d,
            local_x_cols_d,
            free_to_bound,
            meta_gpu.eq_row,
            meta_gpu.eq_zero_two_nnz,
            meta_gpu.lower_two_nnz,
            meta_gpu.zero_lower_two_nnz,
            meta_gpu.raw_col1,
            meta_gpu.raw_col2,
            meta_gpu.raw_val1,
            meta_gpu.raw_val2,
            meta_gpu.zl_col1,
            meta_gpu.zl_col2,
            meta_gpu.zl_val1,
            meta_gpu.zl_val2,
            Int32(start_row),
            Int32(block_count),
        )
    end

    status = CUDA.@allowscalar Int(status_flag[1])
    status == 0 || return nothing
    return (
        start_row=start_row,
        bound_cols=Array(bound_cols_d),
        free_cols=Array(free_cols_d),
        bound_cols_d=bound_cols_d,
        free_cols_d=free_cols_d,
        free_to_bound_d=free_to_bound,
        q_cols=Array(q_cols_d),
        e_cols=Array(e_cols_d),
        s_cols=Array(s_cols_d),
        alphas=Array(alphas_d),
        local_x_cols=Array(local_x_cols_d),
        q_cols_d=q_cols_d,
        e_cols_d=e_cols_d,
        s_cols_d=s_cols_d,
        alphas_d=alphas_d,
        local_x_cols_d=local_x_cols_d,
    )
end

function _kernel_structural_graph_validate_blocks!(
    status_flag,
    block_bad,
    rowPtr,
    colVal,
    atRowPtr,
    atColVal,
    c,
    l,
    u,
    coupling_rows,
    t_cols,
    e_cols,
    block_count::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > block_count
        return
    end

    @inbounds begin
        eq = coupling_rows[idx]
        r1 = eq + Int32(1)
        r2 = eq + Int32(2)
        t_col = t_cols[idx]
        e_col = e_cols[idx]

        valid_bounds =
            isinf(l[Int(e_col)]) &&
            isinf(u[Int(e_col)]) &&
            abs(l[Int(t_col)]) <= STRUCTURAL_L1_SUB_TOL &&
            isinf(u[Int(t_col)]) &&
            c[Int(t_col)] >= -STRUCTURAL_L1_SUB_TOL
        if !valid_bounds
            block_bad[idx] = UInt8(1)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        found_e = false
        duplicate_target = false
        row_start = rowPtr[Int(eq)]
        row_stop = rowPtr[Int(eq) + 1] - Int32(1)
        for p in row_start:row_stop
            col_p = colVal[Int(p)]
            p_count = Int32(1)
            p_t1 = col_p
            p_t2 = Int32(-1)
            if col_p == e_col || col_p == t_col
                found_e |= col_p == e_col
                p_count = Int32(2)
                p_t1 = t_col
                p_t2 = e_col
                p_t1 == p_t2 && (duplicate_target = true)
            end

            for q in (p + Int32(1)):row_stop
                col_q = colVal[Int(q)]
                q_count = Int32(1)
                q_t1 = col_q
                q_t2 = Int32(-1)
                if col_q == e_col || col_q == t_col
                    q_count = Int32(2)
                    q_t1 = t_col
                    q_t2 = e_col
                    q_t1 == q_t2 && (duplicate_target = true)
                end
                if _sls_outer_targets_overlap(p_t1, p_t2, p_count, q_t1, q_t2, q_count)
                    duplicate_target = true
                end
            end
        end

        seen_eq = false
        seen_r1 = false
        seen_r2 = false
        e_start = atRowPtr[Int(e_col)]
        e_stop = atRowPtr[Int(e_col) + 1] - Int32(1)
        for p in e_start:e_stop
            row = atColVal[Int(p)]
            if row == eq
                seen_eq = true
            elseif row == r1
                seen_r1 = true
            elseif row == r2
                seen_r2 = true
            else
                block_bad[idx] = UInt8(1)
                CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
                return
            end
        end

        if !(found_e && !duplicate_target && seen_eq && seen_r1 && seen_r2)
            block_bad[idx] = UInt8(1)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end
    end
    return
end

function _kernel_structural_graph_mark_extra_rows!(
    status_flag,
    block_bad,
    row_to_block,
    row_slack_col,
    row_factor,
    rowPtr,
    colVal,
    nzVal,
    atRowPtr,
    atColVal,
    AL,
    AU,
    coupling_rows,
    t_cols,
    block_count::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > block_count
        return
    end

    @inbounds begin
        t_col = t_cols[idx]
        r1 = coupling_rows[idx] + Int32(1)
        r2 = coupling_rows[idx] + Int32(2)
        t_start = atRowPtr[Int(t_col)]
        t_stop = atRowPtr[Int(t_col) + 1] - Int32(1)
        for p in t_start:t_stop
            row = atColVal[Int(p)]
            if row == r1 || row == r2
                continue
            end

            valid_row = isfinite(AL[Int(row)]) && isinf(AU[Int(row)]) &&
                abs(AL[Int(row)]) <= STRUCTURAL_L1_SUB_TOL
            row_start = rowPtr[Int(row)]
            row_stop = rowPtr[Int(row) + 1] - Int32(1)
            valid_row &= row_stop - row_start + Int32(1) == Int32(2)

            coeff_t = 0.0
            coeff_s = 0.0
            slack_col = Int32(-1)
            if valid_row
                c1 = colVal[Int(row_start)]
                c2 = colVal[Int(row_start + Int32(1))]
                v1 = nzVal[Int(row_start)]
                v2 = nzVal[Int(row_start + Int32(1))]
                if c1 == t_col
                    coeff_t = v1
                    slack_col = c2
                    coeff_s = v2
                elseif c2 == t_col
                    coeff_t = v2
                    slack_col = c1
                    coeff_s = v1
                else
                    valid_row = false
                end
            end

            valid_row &= coeff_t < -STRUCTURAL_L1_SUB_TOL && coeff_s > STRUCTURAL_L1_SUB_TOL
            if !valid_row
                block_bad[idx] = UInt8(1)
                CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
                continue
            end

            existing = row_to_block[Int(row)]
            if existing != Int32(0) && existing != idx
                block_bad[idx] = UInt8(1)
                CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
                continue
            end
            row_to_block[Int(row)] = idx
            row_slack_col[Int(row)] = slack_col
            row_factor[Int(row)] = -coeff_t / coeff_s
        end
    end
    return
end

function _kernel_structural_graph_validate_slacks!(
    block_bad,
    row_removable,
    slack_removable,
    row_to_block,
    row_slack_col,
    atRowPtr,
    atColVal,
    c,
    u,
    m::Int32,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row > m
        return
    end

    @inbounds begin
        block_idx = row_to_block[row]
        block_idx == Int32(0) && return
        slack_col = row_slack_col[row]
        valid_slack =
            slack_col > Int32(0) &&
            abs(c[Int(slack_col)]) <= STRUCTURAL_L1_SUB_TOL &&
            isinf(u[Int(slack_col)])
        if valid_slack
            s_start = atRowPtr[Int(slack_col)]
            s_stop = atRowPtr[Int(slack_col) + 1] - Int32(1)
            for p in s_start:s_stop
                s_row = atColVal[Int(p)]
                if row_to_block[Int(s_row)] != block_idx || row_slack_col[Int(s_row)] != slack_col
                    valid_slack = false
                    break
                end
            end
        end

        if valid_slack
            row_removable[row] = UInt8(1)
            slack_removable[Int(slack_col)] = UInt8(1)
        else
            block_bad[Int(block_idx)] = UInt8(1)
        end
    end
    return
end

function _kernel_structural_graph_apply_plan!(
    status_flag,
    keep_row,
    keep_col,
    new_c,
    new_l,
    new_u,
    block_bad,
    row_removable,
    row_slack_col,
    rowPtr,
    atRowPtr,
    atColVal,
    c,
    l,
    u,
    coupling_rows,
    t_cols,
    e_cols,
    rhos,
    block_count::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > block_count
        return
    end

    @inbounds begin
        if block_bad[idx] != UInt8(0)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        eq = coupling_rows[idx]
        r1 = eq + Int32(1)
        r2 = eq + Int32(2)
        t_col = t_cols[idx]
        e_col = e_cols[idx]
        rho = rhos[idx]

        keep_row[Int(r1)] = UInt8(0)
        keep_row[Int(r2)] = UInt8(0)

        t_start = atRowPtr[Int(t_col)]
        t_stop = atRowPtr[Int(t_col) + 1] - Int32(1)
        for p in t_start:t_stop
            row = atColVal[Int(p)]
            if row == r1 || row == r2
                continue
            end
            if row_removable[Int(row)] == UInt8(1)
                keep_row[Int(row)] = UInt8(0)
                slack_col = row_slack_col[Int(row)]
                slack_col > Int32(0) && (keep_col[Int(slack_col)] = UInt8(0))
            else
                CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
                return
            end
        end

        inv_rho = 1.0 / rho
        new_c[Int(t_col)] = c[Int(t_col)] + c[Int(e_col)] * inv_rho
        new_c[Int(e_col)] = c[Int(t_col)] - c[Int(e_col)] * inv_rho
        new_l[Int(t_col)] = 0.0
        new_l[Int(e_col)] = 0.0
        new_u[Int(t_col)] = Inf
        new_u[Int(e_col)] = Inf
    end
    return
end

function _kernel_structural_graph_extract_blocks!(
    status_flag,
    coupling_rows,
    t_cols,
    e_cols,
    rhos,
    eq_row,
    zero_lower_two_nnz,
    zl_col1,
    zl_col2,
    zl_val1,
    zl_val2,
    nblocks::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > nblocks
        return
    end

    @inbounds begin
        eq = Int32(4) * idx - Int32(3)
        r1 = eq + Int32(1)
        r2 = eq + Int32(2)
        r3 = eq + Int32(3)
        if eq_row[Int(eq)] != UInt8(1) ||
           zero_lower_two_nnz[Int(r1)] != UInt8(1) ||
           zero_lower_two_nnz[Int(r2)] != UInt8(1) ||
           zero_lower_two_nnz[Int(r3)] != UInt8(1)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        t_col, e_col, rho, ok = _sls_extract_l1_orientation_device(
            zl_col1[Int(r1)],
            zl_col2[Int(r1)],
            zl_val1[Int(r1)],
            zl_val2[Int(r1)],
            zl_val1[Int(r2)],
            zl_val2[Int(r2)],
        )
        if ok != UInt8(1)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        r3_c1 = zl_col1[Int(r3)]
        r3_c2 = zl_col2[Int(r3)]
        r3_v1 = zl_val1[Int(r3)]
        r3_v2 = zl_val2[Int(r3)]
        coeff_t = 0.0
        coeff_s = 0.0
        if r3_c1 == t_col
            coeff_t = r3_v1
            coeff_s = r3_v2
        elseif r3_c2 == t_col
            coeff_t = r3_v2
            coeff_s = r3_v1
        else
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end
        if !(coeff_t < -STRUCTURAL_L1_SUB_TOL && coeff_s > STRUCTURAL_L1_SUB_TOL)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        coupling_rows[idx] = eq
        t_cols[idx] = t_col
        e_cols[idx] = e_col
        rhos[idx] = rho
    end
    return
end

function _sls_graph_blocks_from_gpu_meta(meta_gpu, m::Int)
    m % 4 == 0 || return nothing
    nblocks = div(m, 4)
    coupling_rows = CuVector{Int32}(undef, nblocks)
    t_cols = CuVector{Int32}(undef, nblocks)
    e_cols = CuVector{Int32}(undef, nblocks)
    rhos = CuVector{Float64}(undef, nblocks)
    status_flag = CUDA.zeros(Int32, 1)

    if nblocks > 0
        blocks = cld(nblocks, GPU_PRESOLVE_THREADS)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_structural_graph_extract_blocks!(
            status_flag,
            coupling_rows,
            t_cols,
            e_cols,
            rhos,
            meta_gpu.eq_row,
            meta_gpu.zero_lower_two_nnz,
            meta_gpu.zl_col1,
            meta_gpu.zl_col2,
            meta_gpu.zl_val1,
            meta_gpu.zl_val2,
            Int32(nblocks),
        )
    end

    status = CUDA.@allowscalar Int(status_flag[1])
    status == 0 || return nothing
    return (
        coupling_rows=Array(coupling_rows),
        t_cols=Array(t_cols),
        e_cols=Array(e_cols),
        rhos=Array(rhos),
        coupling_rows_d=coupling_rows,
        t_cols_d=t_cols,
        e_cols_d=e_cols,
        rhos_d=rhos,
    )
end

function _kernel_structural_graph_compact_recovery!(
    out_block_idx,
    out_slack_col,
    out_factor,
    prefix,
    row_to_block,
    row_slack_col,
    row_factor,
    row_removable,
    m::Int32,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= m && @inbounds(row_removable[row] != UInt8(0))
        @inbounds write_idx = prefix[row]
        @inbounds out_block_idx[write_idx] = row_to_block[row]
        @inbounds out_slack_col[write_idx] = row_slack_col[row]
        @inbounds out_factor[write_idx] = row_factor[row]
    end
    return
end

function _sls_graph_compact_recovery_to_host(
    row_to_block,
    row_slack_col,
    row_factor,
    row_removable,
)
    m = length(row_removable)
    m == 0 && return Int32[], Int32[], Float64[]

    row_counts = Int32.(row_removable)
    prefix = cumsum(row_counts)
    nitems = Int(_copy_scalar_to_host(prefix, m))
    nitems == 0 && return Int32[], Int32[], Float64[]

    block_idx = CuVector{Int32}(undef, nitems)
    slack_cols = CuVector{Int32}(undef, nitems)
    factors = CuVector{Float64}(undef, nitems)
    blocks = cld(m, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_structural_graph_compact_recovery!(
        block_idx,
        slack_cols,
        factors,
        prefix,
        row_to_block,
        row_slack_col,
        row_factor,
        row_removable,
        Int32(m),
    )

    return Array(block_idx), Array(slack_cols), Array(factors)
end

function _sls_try_gpu_direct_l1_split_result(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    meta,
    ;
    residual_bound_as_free_min::Float64=Inf,
)
    isnothing(plan.new_A) || return nothing

    m, n = size(lp.A)
    m % 3 == 0 || return nothing
    nblocks = div(m, 3)

    splits = StructuralL1SplitRecovery[]
    t_cols = Int32[]
    e_cols = Int32[]
    t_cols_apply_d = nothing
    e_cols_apply_d = nothing

    if _sls_meta_is_gpu(meta)
        extracted = _sls_l1_split_pairs_from_gpu_meta(meta, m)
        extracted === nothing && return nothing
        t_cols = extracted.t_cols
        e_cols = extracted.e_cols
        t_cols_apply_d = extracted.t_cols_d
        e_cols_apply_d = extracted.e_cols_d
        for idx in eachindex(t_cols)
            push!(splits, StructuralL1SplitRecovery(t_cols[idx], e_cols[idx], 1.0))
        end
    else
        for i in 1:nblocks
            eq = 3i - 2
            r1 = 3i - 1
            r2 = 3i
            meta.eq_row[eq] == UInt8(1) || return nothing
            meta.zero_lower_two_nnz[r1] == UInt8(1) || return nothing
            meta.zero_lower_two_nnz[r2] == UInt8(1) || return nothing

            pair = _sls_gpu_extract_l1_pair(meta, r1, r2)
            pair === nothing && return nothing
            t_col = pair.t_col
            e_col = pair.e_col

            push!(splits, StructuralL1SplitRecovery(Int32(t_col), Int32(e_col), 1.0))
            push!(t_cols, Int32(t_col))
            push!(e_cols, Int32(e_col))
        end
    end

    keep_row = CUDA.fill(UInt8(1), m)
    keep_col = CUDA.fill(UInt8(1), n)
    new_c = copy(plan.new_c)
    new_l = copy(plan.new_l)
    new_u = copy(plan.new_u)
    t_cols_apply_d === nothing && (t_cols_apply_d = CuVector(t_cols))
    e_cols_apply_d === nothing && (e_cols_apply_d = CuVector(e_cols))
    status_flag = CUDA.zeros(Int32, 1)

    if nblocks > 0
        blocks = cld(nblocks, GPU_PRESOLVE_THREADS)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_structural_l1_split_build_plan_data!(
            status_flag,
            keep_row,
            keep_col,
            new_c,
            new_l,
            new_u,
            plan.new_c,
            plan.new_l,
            plan.new_u,
            lp.A.rowPtr,
            lp.A.colVal,
            t_cols_apply_d,
            e_cols_apply_d,
            residual_bound_as_free_min,
            Int32(nblocks),
        )
    end

    status = CUDA.@allowscalar Int(status_flag[1])
    status == 0 || return nothing

    return (
        pattern=:l1_split_3row,
        A_new=nothing,
        keep_row=keep_row,
        keep_col=keep_col,
        c_new=new_c,
        AL_new=copy(plan.new_AL),
        AU_new=copy(plan.new_AU),
        l_new=new_l,
        u_new=new_u,
        removed_rows=2 * nblocks,
        removed_cols=0,
        gpu_rewrite=(
            kind=:l1_split_3row,
            t_cols=t_cols,
            e_cols=e_cols,
            t_cols_d=t_cols_apply_d,
            e_cols_d=e_cols_apply_d,
        ),
        primal_recovery=StructuralL1PrimalRecoveryStep(
            :l1_split_3row,
            splits,
            StructuralOuterPairRecovery[],
            StructuralLinkedSlackRecovery[],
            StructuralMaxSlackRecovery[],
        ),
    )
end

function _sls_try_gpu_direct_outer_pair_linked_l1_result(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    meta,
)
    isnothing(plan.new_A) || return nothing

    m, n = size(lp.A)
    outer_pairs = Tuple{Int,Int}[]
    start_row = 0
    blocks = NamedTuple[]
    splits = StructuralL1SplitRecovery[]
    pair_recoveries = StructuralOuterPairRecovery[]
    linked_slacks = StructuralLinkedSlackRecovery[]
    q_cols = Int32[]
    e_cols = Int32[]
    s_cols = Int32[]
    alphas = Float64[]
    local_x_cols = Int32[]
    bound_cols_apply_d = nothing
    free_cols_apply_d = nothing
    free_to_bound_apply_d = nothing
    q_cols_apply_d = nothing
    e_cols_apply_d = nothing
    s_cols_apply_d = nothing
    alphas_apply_d = nothing
    local_x_cols_apply_d = nothing

    if _sls_meta_is_gpu(meta)
        extracted = _sls_outer_pair_linked_l1_from_gpu_meta(meta, m, n)
        extracted === nothing && return nothing
        start_row = Int(extracted.start_row)
        bound_cols_apply_d = extracted.bound_cols_d
        free_cols_apply_d = extracted.free_cols_d
        free_to_bound_apply_d = extracted.free_to_bound_d
        q_cols_apply_d = extracted.q_cols_d
        e_cols_apply_d = extracted.e_cols_d
        s_cols_apply_d = extracted.s_cols_d
        alphas_apply_d = extracted.alphas_d
        local_x_cols_apply_d = extracted.local_x_cols_d

        for idx in eachindex(extracted.bound_cols)
            bound = Int(extracted.bound_cols[idx])
            free = Int(extracted.free_cols[idx])
            push!(outer_pairs, (bound, free))
            push!(pair_recoveries, StructuralOuterPairRecovery(Int32(bound), Int32(free)))
        end

        for idx in eachindex(extracted.q_cols)
            dense_row = start_row + (idx - 1) * 8
            q_col = Int(extracted.q_cols[idx])
            e_col = Int(extracted.e_cols[idx])
            s_col = Int(extracted.s_cols[idx])
            alpha = extracted.alphas[idx]
            push!(splits, StructuralL1SplitRecovery(Int32(q_col), Int32(e_col), 1.0))
            push!(linked_slacks, StructuralLinkedSlackRecovery(Int32(s_col), Int32(q_col), alpha))
            push!(blocks, (
                dense_row=dense_row,
                e_col=e_col,
                q_col=q_col,
                s_col=s_col,
                alpha=alpha,
            ))
            push!(q_cols, Int32(q_col))
            push!(e_cols, Int32(e_col))
            push!(s_cols, Int32(s_col))
            push!(alphas, alpha)
        end
        append!(local_x_cols, extracted.local_x_cols)
    else
        meta = _sls_host_row_metadata(meta)
        row = 1
        while row + 2 <= m
            if !(meta.zero_lower_two_nnz[row] == UInt8(1) &&
                 meta.zero_lower_two_nnz[row + 1] == UInt8(1) &&
                 meta.zero_lower_two_nnz[row + 2] == UInt8(1) &&
                 _sls_row_has_outer_pair_signature(
                     meta.zl_col1[row], meta.zl_col2[row], meta.zl_val1[row], meta.zl_val2[row],
                     meta.zl_col1[row + 1], meta.zl_col2[row + 1], meta.zl_val1[row + 1], meta.zl_val2[row + 1],
                     meta.zl_col1[row + 2], meta.zl_col2[row + 2], meta.zl_val1[row + 2], meta.zl_val2[row + 2],
                 ))
                break
            end
            bound = Int(meta.zl_col1[row])
            free = Int(meta.zl_col2[row])
            push!(outer_pairs, (bound, free))
            push!(pair_recoveries, StructuralOuterPairRecovery(Int32(bound), Int32(free)))
            row += 3
        end
        isempty(outer_pairs) && return nothing
        start_row = row
        row <= m || return nothing
        (m - row + 1) % 8 == 0 || return nothing

        outer_free_to_bound = Dict(free => bound for (bound, free) in outer_pairs)
        while row <= m
            dense_row = row
            ep1 = row + 1
            ep2 = row + 2
            link = row + 3
            local_rows = ntuple(k -> row + 3 + k, 4)

            meta.eq_row[dense_row] == UInt8(1) || return nothing
            meta.zero_lower_two_nnz[ep1] == UInt8(1) || return nothing
            meta.zero_lower_two_nnz[ep2] == UInt8(1) || return nothing
            pair = _sls_gpu_extract_l1_pair(meta, ep1, ep2)
            pair === nothing && return nothing
            q_col = pair.t_col
            e_col = pair.e_col

            meta.eq_zero_two_nnz[link] == UInt8(1) || return nothing
            link_c1 = Int(meta.raw_col1[link])
            link_c2 = Int(meta.raw_col2[link])
            link_v1 = meta.raw_val1[link]
            link_v2 = meta.raw_val2[link]
            if link_c1 == q_col
                s_col = link_c2
                alpha = link_v1
                link_v2 == -1.0 || return nothing
            elseif link_c2 == q_col
                s_col = link_c1
                alpha = link_v2
                link_v1 == -1.0 || return nothing
            else
                return nothing
            end
            alpha > 0.0 || return nothing

            for rr in local_rows
                meta.lower_two_nnz[rr] == UInt8(1) || return nothing
                loc_c1 = Int(meta.raw_col1[rr])
                loc_c2 = Int(meta.raw_col2[rr])
                loc_v1 = meta.raw_val1[rr]
                loc_v2 = meta.raw_val2[rr]
                if loc_c1 == s_col
                    loc_v1 == -1.0 || return nothing
                    loc_v2 == 1.0 || return nothing
                    x_col = loc_c2
                elseif loc_c2 == s_col
                    loc_v2 == -1.0 || return nothing
                    loc_v1 == 1.0 || return nothing
                    x_col = loc_c1
                else
                    return nothing
                end

                local_targets = Int[]
                for col in (s_col, x_col)
                    if col == s_col
                        push!(local_targets, q_col)
                        push!(local_targets, e_col)
                    else
                        push!(local_targets, get(outer_free_to_bound, col, col))
                    end
                end
                _sls_target_cols_unique(local_targets) || return nothing
                push!(local_x_cols, Int32(x_col))
            end

            push!(splits, StructuralL1SplitRecovery(Int32(q_col), Int32(e_col), 1.0))
            push!(linked_slacks, StructuralLinkedSlackRecovery(Int32(s_col), Int32(q_col), alpha))
            push!(blocks, (
                dense_row=dense_row,
                e_col=e_col,
                q_col=q_col,
                s_col=s_col,
                alpha=alpha,
            ))
            push!(q_cols, Int32(q_col))
            push!(e_cols, Int32(e_col))
            push!(s_cols, Int32(s_col))
            push!(alphas, alpha)

            row += 8
        end
    end

    block_count = length(blocks)
    if free_to_bound_apply_d === nothing
        free_to_bound = fill(Int32(-1), n)
        for (bound, free) in outer_pairs
            free_to_bound[free] = Int32(bound)
        end
        free_to_bound_apply_d = CuVector(free_to_bound)
    end

    keep_row = CUDA.fill(UInt8(1), m)
    keep_col = CUDA.fill(UInt8(1), n)
    new_c = copy(plan.new_c)
    new_l = copy(plan.new_l)
    new_u = copy(plan.new_u)
    status_flag = CUDA.zeros(Int32, 1)

    pair_count = length(outer_pairs)
    if pair_count > 0
        pair_blocks = cld(pair_count, GPU_PRESOLVE_THREADS)
        if bound_cols_apply_d === nothing
            bound_cols_apply_d = CuVector(Int32[bound for (bound, _) in outer_pairs])
        end
        if free_cols_apply_d === nothing
            free_cols_apply_d = CuVector(Int32[free for (_, free) in outer_pairs])
        end
        @cuda threads=GPU_PRESOLVE_THREADS blocks=pair_blocks _kernel_structural_outer_pair_apply!(
            status_flag,
            keep_row,
            keep_col,
            new_c,
            plan.new_c,
            plan.new_l,
            plan.new_u,
            bound_cols_apply_d,
            free_cols_apply_d,
            Int32(pair_count),
        )
    end

    if block_count > 0
        block_blocks = cld(block_count, GPU_PRESOLVE_THREADS)
        q_cols_apply_d === nothing && (q_cols_apply_d = CuVector(q_cols))
        e_cols_apply_d === nothing && (e_cols_apply_d = CuVector(e_cols))
        s_cols_apply_d === nothing && (s_cols_apply_d = CuVector(s_cols))
        alphas_apply_d === nothing && (alphas_apply_d = CuVector(alphas))
        local_x_cols_apply_d === nothing && (local_x_cols_apply_d = CuVector(local_x_cols))
        @cuda threads=GPU_PRESOLVE_THREADS blocks=block_blocks _kernel_structural_outer_block_apply!(
            status_flag,
            keep_row,
            keep_col,
            new_c,
            new_l,
            new_u,
            plan.new_c,
            plan.new_l,
            plan.new_u,
            lp.A.rowPtr,
            lp.A.colVal,
            free_to_bound_apply_d,
            q_cols_apply_d,
            e_cols_apply_d,
            s_cols_apply_d,
            alphas_apply_d,
            local_x_cols_apply_d,
            Int32(start_row),
            Int32(block_count),
        )
    end

    status = CUDA.@allowscalar Int(status_flag[1])
    status == 0 || return nothing

    return (
        pattern=:outer_pair_linked_l1,
        A_new=nothing,
        keep_row=keep_row,
        keep_col=keep_col,
        c_new=new_c,
        AL_new=copy(plan.new_AL),
        AU_new=copy(plan.new_AU),
        l_new=new_l,
        u_new=new_u,
        removed_rows=(start_row - 1) + 3 * block_count,
        removed_cols=pair_count + block_count,
        gpu_rewrite=merge(
            _sls_outer_pair_linked_l1_gpu_rewrite_meta(outer_pairs, blocks, start_row),
            (
                bound_cols_d=bound_cols_apply_d,
                free_cols_d=free_cols_apply_d,
                free_to_bound_d=free_to_bound_apply_d,
                q_cols_d=q_cols_apply_d,
                e_cols_d=e_cols_apply_d,
                s_cols_d=s_cols_apply_d,
                alphas_d=alphas_apply_d,
            ),
        ),
        primal_recovery=StructuralL1PrimalRecoveryStep(
            :outer_pair_linked_l1,
            splits,
            pair_recoveries,
            linked_slacks,
            StructuralMaxSlackRecovery[],
        ),
    )
end

function _sls_try_gpu_direct_graph_l1_result(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    meta,
)
    isnothing(plan.new_A) || return nothing

    m, n = size(lp.A)
    m % 4 == 0 || return nothing

    nblocks = div(m, 4)
    blocks = NamedTuple[]
    used_t = Set{Int}()
    used_e = Set{Int}()
    coupling_rows_apply_d = nothing
    t_cols_apply_d = nothing
    e_cols_apply_d = nothing
    rhos_apply_d = nothing

    if _sls_meta_is_gpu(meta)
        extracted = _sls_graph_blocks_from_gpu_meta(meta, m)
        extracted === nothing && return nothing
        coupling_rows_h = extracted.coupling_rows
        t_cols_h = extracted.t_cols
        e_cols_h = extracted.e_cols
        rhos_h = extracted.rhos
        coupling_rows_apply_d = extracted.coupling_rows_d
        t_cols_apply_d = extracted.t_cols_d
        e_cols_apply_d = extracted.e_cols_d
        rhos_apply_d = extracted.rhos_d
        for idx in eachindex(t_cols_h)
            t_col = Int(t_cols_h[idx])
            e_col = Int(e_cols_h[idx])
            (t_col in used_t || e_col in used_e) && return nothing
            push!(used_t, t_col)
            push!(used_e, e_col)
            push!(blocks, (
                coupling_row=Int(coupling_rows_h[idx]),
                t_col=t_col,
                e_col=e_col,
                rho=rhos_h[idx],
            ))
        end
    else
        for i in 1:nblocks
            eq = 4i - 3
            r1 = 4i - 2
            r2 = 4i - 1
            r3 = 4i

            meta.eq_row[eq] == UInt8(1) || return nothing
            meta.zero_lower_two_nnz[r1] == UInt8(1) || return nothing
            meta.zero_lower_two_nnz[r2] == UInt8(1) || return nothing
            meta.zero_lower_two_nnz[r3] == UInt8(1) || return nothing

            orient = _sls_gpu_extract_l1_orientation(meta, r1, r2)
            orient === nothing && return nothing
            t_col = orient.t_col
            e_col = orient.e_col
            rho = orient.rho
            (t_col in used_t || e_col in used_e) && return nothing
            push!(used_t, t_col)
            push!(used_e, e_col)

            r3_c1 = Int(meta.zl_col1[r3])
            r3_c2 = Int(meta.zl_col2[r3])
            r3_v1 = meta.zl_val1[r3]
            r3_v2 = meta.zl_val2[r3]
            if r3_c1 == t_col
                coeff_t = r3_v1
                coeff_s = r3_v2
            elseif r3_c2 == t_col
                coeff_t = r3_v2
                coeff_s = r3_v1
            else
                return nothing
            end
            coeff_t < -STRUCTURAL_L1_SUB_TOL || return nothing
            coeff_s > STRUCTURAL_L1_SUB_TOL || return nothing

            push!(blocks, (coupling_row=eq, t_col=t_col, e_col=e_col, rho=rho))
        end
    end

    block_count = length(blocks)
    coupling_rows = Int32[block.coupling_row for block in blocks]
    t_cols = Int32[block.t_col for block in blocks]
    e_cols = Int32[block.e_col for block in blocks]
    rhos = Float64[block.rho for block in blocks]

    keep_row = CUDA.fill(UInt8(1), m)
    keep_col = CUDA.fill(UInt8(1), n)
    new_c = copy(plan.new_c)
    new_l = copy(plan.new_l)
    new_u = copy(plan.new_u)
    status_flag = CUDA.zeros(Int32, 1)
    block_bad = CUDA.zeros(UInt8, block_count)
    row_to_block = CUDA.zeros(Int32, m)
    row_slack_col = CUDA.fill(Int32(-1), m)
    row_factor = CUDA.zeros(Float64, m)
    row_removable = CUDA.zeros(UInt8, m)
    slack_removable = CUDA.zeros(UInt8, n)

    if block_count > 0
        block_threads = cld(block_count, GPU_PRESOLVE_THREADS)
        coupling_rows_apply_d === nothing && (coupling_rows_apply_d = CuVector(coupling_rows))
        t_cols_apply_d === nothing && (t_cols_apply_d = CuVector(t_cols))
        e_cols_apply_d === nothing && (e_cols_apply_d = CuVector(e_cols))
        rhos_apply_d === nothing && (rhos_apply_d = CuVector(rhos))

        @cuda threads=GPU_PRESOLVE_THREADS blocks=block_threads _kernel_structural_graph_validate_blocks!(
            status_flag,
            block_bad,
            lp.A.rowPtr,
            lp.A.colVal,
            lp.AT.rowPtr,
            lp.AT.colVal,
            plan.new_c,
            plan.new_l,
            plan.new_u,
            coupling_rows_apply_d,
            t_cols_apply_d,
            e_cols_apply_d,
            Int32(block_count),
        )

        @cuda threads=GPU_PRESOLVE_THREADS blocks=block_threads _kernel_structural_graph_mark_extra_rows!(
            status_flag,
            block_bad,
            row_to_block,
            row_slack_col,
            row_factor,
            lp.A.rowPtr,
            lp.A.colVal,
            lp.A.nzVal,
            lp.AT.rowPtr,
            lp.AT.colVal,
            plan.new_AL,
            plan.new_AU,
            coupling_rows_apply_d,
            t_cols_apply_d,
            Int32(block_count),
        )

        row_threads = cld(max(m, 1), GPU_PRESOLVE_THREADS)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=row_threads _kernel_structural_graph_validate_slacks!(
            block_bad,
            row_removable,
            slack_removable,
            row_to_block,
            row_slack_col,
            lp.AT.rowPtr,
            lp.AT.colVal,
            plan.new_c,
            plan.new_u,
            Int32(m),
        )

        @cuda threads=GPU_PRESOLVE_THREADS blocks=block_threads _kernel_structural_graph_apply_plan!(
            status_flag,
            keep_row,
            keep_col,
            new_c,
            new_l,
            new_u,
            block_bad,
            row_removable,
            row_slack_col,
            lp.A.rowPtr,
            lp.AT.rowPtr,
            lp.AT.colVal,
            plan.new_c,
            plan.new_l,
            plan.new_u,
            coupling_rows_apply_d,
            t_cols_apply_d,
            e_cols_apply_d,
            rhos_apply_d,
            Int32(block_count),
        )
    end

    status = CUDA.@allowscalar Int(status_flag[1])
    status == 0 || return nothing

    recovery_block_idx_h, recovery_slack_col_h, recovery_factor_h =
        _sls_graph_compact_recovery_to_host(row_to_block, row_slack_col, row_factor, row_removable)

    splits = StructuralL1SplitRecovery[]
    for block in blocks
        inv_rho = 1.0 / block.rho
        push!(splits, StructuralL1SplitRecovery(Int32(block.t_col), Int32(block.e_col), block.rho))
    end

    slack_rows = Dict{Int,Vector{Tuple{Int,Float64}}}()
    for idx in eachindex(recovery_block_idx_h)
        block_idx = Int(recovery_block_idx_h[idx])
        slack_col = Int(recovery_slack_col_h[idx])
        factor = recovery_factor_h[idx]
        block_idx > 0 && slack_col > 0 || continue
        push!(get!(slack_rows, slack_col, Tuple{Int,Float64}[]), (block_idx, factor))
    end

    slack_recoveries = StructuralMaxSlackRecovery[]
    for slack_col in sort!(collect(keys(slack_rows)))
        rows = slack_rows[slack_col]
        t_cols = Int32[]
        factors = Float64[]
        for (block_idx, factor) in rows
            push!(t_cols, Int32(blocks[block_idx].t_col))
            push!(factors, factor)
        end
        push!(slack_recoveries, StructuralMaxSlackRecovery(Int32(slack_col), t_cols, factors))
    end

    removed_extra_rows = length(recovery_block_idx_h)
    removed_slacks = length(slack_recoveries)

    return (
        pattern=:graph_l1_substitution,
        A_new=nothing,
        keep_row=keep_row,
        keep_col=keep_col,
        c_new=new_c,
        AL_new=copy(plan.new_AL),
        AU_new=copy(plan.new_AU),
        l_new=new_l,
        u_new=new_u,
        removed_rows=2 * block_count + removed_extra_rows,
        removed_cols=removed_slacks,
        gpu_rewrite=merge(
            _sls_graph_gpu_rewrite_meta(blocks),
            (
                coupling_rows_d=coupling_rows_apply_d,
                t_cols_d=t_cols_apply_d,
                e_cols_d=e_cols_apply_d,
                rhos_d=rhos_apply_d,
            ),
        ),
        primal_recovery=StructuralL1PrimalRecoveryStep(
            :graph_l1_substitution,
            splits,
            StructuralOuterPairRecovery[],
            StructuralLinkedSlackRecovery[],
            slack_recoveries,
        ),
    )
end

function _kernel_structural_l1_split_count_rows!(
    row_nnz_new,
    rowPtr_org,
    m,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= m
        if ((row - Int32(1)) % Int32(3)) == Int32(0)
            @inbounds row_nnz_new[row] = (rowPtr_org[row + 1] - rowPtr_org[row]) + Int32(1)
        else
            @inbounds row_nnz_new[row] = Int32(0)
        end
    end
    return
end

function _kernel_structural_l1_split_fill!(
    colVal_new,
    nzVal_new,
    rowPtr_new,
    rowPtr_org,
    colVal_org,
    nzVal_org,
    t_cols,
    e_cols,
    m,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= m && ((row - Int32(1)) % Int32(3)) == Int32(0)
        block_idx = ((row - Int32(1)) ÷ Int32(3)) + Int32(1)
        @inbounds t_col = t_cols[block_idx]
        @inbounds e_col = e_cols[block_idx]
        @inbounds src_first = rowPtr_org[row]
        @inbounds src_last = rowPtr_org[row + 1] - Int32(1)
        @inbounds write_ptr = rowPtr_new[row]
        for p in src_first:src_last
            @inbounds col = colVal_org[p]
            @inbounds val = nzVal_org[p]
            if col == e_col
                @inbounds colVal_new[write_ptr] = t_col
                @inbounds nzVal_new[write_ptr] = val
                write_ptr += Int32(1)
                @inbounds colVal_new[write_ptr] = e_col
                @inbounds nzVal_new[write_ptr] = -val
                write_ptr += Int32(1)
            else
                @inbounds colVal_new[write_ptr] = col
                @inbounds nzVal_new[write_ptr] = val
                write_ptr += Int32(1)
            end
        end
    end
    return
end

function _kernel_structural_outer_count_rows!(
    row_nnz_new,
    rowPtr_org,
    start_row,
    m,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= m
        if row < start_row
            @inbounds row_nnz_new[row] = Int32(0)
        else
            offset = row - start_row
            rem8 = offset % Int32(8)
            if rem8 == Int32(0) || rem8 >= Int32(4)
                @inbounds row_nnz_new[row] = (rowPtr_org[row + 1] - rowPtr_org[row]) + Int32(1)
            else
                @inbounds row_nnz_new[row] = Int32(0)
            end
        end
    end
    return
end

function _kernel_structural_outer_fill!(
    colVal_new,
    nzVal_new,
    rowPtr_new,
    rowPtr_org,
    colVal_org,
    nzVal_org,
    free_to_bound,
    start_row,
    q_cols,
    e_cols,
    s_cols,
    alphas,
    m,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= m && row >= start_row
        offset = row - start_row
        rem8 = offset % Int32(8)
        if rem8 == Int32(0) || rem8 >= Int32(4)
            block_idx = (offset ÷ Int32(8)) + Int32(1)
            @inbounds q_col = q_cols[block_idx]
            @inbounds e_col = e_cols[block_idx]
            @inbounds s_col = s_cols[block_idx]
            @inbounds alpha = alphas[block_idx]
            @inbounds src_first = rowPtr_org[row]
            @inbounds src_last = rowPtr_org[row + 1] - Int32(1)
            @inbounds write_ptr = rowPtr_new[row]
            for p in src_first:src_last
                @inbounds col_org = colVal_org[p]
                @inbounds val = nzVal_org[p]
                mapped = col_org
                @inbounds free_map = free_to_bound[col_org]
                if free_map != Int32(-1)
                    mapped = free_map
                end
                if rem8 == Int32(0) && col_org == e_col
                    @inbounds colVal_new[write_ptr] = q_col
                    @inbounds nzVal_new[write_ptr] = val
                    write_ptr += Int32(1)
                    @inbounds colVal_new[write_ptr] = e_col
                    @inbounds nzVal_new[write_ptr] = -val
                    write_ptr += Int32(1)
                elseif rem8 >= Int32(4) && col_org == s_col
                    coeff = val * alpha
                    @inbounds colVal_new[write_ptr] = q_col
                    @inbounds nzVal_new[write_ptr] = coeff
                    write_ptr += Int32(1)
                    @inbounds colVal_new[write_ptr] = e_col
                    @inbounds nzVal_new[write_ptr] = coeff
                    write_ptr += Int32(1)
                else
                    @inbounds colVal_new[write_ptr] = mapped
                    @inbounds nzVal_new[write_ptr] = val
                    write_ptr += Int32(1)
                end
            end
        end
    end
    return
end

function _kernel_structural_graph_count_rows!(
    row_nnz_new,
    coupling_keep,
    rowPtr_org,
    m,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= m
        if @inbounds coupling_keep[row] != UInt8(0)
            @inbounds row_nnz_new[row] = (rowPtr_org[row + 1] - rowPtr_org[row]) + Int32(1)
        else
            @inbounds row_nnz_new[row] = Int32(0)
        end
    end
    return
end

function _kernel_structural_graph_build_rewrite_maps!(
    coupling_keep,
    row_to_block,
    coupling_rows,
    block_count::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= block_count
        @inbounds row = coupling_rows[idx]
        @inbounds coupling_keep[Int(row)] = UInt8(1)
        @inbounds row_to_block[Int(row)] = idx
    end
    return
end

function _kernel_structural_graph_fill!(
    colVal_new,
    nzVal_new,
    rowPtr_new,
    rowPtr_org,
    colVal_org,
    nzVal_org,
    coupling_keep,
    row_to_block,
    t_cols,
    e_cols,
    rhos,
    m,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= m && @inbounds(coupling_keep[row] != UInt8(0))
        @inbounds block_idx = row_to_block[row]
        @inbounds t_col = t_cols[block_idx]
        @inbounds e_col = e_cols[block_idx]
        @inbounds rho = rhos[block_idx]
        @inbounds src_first = rowPtr_org[row]
        @inbounds src_last = rowPtr_org[row + 1] - Int32(1)
        @inbounds write_ptr = rowPtr_new[row]
        for p in src_first:src_last
            @inbounds col = colVal_org[p]
            @inbounds val = nzVal_org[p]
            if col == e_col
                coeff = val / rho
                @inbounds colVal_new[write_ptr] = t_col
                @inbounds nzVal_new[write_ptr] = coeff
                write_ptr += Int32(1)
                @inbounds colVal_new[write_ptr] = e_col
                @inbounds nzVal_new[write_ptr] = -coeff
                write_ptr += Int32(1)
            elseif col == t_col
                @inbounds colVal_new[write_ptr] = t_col
                @inbounds nzVal_new[write_ptr] = val
                write_ptr += Int32(1)
                @inbounds colVal_new[write_ptr] = e_col
                @inbounds nzVal_new[write_ptr] = val
                write_ptr += Int32(1)
            else
                @inbounds colVal_new[write_ptr] = col
                @inbounds nzVal_new[write_ptr] = val
                write_ptr += Int32(1)
            end
        end
    end
    return
end

function _sls_build_gpu_rewrite_csr(
    A_source::CuSparseMatrixCSR{Float64,Int32},
    gpu_rewrite,
    keep_row,
    keep_col,
)
    kind = gpu_rewrite.kind
    m, n = size(A_source)
    row_nnz_new = CUDA.zeros(Int32, m)
    blocks = cld(max(m, 1), GPU_PRESOLVE_THREADS)

    if kind == :l1_split_3row
        t_cols = hasproperty(gpu_rewrite, :t_cols_d) ? gpu_rewrite.t_cols_d : CuVector(gpu_rewrite.t_cols)
        e_cols = hasproperty(gpu_rewrite, :e_cols_d) ? gpu_rewrite.e_cols_d : CuVector(gpu_rewrite.e_cols)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_structural_l1_split_count_rows!(
            row_nnz_new,
            A_source.rowPtr,
            Int32(m),
        )
        prefix = cumsum(row_nnz_new)
        rowPtr_new = CUDA.fill(Int32(1), m + 1)
        rowPtr_new[2:end] .= prefix .+ Int32(1)
        nnz_new = m == 0 ? 0 : Int(_copy_scalar_to_host(prefix, m))
        colVal_new = CuVector{Int32}(undef, nnz_new)
        nzVal_new = CuVector{Float64}(undef, nnz_new)
        if nnz_new > 0
            @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_structural_l1_split_fill!(
                colVal_new,
                nzVal_new,
                rowPtr_new,
                A_source.rowPtr,
                A_source.colVal,
                A_source.nzVal,
                t_cols,
                e_cols,
                Int32(m),
            )
        end
        return CuSparseMatrixCSR(rowPtr_new, colVal_new, nzVal_new, (m, n))
    elseif kind == :outer_pair_linked_l1
        bound_cols = hasproperty(gpu_rewrite, :bound_cols_d) ? gpu_rewrite.bound_cols_d : CuVector(gpu_rewrite.bound_cols)
        free_cols = hasproperty(gpu_rewrite, :free_cols_d) ? gpu_rewrite.free_cols_d : CuVector(gpu_rewrite.free_cols)
        free_to_bound_d = if hasproperty(gpu_rewrite, :free_to_bound_d)
            gpu_rewrite.free_to_bound_d
        else
            pair_count = length(gpu_rewrite.bound_cols)
            map_d = CUDA.fill(Int32(-1), n)
            if pair_count > 0
                pair_blocks = cld(pair_count, GPU_PRESOLVE_THREADS)
                @cuda threads=GPU_PRESOLVE_THREADS blocks=pair_blocks _kernel_structural_outer_build_free_to_bound!(
                    map_d,
                    bound_cols,
                    free_cols,
                    Int32(pair_count),
                )
            end
            map_d
        end
        q_cols = hasproperty(gpu_rewrite, :q_cols_d) ? gpu_rewrite.q_cols_d : CuVector(gpu_rewrite.q_cols)
        e_cols = hasproperty(gpu_rewrite, :e_cols_d) ? gpu_rewrite.e_cols_d : CuVector(gpu_rewrite.e_cols)
        s_cols = hasproperty(gpu_rewrite, :s_cols_d) ? gpu_rewrite.s_cols_d : CuVector(gpu_rewrite.s_cols)
        alphas = hasproperty(gpu_rewrite, :alphas_d) ? gpu_rewrite.alphas_d : CuVector(gpu_rewrite.alphas)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_structural_outer_count_rows!(
            row_nnz_new,
            A_source.rowPtr,
            gpu_rewrite.start_row,
            Int32(m),
        )
        prefix = cumsum(row_nnz_new)
        rowPtr_new = CUDA.fill(Int32(1), m + 1)
        rowPtr_new[2:end] .= prefix .+ Int32(1)
        nnz_new = m == 0 ? 0 : Int(_copy_scalar_to_host(prefix, m))
        colVal_new = CuVector{Int32}(undef, nnz_new)
        nzVal_new = CuVector{Float64}(undef, nnz_new)
        if nnz_new > 0
            @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_structural_outer_fill!(
                colVal_new,
                nzVal_new,
                rowPtr_new,
                A_source.rowPtr,
                A_source.colVal,
                A_source.nzVal,
                free_to_bound_d,
                gpu_rewrite.start_row,
                q_cols,
                e_cols,
                s_cols,
                alphas,
                Int32(m),
            )
        end
        return CuSparseMatrixCSR(rowPtr_new, colVal_new, nzVal_new, (m, n))
    elseif kind == :graph_l1_substitution
        coupling_rows = hasproperty(gpu_rewrite, :coupling_rows_d) ? gpu_rewrite.coupling_rows_d : CuVector(gpu_rewrite.coupling_rows)
        t_cols = hasproperty(gpu_rewrite, :t_cols_d) ? gpu_rewrite.t_cols_d : CuVector(gpu_rewrite.t_cols)
        e_cols = hasproperty(gpu_rewrite, :e_cols_d) ? gpu_rewrite.e_cols_d : CuVector(gpu_rewrite.e_cols)
        rhos = hasproperty(gpu_rewrite, :rhos_d) ? gpu_rewrite.rhos_d : CuVector(gpu_rewrite.rhos)
        coupling_keep_d = CUDA.zeros(UInt8, m)
        row_to_block_d = CUDA.fill(Int32(-1), m)
        block_count = length(gpu_rewrite.coupling_rows)
        if block_count > 0
            map_blocks = cld(block_count, GPU_PRESOLVE_THREADS)
            @cuda threads=GPU_PRESOLVE_THREADS blocks=map_blocks _kernel_structural_graph_build_rewrite_maps!(
                coupling_keep_d,
                row_to_block_d,
                coupling_rows,
                Int32(block_count),
            )
        end
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_structural_graph_count_rows!(
            row_nnz_new,
            coupling_keep_d,
            A_source.rowPtr,
            Int32(m),
        )
        prefix = cumsum(row_nnz_new)
        rowPtr_new = CUDA.fill(Int32(1), m + 1)
        rowPtr_new[2:end] .= prefix .+ Int32(1)
        nnz_new = m == 0 ? 0 : Int(_copy_scalar_to_host(prefix, m))
        colVal_new = CuVector{Int32}(undef, nnz_new)
        nzVal_new = CuVector{Float64}(undef, nnz_new)
        if nnz_new > 0
            @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_structural_graph_fill!(
                colVal_new,
                nzVal_new,
                rowPtr_new,
                A_source.rowPtr,
                A_source.colVal,
                A_source.nzVal,
                coupling_keep_d,
                row_to_block_d,
                t_cols,
                e_cols,
                rhos,
                Int32(m),
            )
        end
        return CuSparseMatrixCSR(rowPtr_new, colVal_new, nzVal_new, (m, n))
    end

    return nothing
end

_sls_to_cuvector(v::CuVector) = v
_sls_to_cuvector(v) = CuVector(v)
_sls_mask_to_cuvector(mask::CuVector{UInt8}) = mask
_sls_mask_to_cuvector(mask) = CuVector(UInt8.(mask))

@inline function _sls_screen_pattern_code(pattern::Symbol)
    pattern == :l1_split_3row && return Int32(1)
    pattern == :outer_pair_linked_l1 && return Int32(2)
    pattern == :graph_l1_substitution && return Int32(3)
    pattern == :auto && return Int32(4)
    return Int32(0)
end

function _kernel_structural_screen_counts!(
    counts,
    row_nnz,
    col_nnz,
    m::Int32,
    n::Int32,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= m
        @inbounds rnz = row_nnz[idx]
        rnz == Int32(2) && (CUDA.@atomic counts[1] += Int32(1))
        rnz >= Int32(3) && (CUDA.@atomic counts[2] += Int32(1))
    end
    if idx <= n
        @inbounds cnz = col_nnz[idx]
        cnz >= Int32(2) && (CUDA.@atomic counts[3] += Int32(1))
    end
    return
end

function _kernel_structural_screen_flag!(
    screen_flag,
    counts,
    m::Int32,
    n::Int32,
    pattern_code::Int32,
)
    if threadIdx().x == 1 && blockIdx().x == 1
        if m == Int32(0) || n == Int32(0)
            @inbounds screen_flag[1] = UInt8(0)
            return
        end

        @inbounds two_nnz_rows = counts[1]
        @inbounds dense_rows = counts[2]
        @inbounds multiuse_cols = counts[3]
        has_basic_l1_signature =
            multiuse_cols >= Int32(2) &&
            two_nnz_rows >= Int32(2) &&
            (dense_rows >= Int32(1) || two_nnz_rows >= Int32(3))

        passed = if pattern_code == Int32(1)
            m >= Int32(3) && multiuse_cols >= Int32(2) && two_nnz_rows >= Int32(2)
        elseif pattern_code == Int32(2)
            m >= Int32(11) && two_nnz_rows >= Int32(6) && dense_rows >= Int32(1) && multiuse_cols >= Int32(4)
        elseif pattern_code == Int32(3)
            m >= Int32(3) && has_basic_l1_signature
        elseif pattern_code == Int32(4)
            has_basic_l1_signature ||
            (m >= Int32(3) && multiuse_cols >= Int32(2) && two_nnz_rows >= Int32(2))
        else
            true
        end
        @inbounds screen_flag[1] = passed ? UInt8(1) : UInt8(0)
    end
    return
end

@inline function _sls_prefix_row2(
    row::Int32,
    rowPtr,
    colVal,
    nzVal,
    AL,
    AU,
)
    @inbounds al = AL[row]
    @inbounds au = AU[row]
    @inbounds lo = rowPtr[row]
    @inbounds hi = rowPtr[row + Int32(1)] - Int32(1)
    nnz = hi >= lo ? hi - lo + Int32(1) : Int32(0)
    is_eq = isfinite(al) && isfinite(au) && abs(al - au) <= STRUCTURAL_L1_SUB_TOL

    eq_zero_two_nnz = UInt8(0)
    lower_two_nnz = UInt8(0)
    zero_lower_two_nnz = UInt8(0)
    raw_col1 = Int32(-1)
    raw_col2 = Int32(-1)
    raw_val1 = 0.0
    raw_val2 = 0.0
    zl_col1 = Int32(-1)
    zl_col2 = Int32(-1)
    zl_val1 = 0.0
    zl_val2 = 0.0

    if nnz == Int32(2)
        @inbounds c1 = colVal[lo]
        @inbounds c2 = colVal[lo + Int32(1)]
        @inbounds v1 = nzVal[lo]
        @inbounds v2 = nzVal[lo + Int32(1)]
        if c2 < c1
            c1, c2 = c2, c1
            v1, v2 = v2, v1
        end

        raw_col1 = c1
        raw_col2 = c2
        raw_val1 = v1
        raw_val2 = v2

        if is_eq && abs(al) <= STRUCTURAL_L1_SUB_TOL
            eq_zero_two_nnz = UInt8(1)
        end

        if isfinite(al) && isinf(au)
            lower_two_nnz = UInt8(1)
        end

        is_zero_lower = isfinite(al) && isinf(au) && abs(al) <= STRUCTURAL_L1_SUB_TOL
        is_zero_upper = isinf(al) && isfinite(au) && abs(au) <= STRUCTURAL_L1_SUB_TOL
        if is_zero_lower || is_zero_upper
            scale = is_zero_lower ? 1.0 : -1.0
            zero_lower_two_nnz = UInt8(1)
            zl_col1 = c1
            zl_col2 = c2
            zl_val1 = scale * v1
            zl_val2 = scale * v2
        end
    end

    return (
        is_eq ? UInt8(1) : UInt8(0),
        eq_zero_two_nnz,
        lower_two_nnz,
        zero_lower_two_nnz,
        raw_col1,
        raw_col2,
        raw_val1,
        raw_val2,
        zl_col1,
        zl_col2,
        zl_val1,
        zl_val2,
    )
end

@inline function _sls_prefix_possible_l1_split(
    rowPtr,
    colVal,
    nzVal,
    AL,
    AU,
    m::Int32,
)
    m >= Int32(3) || return false
    m % Int32(3) == Int32(0) || return false
    eq = _sls_prefix_row2(Int32(1), rowPtr, colVal, nzVal, AL, AU)
    r1 = _sls_prefix_row2(Int32(2), rowPtr, colVal, nzVal, AL, AU)
    r2 = _sls_prefix_row2(Int32(3), rowPtr, colVal, nzVal, AL, AU)
    eq[1] == UInt8(1) || return false
    r1[4] == UInt8(1) || return false
    r2[4] == UInt8(1) || return false
    return _sls_row_has_l1_pair(
        r1[9], r1[10], r1[11], r1[12],
        r2[9], r2[10], r2[11], r2[12],
    )
end

@inline function _sls_prefix_possible_outer_pair(
    rowPtr,
    colVal,
    nzVal,
    AL,
    AU,
    m::Int32,
)
    m >= Int32(11) || return false
    r1 = _sls_prefix_row2(Int32(1), rowPtr, colVal, nzVal, AL, AU)
    r2 = _sls_prefix_row2(Int32(2), rowPtr, colVal, nzVal, AL, AU)
    r3 = _sls_prefix_row2(Int32(3), rowPtr, colVal, nzVal, AL, AU)
    r1[4] == UInt8(1) || return false
    r2[4] == UInt8(1) || return false
    r3[4] == UInt8(1) || return false
    return _sls_row_has_outer_pair_signature(
        r1[9], r1[10], r1[11], r1[12],
        r2[9], r2[10], r2[11], r2[12],
        r3[9], r3[10], r3[11], r3[12],
    )
end

@inline function _sls_prefix_possible_graph(
    rowPtr,
    colVal,
    nzVal,
    AL,
    AU,
    m::Int32,
)
    m >= Int32(4) || return false
    m % Int32(4) == Int32(0) || return false
    eq = _sls_prefix_row2(Int32(1), rowPtr, colVal, nzVal, AL, AU)
    r1 = _sls_prefix_row2(Int32(2), rowPtr, colVal, nzVal, AL, AU)
    r2 = _sls_prefix_row2(Int32(3), rowPtr, colVal, nzVal, AL, AU)
    r3 = _sls_prefix_row2(Int32(4), rowPtr, colVal, nzVal, AL, AU)
    eq[1] == UInt8(1) || return false
    r1[4] == UInt8(1) || return false
    r2[4] == UInt8(1) || return false
    _sls_row_has_l1_pair(
        r1[9], r1[10], r1[11], r1[12],
        r2[9], r2[10], r2[11], r2[12],
    ) || return false
    r3[4] == UInt8(1) || return false
    has_pos = r3[11] > STRUCTURAL_L1_SUB_TOL || r3[12] > STRUCTURAL_L1_SUB_TOL
    has_neg = r3[11] < -STRUCTURAL_L1_SUB_TOL || r3[12] < -STRUCTURAL_L1_SUB_TOL
    has_pos && has_neg || return false
    return (
        r3[9] == r1[9] || r3[9] == r1[10] ||
        r3[10] == r1[9] || r3[10] == r1[10]
    )
end

function _kernel_structural_prefix_screen_flag!(
    screen_flag,
    rowPtr,
    colVal,
    nzVal,
    AL,
    AU,
    m::Int32,
    n::Int32,
    pattern_code::Int32,
)
    if threadIdx().x == 1 && blockIdx().x == 1
        passed = if m == Int32(0) || n == Int32(0)
            false
        elseif pattern_code == Int32(1)
            _sls_prefix_possible_l1_split(rowPtr, colVal, nzVal, AL, AU, m)
        elseif pattern_code == Int32(2)
            _sls_prefix_possible_outer_pair(rowPtr, colVal, nzVal, AL, AU, m)
        elseif pattern_code == Int32(3)
            _sls_prefix_possible_graph(rowPtr, colVal, nzVal, AL, AU, m)
        elseif pattern_code == Int32(4)
            _sls_prefix_possible_graph(rowPtr, colVal, nzVal, AL, AU, m) ||
            _sls_prefix_possible_outer_pair(rowPtr, colVal, nzVal, AL, AU, m) ||
            _sls_prefix_possible_l1_split(rowPtr, colVal, nzVal, AL, AU, m)
        else
            true
        end
        @inbounds screen_flag[1] = passed ? UInt8(1) : UInt8(0)
    end
    return
end

function _sls_gpu_screen_flag(
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    pattern::Symbol,
)
    m, n = size(lp.A)
    pattern_code = _sls_screen_pattern_code(pattern)
    if stats.structural_screen_valid && stats.structural_screen_pattern_code == pattern_code
        return stats.structural_screen_flag
    end

    stats.structural_screen_flag .= UInt8(0)
    stats.structural_screen_counts .= Int32(0)
    stats.structural_screen_pattern_code = pattern_code
    (m == 0 || n == 0) && return stats.structural_screen_flag

    @cuda threads=1 blocks=1 _kernel_structural_prefix_screen_flag!(
        stats.structural_screen_flag,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        lp.AL,
        lp.AU,
        Int32(m),
        Int32(n),
        pattern_code,
    )
    stats.structural_screen_valid = true
    return stats.structural_screen_flag
end

function _sls_gpu_prefix_screen(
    lp::LP_info_gpu,
    pattern::Symbol,
)
    m, n = size(lp.A)
    (m == 0 || n == 0) && return false
    pattern_code = _sls_screen_pattern_code(pattern)
    screen_flag = CUDA.zeros(UInt8, 1)
    @cuda threads=1 blocks=1 _kernel_structural_prefix_screen_flag!(
        screen_flag,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        lp.AL,
        lp.AU,
        Int32(m),
        Int32(n),
        pattern_code,
    )
    return CUDA.@allowscalar Int(screen_flag[1]) != 0
end

function _sls_gpu_screen(
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    pattern::Symbol,
)
    screen_flag = _sls_gpu_screen_flag(lp, stats, pattern)
    return CUDA.@allowscalar Int(screen_flag[1]) != 0
end

function _scan_structural_l1_substitution_candidate(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if !_sls_gpu_screen(lp, stats, pparams.structural_l1_pattern)
        return (
            matched=false,
            pattern=:none,
            removed_rows=0,
            removed_cols=0,
            result=nothing,
            errors=String["gpu_screen: no structural L1 signature candidates"],
        )
    end

    A_source = isnothing(plan.new_A) ? lp.A : plan.new_A
    AL_source = plan.new_AL
    AU_source = plan.new_AU

    gpu_candidates, gpu_errors, gpu_meta =
        _sls_gpu_exact_candidates(A_source, AL_source, AU_source, pparams.structural_l1_pattern)
    isempty(gpu_candidates) && return (
        matched=false,
        pattern=:none,
        removed_rows=0,
        removed_cols=0,
        result=nothing,
        errors=vcat(String["gpu_exact: no structural L1 block layout matched"], gpu_errors),
    )

    if :l1_split_3row in gpu_candidates
        direct_result = _sls_try_gpu_direct_l1_split_result(
            plan,
            lp,
            gpu_meta;
            residual_bound_as_free_min=pparams.structural_l1_residual_bound_as_free_min,
        )
        if direct_result !== nothing
            return (
                matched=true,
                pattern=direct_result.pattern,
                removed_rows=direct_result.removed_rows,
                removed_cols=direct_result.removed_cols,
                result=direct_result,
                errors=copy(gpu_errors),
            )
        end
    end

    if :outer_pair_linked_l1 in gpu_candidates
        direct_result = _sls_try_gpu_direct_outer_pair_linked_l1_result(plan, lp, gpu_meta)
        if direct_result !== nothing
            return (
                matched=true,
                pattern=direct_result.pattern,
                removed_rows=direct_result.removed_rows,
                removed_cols=direct_result.removed_cols,
                result=direct_result,
                errors=copy(gpu_errors),
            )
        end
    end

    if :graph_l1_substitution in gpu_candidates
        direct_result = _sls_try_gpu_direct_graph_l1_result(plan, lp, gpu_meta)
        if direct_result !== nothing
            return (
                matched=true,
                pattern=direct_result.pattern,
                removed_rows=direct_result.removed_rows,
                removed_cols=direct_result.removed_cols,
                result=direct_result,
                errors=copy(gpu_errors),
            )
        end
    end

    if pparams.structural_l1_gpu_only
        return (
            matched=false,
            pattern=:none,
            removed_rows=0,
            removed_cols=0,
            result=nothing,
            errors=vcat(copy(gpu_errors), String["gpu_only: skipped structural L1 CPU fallback"]),
        )
    end

    AT_source = isnothing(plan.new_A) ? lp.AT : transpose_csr(A_source)
    A = SparseMatrixCSC(A_source)
    AT = SparseMatrixCSC(AT_source)
    c = Array(plan.new_c)
    AL = Array(AL_source)
    AU = Array(AU_source)
    l = Array(plan.new_l)
    u = Array(plan.new_u)

    errors = copy(gpu_errors)
    for pattern in gpu_candidates
        try
            result = _try_structural_l1_substitution(A, AT, c, AL, AU, l, u, pattern)
            return (
                matched=true,
                pattern=result.pattern,
                removed_rows=result.removed_rows,
                removed_cols=result.removed_cols,
                result=result,
                errors=errors,
            )
        catch err
            push!(errors, "$(pattern): $(sprint(showerror, err))")
        end
    end

    return (
        matched=false,
        pattern=:none,
        removed_rows=0,
        removed_cols=0,
        result=nothing,
        errors=errors,
    )
end

function apply_rule_structural_l1_substitution!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    summary = _scan_structural_l1_substitution_candidate(plan, lp, stats, pparams)
    if !summary.matched
        pparams.verbose && @debug (
            "structural_l1_substitution: matched=0" *
            (isempty(summary.errors) ? "" : " (" * join(summary.errors, " | ") * ")")
        )
        return nothing
    end

    result = summary.result
    A_source = isnothing(plan.new_A) ? lp.A : plan.new_A
    if hasproperty(result, :gpu_rewrite) && result.gpu_rewrite !== nothing
        A_gpu = _sls_build_gpu_rewrite_csr(A_source, result.gpu_rewrite, result.keep_row, result.keep_col)
        plan.new_A = isnothing(A_gpu) ? CuSparseMatrixCSR(result.A_new) : A_gpu
    else
        plan.new_A = CuSparseMatrixCSR(result.A_new)
    end
    plan.new_AT_leading_slack = nothing
    plan.new_AT_slack_after = nothing
    plan.keep_row_mask = _sls_mask_to_cuvector(result.keep_row)
    plan.keep_col_mask = _sls_mask_to_cuvector(result.keep_col)
    plan.new_c = _sls_to_cuvector(result.c_new)
    plan.new_AL = _sls_to_cuvector(result.AL_new)
    plan.new_AU = _sls_to_cuvector(result.AU_new)
    plan.new_l = _sls_to_cuvector(result.l_new)
    plan.new_u = _sls_to_cuvector(result.u_new)
    plan.structural_primal_recovery = result.primal_recovery
    plan.has_col_action = true
    plan.has_change = true

    pparams.verbose && @debug (
        "structural_l1_substitution: matched=1, pattern=$(result.pattern), " *
        "removed_rows=$(result.removed_rows), removed_cols=$(result.removed_cols) " *
        "(rewrite applied with structural primal recovery)"
    )

    return nothing
end
