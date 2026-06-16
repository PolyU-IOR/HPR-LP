"""
FME-simple presolve rule (fresh implementation).

This is a from-scratch reimplementation of a Fourier-Motzkin-elimination (FME)
presolve following Zhang, Ploskas & Sahinidis (2026) and the reference
`scan_fme_simple_hans.jl` candidate criterion. The previous orchestration
(single-best-column greedy with a full CPU matrix rebuild per eliminated
column) has been abandoned; only the small, correct projection primitives are
kept.

Design
- LP only, columns with zero objective coefficient (J^<= in the paper).
- A column is a "simple" FME candidate when, counting each finite constraint
  side (AL and AU) and each finite variable bound as one inequality side:
      * it touches no equality row,
      * m^+ <= S and m^- <= S            (S = fme_simple_side_limit, default 2)
      * m^+ * m^- < m^+ + m^-            (strict size reduction; this admits
        exactly the (1,0),(0,1),(1,1),(1,2),(2,1) side patterns and the
        one-sided redundant cases, and rejects (2,2)).
- Each rule call performs ONE batch pass: it greedily selects an independent
  set of candidate columns (pairwise-disjoint constraint-row sets), projects
  every selected column out, and rebuilds the working matrix exactly once.
  Chained eliminations are realized across the outer presolve iterations
  (max_iters), each of which compacts the model.

Postsolve
- Primal recovery is implemented via the `FME_COL` tape op: each eliminated
  column `x_j` (objective-free) is recovered as `max` over its lower-side
  candidates (or `min` over upper sides, else 0), which is always feasible
  because every (lower, upper) side pair was projected out as a kept row.
- Dual recovery is intentionally partial: the reduced cost of each eliminated
  column is set to 0 and the duals of the FME-deleted constraint rows are left
  at 0 (not reconstructed). Reconstructing those duals would require
  distributing each projected row's dual back to its two source rows with the
  FME combination weights; that is not implemented.
- The appended projected rows have no original-model counterpart and are mapped
  to a sentinel ("sink") deleted row in the row index map (see
  `presolve_apply_plan`), so they never pollute the recovered original duals.
"""

@inline function _fme_row_is_equality(AL_i::Float64, AU_i::Float64, tol::Float64)
    return isfinite(AL_i) && isfinite(AU_i) && abs(AU_i - AL_i) <= tol
end

@inline function _fme_live_mask(mask::CuVector{UInt8})
    return Array(mask) .!= UInt8(0)
end

function _kernel_fme_simple_candidate_count!(
    count,
    rowPtr_AT,
    colVal_AT,
    nzVal_AT,
    keep_row_mask,
    keep_col_mask,
    c,
    AL,
    AU,
    l,
    u,
    n::Int32,
    side_limit::Int32,
    zero_tol::Float64,
    bound_tol::Float64,
    zero_objective_only::Bool,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        @inbounds keep_col_mask[j] == UInt8(0) && return
        if zero_objective_only
            @inbounds abs(c[j]) > zero_tol && return
        end

        plus = isfinite(u[j]) ? Int32(1) : Int32(0)
        minus = isfinite(l[j]) ? Int32(1) : Int32(0)
        ok = true

        @inbounds start = rowPtr_AT[j]
        @inbounds stop = rowPtr_AT[j + Int32(1)] - Int32(1)
        for p in start:stop
            @inbounds row = colVal_AT[p]
            @inbounds keep_row_mask[row] == UInt8(0) && continue
            @inbounds aij = nzVal_AT[p]
            abs(aij) > zero_tol || continue

            @inbounds al = AL[row]
            @inbounds au = AU[row]
            if isfinite(al) && isfinite(au) && abs(au - al) <= bound_tol
                ok = false
                break
            end

            if isfinite(au)
                aij > 0.0 ? (plus += Int32(1)) : (minus += Int32(1))
            end
            if isfinite(al)
                -aij > 0.0 ? (plus += Int32(1)) : (minus += Int32(1))
            end
            if plus > side_limit && minus > side_limit
                ok = false
                break
            end
        end

        affected = plus + minus
        if ok &&
           affected >= Int32(1) &&
           plus <= side_limit &&
           minus <= side_limit &&
           plus * minus <= affected
            CUDA.@atomic count[1] += Int32(1)
        end
    end
    return
end

function _fme_gpu_simple_candidate_count(
    AT_source,
    plan::PresolvePlan_gpu,
    pparams::PresolveParams,
)
    # `AT_source` is CSR of A', so its row count is the number of columns in A.
    n = size(AT_source, 1)
    n == 0 && return 0

    count = CUDA.zeros(Int32, 1)
    blocks = cld(n, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_fme_simple_candidate_count!(
        count,
        AT_source.rowPtr,
        AT_source.colVal,
        AT_source.nzVal,
        plan.keep_row_mask,
        plan.keep_col_mask,
        plan.new_c,
        plan.new_AL,
        plan.new_AU,
        plan.new_l,
        plan.new_u,
        Int32(n),
        Int32(max(1, pparams.fme_simple_side_limit)),
        pparams.zero_tol,
        pparams.bound_tol,
        pparams.fme_zero_objective_only,
    )
    return Int(Array(count)[1])
end

@inline function _fme_collect_row_entries(
    AT::SparseMatrixCSC{Float64,Int32},
    row::Int,
    elim_col::Int,
    keep_col::AbstractVector{Bool},
    zero_tol::Float64,
)
    cols = Int32[]
    vals = Float64[]
    start = AT.colptr[row]
    stop = AT.colptr[row + 1] - 1
    if start <= stop
        for p in start:stop
            col = Int(AT.rowval[p])
            col == elim_col && continue
            keep_col[col] || continue
            val = AT.nzval[p]
            abs(val) > zero_tol || continue
            push!(cols, Int32(col))
            push!(vals, val)
        end
    end
    return cols, vals
end

@inline function _fme_make_side(
    source_id::Int32,
    rhs::Float64,
    aij::Float64,
    cols::Vector{Int32},
    vals::Vector{Float64},
)
    coeffs = Vector{Float64}(undef, length(vals))
    inv_aij = 1.0 / aij
    for t in eachindex(vals)
        coeffs[t] = -vals[t] * inv_aij
    end
    return (
        source_row=source_id,
        rhs=rhs * inv_aij,
        cols=cols,
        coeffs=coeffs,
    )
end

"""
Project one lower side (l(x') <= x_j) against one upper side (x_j <= u(x'))
to obtain a new inequality `coeffs . x' <= rhs` that no longer contains x_j.
"""
function _fme_project_pair(lower_side, upper_side, zero_tol::Float64)
    cols = Int32[]
    vals = Float64[]

    p = 1
    q = 1
    cols_l = lower_side.cols
    cols_u = upper_side.cols
    coeffs_l = lower_side.coeffs
    coeffs_u = upper_side.coeffs

    while p <= length(cols_l) || q <= length(cols_u)
        if q > length(cols_u) || (p <= length(cols_l) && cols_l[p] < cols_u[q])
            coeff = coeffs_l[p]
            col = cols_l[p]
            p += 1
        elseif p > length(cols_l) || cols_u[q] < cols_l[p]
            coeff = -coeffs_u[q]
            col = cols_u[q]
            q += 1
        else
            coeff = coeffs_l[p] - coeffs_u[q]
            col = cols_l[p]
            p += 1
            q += 1
        end

        if abs(coeff) > zero_tol
            push!(cols, col)
            push!(vals, coeff)
        end
    end

    return (cols=cols, vals=vals, rhs=upper_side.rhs - lower_side.rhs)
end

function _fme_normalize_projected_row(projected_row, zero_tol::Float64)
    isempty(projected_row.cols) && return nothing
    scale = maximum(abs.(projected_row.vals))
    scale <= zero_tol && return nothing
    norm_vals = round.(projected_row.vals ./ scale; digits=10)
    norm_rhs = projected_row.rhs / scale
    return (
        cols = Tuple(Int.(projected_row.cols)),
        vals = Tuple(norm_vals),
        rhs = norm_rhs,
    )
end

"""
Deduplicate projected rows that share an identical (normalized) coefficient
pattern, keeping the tightest right-hand side.
"""
function _fme_dedup_projected_rows(projected_rows, zero_tol::Float64)
    best_by_key = Dict{Tuple{Tuple{Vararg{Int}},Tuple{Vararg{Float64}}}, NamedTuple}()
    for row in projected_rows
        norm = _fme_normalize_projected_row(row, zero_tol)
        isnothing(norm) && continue
        key = (norm.cols, norm.vals)
        if haskey(best_by_key, key)
            existing = best_by_key[key]
            if norm.rhs < existing.rhs - zero_tol
                best_by_key[key] = (rhs = norm.rhs, row = row)
            end
        else
            best_by_key[key] = (rhs = norm.rhs, row = row)
        end
    end
    return [entry.row for entry in values(best_by_key)]
end

"""
Single-column profile: count positive/negative inequality sides (constraints
and, optionally, finite variable bounds), collect the involved constraint rows,
and report whether the column touches an equality row.
"""
function _fme_simple_profile(
    A::SparseMatrixCSC{Float64,Int32},
    keep_row::AbstractVector{Bool},
    AL::AbstractVector{Float64},
    AU::AbstractVector{Float64},
    l::AbstractVector{Float64},
    u::AbstractVector{Float64},
    row_is_eq::AbstractVector{Bool},
    pparams::PresolveParams,
    j::Int,
    side_limit::Int,
)
    ztol = pparams.zero_tol
    plus = 0
    minus = 0
    involved = Int[]
    for p in A.colptr[j]:(A.colptr[j + 1] - 1)
        row = Int(A.rowval[p])
        keep_row[row] || continue
        aij = A.nzval[p]
        abs(aij) > ztol || continue
        row_is_eq[row] && return (ok=false, plus=0, minus=0, affected=0, involved=Int[])
        push!(involved, row)
        if isfinite(AU[row])
            aij > 0 ? (plus += 1) : (minus += 1)
        end
        if isfinite(AL[row])
            (-aij) > 0 ? (plus += 1) : (minus += 1)
        end
        # Early out for clearly non-simple columns to keep the scan cheap.
        if plus > side_limit && minus > side_limit
            return (ok=false, plus=plus, minus=minus, affected=plus + minus, involved=involved)
        end
    end

    # Variable bounds are always counted as inequality sides (matching the
    # reference `scan_fme_simple`): l <= x_j and x_j <= u.
    isfinite(u[j]) && (plus += 1)
    isfinite(l[j]) && (minus += 1)

    affected = plus + minus
    return (ok=true, plus=plus, minus=minus, affected=affected, involved=involved)
end

@inline function _fme_is_simple_candidate(profile, side_limit::Int)
    profile.ok || return false
    profile.affected >= 1 || return false
    profile.plus <= side_limit || return false
    profile.minus <= side_limit || return false
    # No-growth in the number of inequality sides (matches the reference
    # `scan_fme_simple` candidate set; for side_limit<=2 this is always true,
    # i.e. it admits the (1,1),(1,2),(2,1),(2,2) and one-sided patterns).
    return profile.plus * profile.minus <= profile.affected
end

"""
Bound-propagation redundancy / infeasibility test for a projected inequality
`sum coeff_k x_k <= rhs` (paper Sec. 3.3). Returns `:redundant` if current
variable bounds already imply the row, `:infeasible` if they violate it, and
`:keep` otherwise.
"""
@inline function _fme_row_bound_status(
    cols::Vector{Int32},
    vals::Vector{Float64},
    rhs::Float64,
    l::AbstractVector{Float64},
    u::AbstractVector{Float64},
    tol::Float64,
)
    max_lhs = 0.0
    min_lhs = 0.0
    for t in eachindex(cols)
        col = Int(cols[t])
        a = vals[t]
        if a > 0
            max_lhs += isfinite(u[col]) ? a * u[col] : Inf
            min_lhs += isfinite(l[col]) ? a * l[col] : -Inf
        else
            max_lhs += isfinite(l[col]) ? a * l[col] : Inf
            min_lhs += isfinite(u[col]) ? a * u[col] : -Inf
        end
    end
    if max_lhs <= rhs + tol
        return :redundant
    elseif min_lhs > rhs + tol
        return :infeasible
    end
    return :keep
end

"""
Build the projected (FME) rows for eliminating column `elim_col`. Returns the
list of involved constraint rows and the deduplicated projected rows
(each a NamedTuple with `cols`, `vals`, `rhs` for `cols . x <= rhs`).

Returns `nothing` for `projected_rows` if the projection proves the model
infeasible (a constant inequality with negative slack).
"""
function _fme_project_column(
    A::SparseMatrixCSC{Float64,Int32},
    AT::SparseMatrixCSC{Float64,Int32},
    keep_row::AbstractVector{Bool},
    keep_col::AbstractVector{Bool},
    AL::AbstractVector{Float64},
    AU::AbstractVector{Float64},
    l::AbstractVector{Float64},
    u::AbstractVector{Float64},
    pparams::PresolveParams,
    elim_col::Int,
)
    ztol = pparams.zero_tol
    btol = pparams.bound_tol
    lower_sides = NamedTuple[]
    upper_sides = NamedTuple[]
    involved_rows = Int[]

    for p in A.colptr[elim_col]:(A.colptr[elim_col + 1] - 1)
        row = Int(A.rowval[p])
        keep_row[row] || continue
        aij = A.nzval[p]
        abs(aij) > ztol || continue
        push!(involved_rows, row)
        row_cols, row_vals = _fme_collect_row_entries(AT, row, elim_col, keep_col, ztol)
        if aij > 0
            isfinite(AL[row]) && push!(lower_sides, _fme_make_side(Int32(row), AL[row], aij, row_cols, row_vals))
            isfinite(AU[row]) && push!(upper_sides, _fme_make_side(Int32(row), AU[row], aij, row_cols, row_vals))
        else
            isfinite(AL[row]) && push!(upper_sides, _fme_make_side(Int32(row), AL[row], aij, row_cols, row_vals))
            isfinite(AU[row]) && push!(lower_sides, _fme_make_side(Int32(row), AU[row], aij, row_cols, row_vals))
        end
    end

    # Variable bounds always participate as inequality sides.
    isfinite(l[elim_col]) && push!(lower_sides, _fme_make_side(Int32(-1), l[elim_col], 1.0, Int32[], Float64[]))
    isfinite(u[elim_col]) && push!(upper_sides, _fme_make_side(Int32(-2), u[elim_col], 1.0, Int32[], Float64[]))

    unique!(involved_rows)

    # One-sided variable (m^+ m^- == 0): every involved row is redundant, and a
    # column with no live constraint rows is simply dropped (empty column).
    if isempty(lower_sides) || isempty(upper_sides)
        return (involved_rows=involved_rows, projected_rows=NamedTuple[], infeasible=false, lower_sides=lower_sides, upper_sides=upper_sides)
    end

    projected = NamedTuple[]
    for ls in lower_sides
        for us in upper_sides
            ls.source_row == us.source_row && continue
            pr = _fme_project_pair(ls, us, ztol)
            if isempty(pr.cols)
                # Constant inequality 0 <= rhs; negative slack proves infeasibility.
                pr.rhs < -btol && return (involved_rows=involved_rows, projected_rows=nothing, infeasible=true, lower_sides=lower_sides, upper_sides=upper_sides)
                continue
            end
            # Drop rows already implied by variable bounds; flag infeasibility.
            status = _fme_row_bound_status(pr.cols, pr.vals, pr.rhs, l, u, btol)
            status == :infeasible && return (involved_rows=involved_rows, projected_rows=nothing, infeasible=true, lower_sides=lower_sides, upper_sides=upper_sides)
            status == :redundant && continue
            push!(projected, pr)
        end
    end

    projected = _fme_dedup_projected_rows(projected, ztol)
    return (involved_rows=involved_rows, projected_rows=projected, infeasible=false, lower_sides=lower_sides, upper_sides=upper_sides)
end

"""
One batch FME-simple pass. Selects an independent set of simple candidate
columns and projects them all out with a single matrix rebuild.
"""
function apply_rule_fme_projection!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    _stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    A_source = isnothing(plan.new_A) ? lp.A : plan.new_A
    AT_source = isnothing(plan.new_A) ? lp.AT : transpose_csr(A_source)
    candidate_count_gpu = _fme_gpu_simple_candidate_count(AT_source, plan, pparams)
    if candidate_count_gpu == 0
        pparams.verbose_fme && println("fme_simple: gpu_screen candidates=0, skipped CPU scan")
        return nothing
    end

    A = SparseMatrixCSC(A_source)
    AT = SparseMatrixCSC(AT_source)
    c = Array(plan.new_c)
    AL = Array(plan.new_AL)
    AU = Array(plan.new_AU)
    l = Array(plan.new_l)
    u = Array(plan.new_u)
    keep_row = _fme_live_mask(plan.keep_row_mask)
    keep_col = _fme_live_mask(plan.keep_col_mask)

    m, n = size(A)
    side_limit = max(1, pparams.fme_simple_side_limit)
    ztol = pparams.zero_tol

    row_is_eq = Vector{Bool}(undef, m)
    @inbounds for i in 1:m
        row_is_eq[i] = keep_row[i] && _fme_row_is_equality(AL[i], AU[i], pparams.bound_tol)
    end

    scanned = 0
    candidates = 0
    rej_conflict = 0
    removed_empty = 0
    rej_grow = 0
    used_row = falses(m)
    elim_cols = Int[]
    elim_projected = Vector{Vector{NamedTuple}}()
    elim_involved = Vector{Vector{Int}}()
    elim_sides = Vector{NamedTuple}()
    record_tape = pparams.record_postsolve_tape || pparams.record_postsolve_tape_cpu
    removed_rows_total = 0
    added_rows_total = 0
    removed_nnz_total = 0
    added_nnz_total = 0
    infeasible = false

    @inbounds for j in 1:n
        keep_col[j] || continue
        if pparams.fme_zero_objective_only && abs(c[j]) > ztol
            continue
        end
        scanned += 1

        profile = _fme_simple_profile(A, keep_row, AL, AU, l, u, row_is_eq, pparams, j, side_limit)
        _fme_is_simple_candidate(profile, side_limit) || continue
        candidates += 1

        # Independent-set constraint: skip if any involved row is already taken.
        conflict = false
        for row in profile.involved
            if used_row[row]
                conflict = true
                break
            end
        end
        if conflict
            rej_conflict += 1
            continue
        end

        result = _fme_project_column(A, AT, keep_row, keep_col, AL, AU, l, u, pparams, j)
        if result.infeasible
            infeasible = true
            break
        end
        projected = result.projected_rows
        involved = result.involved_rows

        # Empty column (no live constraint rows): pure column removal.
        if isempty(involved)
            removed_empty += 1
            push!(elim_cols, j)
            push!(elim_projected, NamedTuple[])
            push!(elim_involved, Int[])
            record_tape && push!(elim_sides, (lower=result.lower_sides, upper=result.upper_sides))
            continue
        end

        # For "simple" columns (sides bounded by side_limit) the projected row
        # count is bounded, so we apply even when a few rows are added; the
        # column is always removed and dominated/duplicate rows are deduped.
        length(projected) > length(involved) && (rej_grow += 1)

        for row in involved
            used_row[row] = true
        end
        push!(elim_cols, j)
        push!(elim_projected, projected)
        push!(elim_involved, involved)
        record_tape && push!(elim_sides, (lower=result.lower_sides, upper=result.upper_sides))
        removed_rows_total += length(involved)
        added_rows_total += length(projected)
        for row in involved
            removed_nnz_total += AT.colptr[row + 1] - AT.colptr[row]
        end
        for pr in projected
            added_nnz_total += length(pr.cols)
        end
    end

    if infeasible
        plan.has_infeasible = true
        plan.has_change = true
        plan.status_message = "fme_simple detected infeasibility during projection"
        pparams.verbose_fme && println("fme_simple: infeasibility detected (projected constant inequality)")
        return nothing
    end

    if isempty(elim_cols)
        pparams.verbose_fme && println(
            "fme_simple: scanned=$scanned, candidates=$candidates, eliminated=0 " *
            "(rej_conflict=$rej_conflict, rej_grow=$rej_grow)"
        )
        return nothing
    end

    # Rebuild the matrix once: drop involved rows + eliminated columns, append
    # the projected rows.
    row_delete = falses(m)
    for involved in elim_involved
        for row in involved
            row_delete[row] = true
        end
    end

    n_new_rows = added_rows_total
    old_nnz = nnz(A)
    I, J, V = findnz(A)
    new_I = Int32[]
    new_J = Int32[]
    new_V = Float64[]
    sizehint!(new_I, length(I))
    sizehint!(new_J, length(J))
    sizehint!(new_V, length(V))

    @inbounds for idx in eachindex(I)
        row = Int(I[idx])
        row_delete[row] && continue
        push!(new_I, I[idx])
        push!(new_J, J[idx])
        push!(new_V, V[idx])
    end

    AL_new = vcat(AL, fill(-Inf, n_new_rows))
    AU_new = vcat(AU, fill(Inf, n_new_rows))
    keep_row_new = vcat(keep_row, fill(true, n_new_rows))

    next_row = m
    for projected in elim_projected
        for pr in projected
            next_row += 1
            for t in eachindex(pr.cols)
                push!(new_I, Int32(next_row))
                push!(new_J, pr.cols[t])
                push!(new_V, pr.vals[t])
            end
            AU_new[next_row] = pr.rhs
        end
    end

    for j in elim_cols
        keep_col[j] = false
    end
    for row in 1:m
        row_delete[row] && (keep_row_new[row] = false)
    end

    total_rows = m + n_new_rows
    A_built = sparse(new_I, new_J, new_V, total_rows, n)

    plan.new_A = CuSparseMatrixCSR(A_built)
    plan.new_AT_leading_slack = nothing
    plan.new_AT_slack_after = nothing
    plan.keep_row_mask = CuVector(UInt8.(keep_row_new))
    copyto!(plan.keep_col_mask, CuVector(UInt8.(keep_col)))
    plan.new_AL = CuVector(AL_new)
    plan.new_AU = CuVector(AU_new)
    plan.has_col_action = true
    plan.has_row_action = true
    plan.has_change = true

    # Record postsolve tape (FME_COL primal recovery). Recovery of each
    # eliminated column reads the current-model column/row indices; the merge
    # step globalizes them to original coordinates.
    if record_tape && !isempty(elim_cols)
        fme_tape = PostsolveTape()
        for c in eachindex(elim_cols)
            sides = NamedTuple[]
            for ls in elim_sides[c].lower
                push!(sides, (orient=1.0, rhs=ls.rhs, cols=ls.cols, coeffs=ls.coeffs))
            end
            for us in elim_sides[c].upper
                push!(sides, (orient=-1.0, rhs=us.rhs, cols=us.cols, coeffs=us.coeffs))
            end
            append_fme_col_record!(fme_tape, elim_cols[c], elim_involved[c], sides)
        end
        if pparams.record_postsolve_tape
            append_postsolve_tape!(plan.tape_gpu, PostsolveTape_gpu(fme_tape))
        end
        if pparams.record_postsolve_tape_cpu
            append_postsolve_tape!(plan.tape, fme_tape)
        end
    end

    if pparams.verbose_fme
        net_rows = removed_rows_total - added_rows_total
        println(
            "fme_simple: scanned=$scanned, candidates=$candidates, " *
            "eliminated_cols=$(length(elim_cols)), empty_cols=$removed_empty, " *
            "removed_rows=$removed_rows_total, added_rows=$added_rows_total, " *
            "net_rows=$net_rows, removed_nnz=$removed_nnz_total, added_nnz=$added_nnz_total, " *
            "net_nnz=$(added_nnz_total - removed_nnz_total), old_nnz=$old_nnz, " *
            "rej_conflict=$rej_conflict, rej_grow=$rej_grow"
        )
    end

    return nothing
end
