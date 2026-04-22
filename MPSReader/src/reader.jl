struct LPFileData{Tv<:AbstractFloat,Ti<:Integer}
    nrow::Int
    ncol::Int
    arows::Vector{Ti}
    acols::Vector{Ti}
    avals::Vector{Tv}
    c::Vector{Tv}
    lcon::Vector{Tv}
    ucon::Vector{Tv}
    lvar::Vector{Tv}
    uvar::Vector{Tv}
    obj_constant::Tv
    rownames::Union{Nothing,Vector{String}}
    colnames::Union{Nothing,Vector{String}}
end

function sparse_matrix(data::LPFileData)
    return sparse(data.arows, data.acols, data.avals, data.nrow, data.ncol)
end

@enum RowType::UInt8 begin
    RTYPE_OBJECTIVE
    RTYPE_EQUALTO
    RTYPE_LESSTHAN
    RTYPE_GREATERTHAN
end

@enum VariableType::UInt8 begin
    VTYPE_MARKED
    VTYPE_CONTINUOUS
    VTYPE_BINARY
    VTYPE_INTEGER
end

@enum SectionType::UInt8 begin
    SECTION_NONE
    SECTION_NAME
    SECTION_OBJSENSE
    SECTION_ROWS
    SECTION_COLUMNS
    SECTION_RHS
    SECTION_BOUNDS
    SECTION_RANGES
    SECTION_QUADOBJ
    SECTION_QMATRIX
    SECTION_OBJECT_BOUND
    SECTION_ENDATA
end

struct MPSCard
    nline::Int
    iscomment::Bool
    isheader::Bool
    nfields::Int
    f1::SubString{String}
    f2::SubString{String}
    f3::SubString{String}
    f4::SubString{String}
    f5::SubString{String}
    f6::SubString{String}
end

mutable struct NameIndexMap
    size::Int
    num_buckets::Int
    names::Vector{String}
    indices::Vector{Int}
    next::Vector{Int}
    buckets::Vector{Int}
end

mutable struct MPSData
    nvar::Int
    ncon::Int
    objsense::Symbol
    c0::Float64
    c::Vector{Float64}
    arows::Vector{Int}
    acols::Vector{Int}
    avals::Vector{Float64}
    lcon::Vector{Float64}
    ucon::Vector{Float64}
    lvar::Vector{Float64}
    uvar::Vector{Float64}
    name::Union{Nothing,String}
    objname::Union{Nothing,String}
    rhsname::Union{Nothing,String}
    bndname::Union{Nothing,String}
    rngname::Union{Nothing,String}
    varnames::Union{Nothing,Vector{String}}
    connames::Union{Nothing,Vector{String}}
    varindices::NameIndexMap
    conindices::NameIndexMap
    packed_varindices::Dict{UInt64,Int}
    packed_conindices::Dict{UInt64,Int}
    vartypes::Vector{VariableType}
    contypes::Vector{RowType}
end

struct CapacityHints
    est_vars::Int
    est_cons::Int
    est_nnz::Int
    est_var_map::Int
    est_con_map::Int
end

mutable struct ParserState
    current_section::SectionType
    integer_section::Bool
    endata_read::Bool
    last_col_name::SubString{String}
    last_col_key::UInt64
    last_col::Int
    use_packed_names::Bool
end

const _EMPTY_FIELD = SubString("", 1, 0)
const _INITIAL_CAPACITY = 8192
const _INITIAL_NNZ_CAPACITY = 100_000

const _READALL_THRESHOLD_BYTES = 2_000_000_000
const _STREAM_CHUNK_BYTES = 8 * 1024 * 1024
const _GZIP_SIZE_ESTIMATE_FACTOR = 12
const _GZIP_BYTES_PER_NNZ_ESTIMATE = 80
const _GZIP_BYTES_PER_DIM_ESTIMATE = 192
const _MAX_GZIP_DIM_HINT = 32_000_000
const _MAX_GZIP_NNZ_HINT = 120_000_000
const _MAX_GZIP_MAP_HINT = 8_000_000
const _DEFAULT_CAPACITY_HINTS = CapacityHints(_INITIAL_CAPACITY, _INITIAL_CAPACITY, _INITIAL_NNZ_CAPACITY, _INITIAL_CAPACITY, _INITIAL_CAPACITY)

_isspacebyte(c::UInt8) = c == 0x20 || c == 0x09 || c == 0x0d || c == 0x0a
_isdigitbyte(c::UInt8) = 0x30 <= c <= 0x39

@inline function _pack_fixed_name(name::AbstractString)
    key = UInt64(0)
    shift = 0
    for byte in codeunits(name)
        key |= UInt64(byte) << shift
        shift += 8
    end
    return key
end

@inline function _pack_fixed_name(bytes, start::Int, stop::Int)
    key = UInt64(0)
    shift = 0
    @inbounds for i in start:stop
        key |= UInt64(bytes[i]) << shift
        shift += 8
    end
    return key
end

function _next_power_of_two(n::Int)
    p = 1
    while p < n
        p <<= 1
    end
    return p
end

function NameIndexMap(capacity::Int)
    map_capacity = max(16, capacity)
    bucket_count = max(4096, _next_power_of_two(map_capacity))
    return NameIndexMap(
        0,
        bucket_count,
        Vector{String}(undef, map_capacity),
        Vector{Int}(undef, map_capacity),
        fill(0, map_capacity),
        fill(0, bucket_count),
    )
end

@inline _packedmap_get(map::Dict{UInt64,Int}, key::UInt64, default::Int) = get(map, key, default)
@inline _packedmap_haskey(map::Dict{UInt64,Int}, key::UInt64) = haskey(map, key)

function _gzip_isize_hint(path::AbstractString)
    file_size = try filesize(path) catch; 0 end
    file_size < 4 && return 0
    open(path, "r") do io
        seek(io, file_size - 4)
        footer = read(io, 4)
        length(footer) == 4 || return 0
        return Int(
            UInt32(footer[1]) |
            (UInt32(footer[2]) << 8) |
            (UInt32(footer[3]) << 16) |
            (UInt32(footer[4]) << 24)
        )
    end
end

@inline function _namemap_hash(name::AbstractString, num_buckets::Int)
    hash_value = UInt(5381)
    for byte in codeunits(name)
        hash_value = ((hash_value << 5) + hash_value) + UInt(byte)
    end
    return Int(hash_value & UInt(num_buckets - 1)) + 1
end

@inline function _namemap_find(map::NameIndexMap, name::AbstractString)
    map.size == 0 && return 0
    bucket = _namemap_hash(name, map.num_buckets)
    idx = map.buckets[bucket]
    while idx != 0
        if map.names[idx] == name
            return idx
        end
        idx = map.next[idx]
    end
    return 0
end

@inline function _namemap_get(map::NameIndexMap, name::AbstractString, default::Int)
    idx = _namemap_find(map, name)
    return idx == 0 ? default : map.indices[idx]
end

_namemap_haskey(map::NameIndexMap, name::AbstractString) = _namemap_find(map, name) != 0

function _namemap_rehash!(map::NameIndexMap, new_bucket_count::Int)
    map.num_buckets = new_bucket_count
    map.buckets = fill(0, new_bucket_count)
    for idx in 1:map.size
        bucket = _namemap_hash(map.names[idx], map.num_buckets)
        map.next[idx] = map.buckets[bucket]
        map.buckets[bucket] = idx
    end
    return map
end

function _namemap_ensure_capacity!(map::NameIndexMap)
    if map.size >= length(map.names)
        new_capacity = max(16, 2 * length(map.names))
        resize!(map.names, new_capacity)
        resize!(map.indices, new_capacity)
        resize!(map.next, new_capacity)
        fill!(view(map.next, map.size + 1:new_capacity), 0)
    end
    if map.size * 4 >= map.num_buckets * 3
        _namemap_rehash!(map, 2 * map.num_buckets)
    end
    return map
end

function _namemap_set!(map::NameIndexMap, name::String, index::Int)
    idx = _namemap_find(map, name)
    if idx != 0
        map.indices[idx] = index
        return index
    end

    _namemap_ensure_capacity!(map)
    map.size += 1
    idx = map.size
    bucket = _namemap_hash(name, map.num_buckets)
    map.names[idx] = name
    map.indices[idx] = index
    map.next[idx] = map.buckets[bucket]
    map.buckets[bucket] = idx
    return index
end

function _trim_range(line::AbstractString, start::Int, stop::Int)
    n = ncodeunits(line)
    start = max(start, 1)
    stop = min(stop, n)
    start > stop && return _EMPTY_FIELD
    bytes = codeunits(line)
    while start <= stop && _isspacebyte(bytes[start])
        start += 1
    end
    while stop >= start && _isspacebyte(bytes[stop])
        stop -= 1
    end
    return start <= stop ? SubString(line, start, stop) : _EMPTY_FIELD
end

@inline function _trim_range_fast(line::AbstractString, bytes, start::Int, stop::Int, len::Int)
    start > len && return _EMPTY_FIELD
    stop = stop < len ? stop : len
    while start <= stop && _isspacebyte(bytes[start])
        start += 1
    end
    while stop >= start && _isspacebyte(bytes[stop])
        stop -= 1
    end
    return start <= stop ? SubString(line, start, stop) : _EMPTY_FIELD
end

@inline function _trim_span_fast(bytes, start::Int, stop::Int, len::Int)
    start > len && return 1, 0
    stop = stop < len ? stop : len
    while start <= stop && _isspacebyte(bytes[start])
        start += 1
    end
    while stop >= start && _isspacebyte(bytes[stop])
        stop -= 1
    end
    return start, stop
end

@inline function _count_spans5(s1::Int, e1::Int, s2::Int, e2::Int, s3::Int, e3::Int, s4::Int, e4::Int, s5::Int, e5::Int)
    s1 > e1 && return 0
    s2 > e2 && return 1
    s3 > e3 && return 2
    s4 > e4 && return 3
    s5 > e5 && return 4
    return 5
end

@inline function _count_spans4(s1::Int, e1::Int, s2::Int, e2::Int, s3::Int, e3::Int, s4::Int, e4::Int)
    s1 > e1 && return 0
    s2 > e2 && return 1
    s3 > e3 && return 2
    s4 > e4 && return 3
    return 4
end

@inline function _span_equals(bytes, start::Int, stop::Int, literal::AbstractString)
    start > stop && return isempty(literal)
    len = stop - start + 1
    litbytes = codeunits(literal)
    len == length(litbytes) || return false
    @inbounds for offset in 1:len
        bytes[start + offset - 1] == litbytes[offset] || return false
    end
    return true
end

@inline _span_string(line::AbstractString, start::Int, stop::Int) = String(SubString(line, start, stop))

@inline function _count_fields5(f1, f2, f3, f4, f5)
    isempty(f1) && return 0
    isempty(f2) && return 1
    isempty(f3) && return 2
    isempty(f4) && return 3
    isempty(f5) && return 4
    return 5
end

@inline function _count_fields4(f1, f2, f3, f4)
    isempty(f1) && return 0
    isempty(f2) && return 1
    isempty(f3) && return 2
    isempty(f4) && return 3
    return 4
end

function _parse_section_header(line::AbstractString, nline::Int)
    fields = _parse_free_fields(line)
    f1, f2, f3, f4, f5, f6, nfields = fields
    if f1 == "OBJECT" && f2 == "BOUND"
        f1 = SubString("OBJECT BOUND", 1, lastindex("OBJECT BOUND"))
        f2 = _EMPTY_FIELD
        nfields = 1
    end
    return MPSCard(nline, false, true, nfields, f1, f2, f3, f4, f5, f6)
end

function _parse_fixed_fields(line::AbstractString, nline::Int)
    len = ncodeunits(line)
    if len == 0
        return MPSCard(nline, true, false, 0, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD)
    end
    firstb = codeunits(line)[1]
    if firstb == UInt8('*') || firstb == UInt8('&')
        return MPSCard(nline, true, false, 0, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD)
    end
    if !_isspacebyte(firstb)
        return _parse_section_header(line, nline)
    end

    f1 = len >= 3 ? _trim_range(line, 2, 3) : _EMPTY_FIELD
    f2 = len >= 5 ? _trim_range(line, 5, 12) : _EMPTY_FIELD
    f3 = len >= 15 ? _trim_range(line, 15, 22) : _EMPTY_FIELD
    f4 = len >= 25 ? _trim_range(line, 25, 36) : _EMPTY_FIELD
    f5 = len >= 40 ? _trim_range(line, 40, 47) : _EMPTY_FIELD
    f6 = len >= 50 ? _trim_range(line, 50, 61) : _EMPTY_FIELD

    if isempty(f1)
        f1, f2, f3, f4, f5, f6 = f2, f3, f4, f5, f6, _EMPTY_FIELD
    end

    nfields = 0
    for field in (f1, f2, f3, f4, f5, f6)
        isempty(field) && break
        nfields += 1
    end

    return MPSCard(nline, false, false, nfields, f1, f2, f3, f4, f5, f6)
end

function _parse_free_fields(line::AbstractString)
    len = ncodeunits(line)
    bytes = codeunits(line)
    f1 = _EMPTY_FIELD
    f2 = _EMPTY_FIELD
    f3 = _EMPTY_FIELD
    f4 = _EMPTY_FIELD
    f5 = _EMPTY_FIELD
    f6 = _EMPTY_FIELD
    field_idx = 0
    i = 1
    while i <= len && field_idx < 6
        while i <= len && _isspacebyte(bytes[i])
            i += 1
        end
        i > len && break
        start = i
        while i <= len && !_isspacebyte(bytes[i])
            i += 1
        end
        field_idx += 1
        field = SubString(line, start, i - 1)
        if field_idx == 1
            f1 = field
        elseif field_idx == 2
            f2 = field
        elseif field_idx == 3
            f3 = field
        elseif field_idx == 4
            f4 = field
        elseif field_idx == 5
            f5 = field
        else
            f6 = field
        end
    end
    return (f1, f2, f3, f4, f5, f6, field_idx)
end

function _parse_free_card(line::AbstractString, nline::Int)
    len = ncodeunits(line)
    if len == 0
        return MPSCard(nline, true, false, 0, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD)
    end
    firstb = codeunits(line)[1]
    if firstb == UInt8('*') || firstb == UInt8('&')
        return MPSCard(nline, true, false, 0, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD)
    end
    if !_isspacebyte(firstb)
        return _parse_section_header(line, nline)
    end

    f1, f2, f3, f4, f5, f6, nfields = _parse_free_fields(line)
    return MPSCard(nline, false, false, nfields, f1, f2, f3, f4, f5, f6)
end

function _estimate_capacities(path::String)
    file_size = try filesize(path) catch; 0 end
    lower_path = lowercase(path)

    if endswith(lower_path, ".mps.gz") && file_size > 0
        footer_hint = _gzip_isize_hint(path)
        scaled_hint = file_size * _GZIP_SIZE_ESTIMATE_FACTOR
        logical_size = max(file_size, max(footer_hint, scaled_hint))
        est_nnz = clamp(Int(logical_size ÷ _GZIP_BYTES_PER_NNZ_ESTIMATE), _INITIAL_NNZ_CAPACITY, _MAX_GZIP_NNZ_HINT)
        est_dim = clamp(Int(logical_size ÷ _GZIP_BYTES_PER_DIM_ESTIMATE), _INITIAL_CAPACITY, _MAX_GZIP_DIM_HINT)
        est_map = clamp(est_dim, _INITIAL_CAPACITY, _MAX_GZIP_MAP_HINT)
        return CapacityHints(est_dim, est_dim, est_nnz, est_map, est_map)
    elseif file_size > 0
        est_nnz = clamp(Int(file_size ÷ 50), _INITIAL_NNZ_CAPACITY, 10_000_000)
        est_vars = clamp(est_nnz ÷ 10, _INITIAL_CAPACITY, 1_000_000)
        est_cons = clamp(est_vars ÷ 2, _INITIAL_CAPACITY, 1_000_000)
        return CapacityHints(est_vars, est_cons, est_nnz, est_vars, est_cons)
    end
    return _DEFAULT_CAPACITY_HINTS
end

function _create_data(; hints::CapacityHints = _DEFAULT_CAPACITY_HINTS, keep_names::Bool = false, use_packed_names::Bool = false)
    est_vars = hints.est_vars
    est_cons = hints.est_cons
    est_nnz = hints.est_nnz
    c = Float64[]
    lvar = Float64[]
    uvar = Float64[]
    varnames = keep_names ? String[] : nothing
    vartypes = VariableType[]
    lcon = Float64[]
    ucon = Float64[]
    connames = keep_names ? String[] : nothing
    contypes = RowType[]
    arows = Int[]
    acols = Int[]
    avals = Float64[]
    sizehint!(c, est_vars)
    sizehint!(lvar, est_vars)
    sizehint!(uvar, est_vars)
    varnames !== nothing && sizehint!(varnames, est_vars)
    sizehint!(vartypes, est_vars)
    sizehint!(lcon, est_cons)
    sizehint!(ucon, est_cons)
    connames !== nothing && sizehint!(connames, est_cons)
    sizehint!(contypes, est_cons)
    sizehint!(arows, est_nnz)
    sizehint!(acols, est_nnz)
    sizehint!(avals, est_nnz)
    varindices = NameIndexMap(use_packed_names ? _INITIAL_CAPACITY : hints.est_var_map)
    conindices = NameIndexMap(use_packed_names ? _INITIAL_CAPACITY : hints.est_con_map)
    packed_varindices = Dict{UInt64,Int}()
    packed_conindices = Dict{UInt64,Int}()
    if use_packed_names
        sizehint!(packed_varindices, hints.est_var_map)
        sizehint!(packed_conindices, hints.est_con_map)
    end
    return MPSData(
        0,
        0,
        :notset,
        0.0,
        c,
        arows,
        acols,
        avals,
        lcon,
        ucon,
        lvar,
        uvar,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        varnames,
        connames,
        varindices,
        conindices,
        packed_varindices,
        packed_conindices,
        vartypes,
        contypes,
    )
end

function _section_type(header::AbstractString)
    header == "ENDATA" && return SECTION_ENDATA
    header == "NAME" && return SECTION_NAME
    header == "OBJSENSE" && return SECTION_OBJSENSE
    header == "ROWS" && return SECTION_ROWS
    header == "COLUMNS" && return SECTION_COLUMNS
    header == "RHS" && return SECTION_RHS
    header == "BOUNDS" && return SECTION_BOUNDS
    header == "RANGES" && return SECTION_RANGES
    header == "QUADOBJ" && return SECTION_QUADOBJ
    header == "QMATRIX" && return SECTION_QMATRIX
    header == "OBJECT BOUND" && return SECTION_OBJECT_BOUND
    return SECTION_NONE
end

function _row_type(type_str::AbstractString)
    type_str == "E" && return RTYPE_EQUALTO
    type_str == "L" && return RTYPE_LESSTHAN
    type_str == "G" && return RTYPE_GREATERTHAN
    return RTYPE_OBJECTIVE
end

@inline function _parse_float(str::AbstractString)
    bytes = codeunits(str)
    n = ncodeunits(str)
    i = 1
    negative = false

    if i <= n
        b = bytes[i]
        if b == 0x2d
            negative = true
            i += 1
        elseif b == 0x2b
            i += 1
        end
    end

    value = 0.0
    frac_scale = 0.1
    ndigits = 0
    seen_digit = false

    while i <= n
        b = bytes[i]
        if _isdigitbyte(b)
            value = muladd(value, 10.0, Float64(b - 0x30))
            ndigits += 1
            seen_digit = true
            if ndigits > 18
                return parse(Float64, str)
            end
            i += 1
        else
            break
        end
    end

    if i <= n && bytes[i] == 0x2e
        i += 1
        while i <= n
            b = bytes[i]
            if _isdigitbyte(b)
                value += Float64(b - 0x30) * frac_scale
                frac_scale *= 0.1
                ndigits += 1
                seen_digit = true
                if ndigits > 18
                    return parse(Float64, str)
                end
                i += 1
            else
                break
            end
        end
    end

    seen_digit || return parse(Float64, str)

    if i <= n
        b = bytes[i]
        if b == 0x65 || b == 0x45 || b == 0x64 || b == 0x44
            i += 1
            exp_negative = false
            if i <= n
                b = bytes[i]
                if b == 0x2d
                    exp_negative = true
                    i += 1
                elseif b == 0x2b
                    i += 1
                end
            end

            exp_value = 0
            exp_digits = 0
            while i <= n
                b = bytes[i]
                if _isdigitbyte(b)
                    exp_value = 10 * exp_value + Int(b - 0x30)
                    exp_digits += 1
                    i += 1
                else
                    return parse(Float64, str)
                end
            end
            exp_digits == 0 && return parse(Float64, str)
            scale = 10.0 ^ exp_value
            value = exp_negative ? value / scale : value * scale
        else
            return parse(Float64, str)
        end
    end

    return negative ? -value : value
end

@inline function _parse_float(line::AbstractString, bytes, start::Int, stop::Int)
    i = start
    negative = false

    b = bytes[i]
    if b == 0x2d
        negative = true
        i += 1
    elseif b == 0x2b
        i += 1
    end

    value = 0.0
    frac_scale = 0.1
    ndigits = 0
    seen_digit = false

    while i <= stop
        b = bytes[i]
        if _isdigitbyte(b)
            value = muladd(value, 10.0, Float64(b - 0x30))
            ndigits += 1
            seen_digit = true
            if ndigits > 18
                return parse(Float64, SubString(line, start, stop))
            end
            i += 1
        else
            break
        end
    end

    if i <= stop && bytes[i] == 0x2e
        i += 1
        while i <= stop
            b = bytes[i]
            if _isdigitbyte(b)
                value += Float64(b - 0x30) * frac_scale
                frac_scale *= 0.1
                ndigits += 1
                seen_digit = true
                if ndigits > 18
                    return parse(Float64, SubString(line, start, stop))
                end
                i += 1
            else
                break
            end
        end
    end

    seen_digit || return parse(Float64, SubString(line, start, stop))

    if i <= stop
        b = bytes[i]
        if b == 0x65 || b == 0x45 || b == 0x64 || b == 0x44
            i += 1
            exp_negative = false
            if i <= stop
                b = bytes[i]
                if b == 0x2d
                    exp_negative = true
                    i += 1
                elseif b == 0x2b
                    i += 1
                end
            end

            exp_value = 0
            exp_digits = 0
            while i <= stop
                b = bytes[i]
                if _isdigitbyte(b)
                    exp_value = 10 * exp_value + Int(b - 0x30)
                    exp_digits += 1
                    i += 1
                else
                    return parse(Float64, SubString(line, start, stop))
                end
            end
            exp_digits == 0 && return parse(Float64, SubString(line, start, stop))
            scale = 10.0 ^ exp_value
            value = exp_negative ? value / scale : value * scale
        else
            return parse(Float64, SubString(line, start, stop))
        end
    end

    return negative ? -value : value
end

@inline function _reset_column_cache!(state::ParserState)
    state.last_col_name = _EMPTY_FIELD
    state.last_col_key = 0
    state.last_col = 0
    return nothing
end

@inline function _conindex_get(data::MPSData, name::AbstractString, packed_key::UInt64, default::Int, state::ParserState)
    return state.use_packed_names ? _packedmap_get(data.packed_conindices, packed_key, default) : _namemap_get(data.conindices, name, default)
end

@inline function _conindex_haskey(data::MPSData, name::AbstractString, packed_key::UInt64, state::ParserState)
    return state.use_packed_names ? _packedmap_haskey(data.packed_conindices, packed_key) : _namemap_haskey(data.conindices, name)
end

@inline function _varindex_get(data::MPSData, name::AbstractString, packed_key::UInt64, default::Int, state::ParserState)
    return state.use_packed_names ? _packedmap_get(data.packed_varindices, packed_key, default) : _namemap_get(data.varindices, name, default)
end

function _read_rows_fields!(data::MPSData, f1::SubString{String}, f2::SubString{String}, nfields::Int, nline::Int, state::ParserState)
    nfields < 2 && error("Line $nline contains only $nfields fields")
    rtype = _row_type(f1)
    rowkey = state.use_packed_names ? _pack_fixed_name(f2) : UInt64(0)

    if rtype == RTYPE_OBJECTIVE
        if data.objname === nothing
            rowname = String(f2)
            data.objname = rowname
            if state.use_packed_names
                data.packed_conindices[rowkey] = 0
            else
                _namemap_set!(data.conindices, rowname, 0)
            end
        else
            if state.use_packed_names
                data.packed_conindices[rowkey] = -1
            else
                _namemap_set!(data.conindices, String(f2), -1)
            end
        end
        return
    end

    _conindex_haskey(data, f2, rowkey, state) && error("Duplicate row name $(f2) at line $nline")
    data.ncon += 1
    if state.use_packed_names
        data.packed_conindices[rowkey] = data.ncon
        if data.connames !== nothing
            push!(data.connames, String(f2))
        end
    else
        rowname = String(f2)
        _namemap_set!(data.conindices, rowname, data.ncon)
        data.connames !== nothing && push!(data.connames, rowname)
    end
    push!(data.contypes, rtype)

    if rtype == RTYPE_EQUALTO
        push!(data.lcon, 0.0); push!(data.ucon, 0.0)
    elseif rtype == RTYPE_GREATERTHAN
        push!(data.lcon, 0.0); push!(data.ucon, Inf)
    else
        push!(data.lcon, -Inf); push!(data.ucon, 0.0)
    end
end

function _read_rows_line!(data::MPSData, card::MPSCard, state::ParserState)
    return _read_rows_fields!(data, card.f1, card.f2, card.nfields, card.nline, state)
end

function _ensure_variable!(data::MPSData, varname::AbstractString, state::ParserState, packed_key::UInt64 = UInt64(0))
    col = _varindex_get(data, varname, packed_key, 0, state)
    if col != 0
        return col
    end
    data.nvar += 1
    col = data.nvar
    if state.use_packed_names
        data.packed_varindices[packed_key] = col
        if data.varnames !== nothing
            push!(data.varnames, String(varname))
        end
    else
        key = String(varname)
        _namemap_set!(data.varindices, key, col)
        data.varnames !== nothing && push!(data.varnames, key)
    end
    push!(data.c, 0.0)
    push!(data.lvar, NaN)
    push!(data.uvar, NaN)
    push!(data.vartypes, state.integer_section ? VTYPE_MARKED : VTYPE_CONTINUOUS)
    return col
end

function _add_entry!(data::MPSData, row::Int, col::Int, val::Float64)
    push!(data.arows, row)
    push!(data.acols, col)
    push!(data.avals, val)
end

function _apply_column_pair!(data::MPSData, col::Int, rowname::AbstractString, val::Float64, nline::Int, state::ParserState)
    rowkey = state.use_packed_names ? _pack_fixed_name(rowname) : UInt64(0)
    row = _conindex_get(data, rowname, rowkey, -2, state)
    if row == 0
        data.c[col] = val
    elseif row > 0
        _add_entry!(data, row, col, val)
    elseif row != -1
        error("Unknown row $(rowname) at line $nline")
    end
end

@inline function _ensure_variable_fixed!(data::MPSData, line::AbstractString, bytes, start::Int, stop::Int, state::ParserState, packed_key::UInt64)
    col = get(data.packed_varindices, packed_key, 0)
    if col != 0
        return col
    end
    data.nvar += 1
    col = data.nvar
    data.packed_varindices[packed_key] = col
    if data.varnames !== nothing
        push!(data.varnames, String(SubString(line, start, stop)))
    end
    push!(data.c, 0.0)
    push!(data.lvar, NaN)
    push!(data.uvar, NaN)
    push!(data.vartypes, state.integer_section ? VTYPE_MARKED : VTYPE_CONTINUOUS)
    return col
end

@inline function _apply_column_pair_fixed!(data::MPSData, col::Int, rowkey::UInt64, val::Float64, nline::Int)
    row = get(data.packed_conindices, rowkey, -2)
    if row == 0
        data.c[col] = val
    elseif row > 0
        _add_entry!(data, row, col, val)
    elseif row != -1
        error("Unknown row at line $nline")
    end
end

function _read_columns_fixed!(data::MPSData, line::AbstractString, bytes, f1s::Int, f1e::Int, f2s::Int, f2e::Int, f3s::Int, f3e::Int, f4s::Int, f4e::Int, f5s::Int, f5e::Int, nfields::Int, nline::Int, state::ParserState)
    nfields < 3 && error("Line $nline contains only $nfields fields")

    colkey = _pack_fixed_name(bytes, f1s, f1e)
    col = if state.last_col != 0 && colkey == state.last_col_key
        state.last_col
    else
        new_col = _ensure_variable_fixed!(data, line, bytes, f1s, f1e, state, colkey)
        state.last_col_key = colkey
        state.last_col = new_col
        new_col
    end

    rowkey2 = _pack_fixed_name(bytes, f2s, f2e)
    _apply_column_pair_fixed!(data, col, rowkey2, _parse_float(line, bytes, f3s, f3e), nline)
    if nfields >= 5
        rowkey4 = _pack_fixed_name(bytes, f4s, f4e)
        _apply_column_pair_fixed!(data, col, rowkey4, _parse_float(line, bytes, f5s, f5e), nline)
    end
end

@inline function _apply_rhs_fixed!(data::MPSData, rowkey::UInt64, val::Float64, nline::Int)
    row = get(data.packed_conindices, rowkey, -2)
    if row == 0
        data.c0 = -val
    elseif row == -1
        return
    elseif row > 0
        rtype = data.contypes[row]
        if rtype == RTYPE_EQUALTO
            data.lcon[row] = val
            data.ucon[row] = val
        elseif rtype == RTYPE_LESSTHAN
            data.ucon[row] = val
        else
            data.lcon[row] = val
        end
    else
        error("Unknown row at line $nline")
    end
end

function _read_rhs_fixed!(data::MPSData, line::AbstractString, bytes, f1s::Int, f1e::Int, f2s::Int, f2e::Int, f3s::Int, f3e::Int, f4s::Int, f4e::Int, f5s::Int, f5e::Int, nfields::Int, nline::Int)
    nfields < 3 && error("Line $nline contains only $nfields fields")
    if data.rhsname === nothing
        data.rhsname = _span_string(line, f1s, f1e)
    elseif !_span_equals(bytes, f1s, f1e, data.rhsname)
        return
    end
    rowkey2 = _pack_fixed_name(bytes, f2s, f2e)
    _apply_rhs_fixed!(data, rowkey2, _parse_float(line, bytes, f3s, f3e), nline)
    if nfields >= 5
        rowkey4 = _pack_fixed_name(bytes, f4s, f4e)
        _apply_rhs_fixed!(data, rowkey4, _parse_float(line, bytes, f5s, f5e), nline)
    end
end

@inline function _apply_range_fixed!(data::MPSData, rowkey::UInt64, val::Float64, nline::Int)
    row = get(data.packed_conindices, rowkey, -2)
    if row <= 0
        error("Encountered objective or unknown row in RANGES section (l. $nline)")
    end
    rtype = data.contypes[row]
    if rtype == RTYPE_EQUALTO
        if val >= 0.0
            data.ucon[row] += val
        else
            data.lcon[row] += val
        end
    elseif rtype == RTYPE_LESSTHAN
        data.lcon[row] = data.ucon[row] - abs(val)
    else
        data.ucon[row] = data.lcon[row] + abs(val)
    end
end

function _read_ranges_fixed!(data::MPSData, line::AbstractString, bytes, f1s::Int, f1e::Int, f2s::Int, f2e::Int, f3s::Int, f3e::Int, f4s::Int, f4e::Int, f5s::Int, f5e::Int, nfields::Int, nline::Int)
    nfields < 3 && error("Line $nline contains only $nfields fields")
    if data.rngname === nothing
        data.rngname = _span_string(line, f1s, f1e)
    elseif !_span_equals(bytes, f1s, f1e, data.rngname)
        return
    end
    rowkey2 = _pack_fixed_name(bytes, f2s, f2e)
    _apply_range_fixed!(data, rowkey2, _parse_float(line, bytes, f3s, f3e), nline)
    if nfields >= 5 && f4s <= f4e
        rowkey4 = _pack_fixed_name(bytes, f4s, f4e)
        _apply_range_fixed!(data, rowkey4, _parse_float(line, bytes, f5s, f5e), nline)
    end
end

function _read_bounds_fixed!(data::MPSData, line::AbstractString, bytes, f1s::Int, f1e::Int, f2s::Int, f2e::Int, f3s::Int, f3e::Int, f4s::Int, f4e::Int, nfields::Int, nline::Int)
    nfields < 3 && error("Line $nline contains only $nfields fields")
    if data.bndname === nothing
        data.bndname = _span_string(line, f2s, f2e)
    elseif !_span_equals(bytes, f2s, f2e, data.bndname)
        return
    end
    varkey = _pack_fixed_name(bytes, f3s, f3e)
    col = get(data.packed_varindices, varkey, 0)
    col == 0 && error("Unknown column at line $nline")

    if _span_equals(bytes, f1s, f1e, "FR")
        data.lvar[col] = -Inf; data.uvar[col] = Inf; return
    elseif _span_equals(bytes, f1s, f1e, "MI")
        data.lvar[col] = -Inf; return
    elseif _span_equals(bytes, f1s, f1e, "PL")
        data.uvar[col] = Inf; return
    elseif _span_equals(bytes, f1s, f1e, "BV")
        data.vartypes[col] = VTYPE_BINARY; data.lvar[col] = 0.0; data.uvar[col] = 1.0; return
    end

    nfields < 4 && error("At least 4 fields required for bounds at line $nline")
    val = _parse_float(line, bytes, f4s, f4e)
    if _span_equals(bytes, f1s, f1e, "LO")
        data.lvar[col] = val
    elseif _span_equals(bytes, f1s, f1e, "UP")
        data.uvar[col] = val
    elseif _span_equals(bytes, f1s, f1e, "FX")
        data.lvar[col] = val; data.uvar[col] = val
    elseif _span_equals(bytes, f1s, f1e, "LI")
        data.vartypes[col] = VTYPE_INTEGER; data.lvar[col] = val
    elseif _span_equals(bytes, f1s, f1e, "UI")
        data.vartypes[col] = VTYPE_INTEGER; data.uvar[col] = val
    end
end

function _read_columns_fields!(data::MPSData, f1::SubString{String}, f2::SubString{String}, f3::SubString{String}, f4::SubString{String}, f5::SubString{String}, nfields::Int, nline::Int, state::ParserState)
    nfields < 3 && error("Line $nline contains only $nfields fields")
    col = if state.use_packed_names
        colkey = _pack_fixed_name(f1)
        if state.last_col != 0 && colkey == state.last_col_key
            state.last_col
        else
            new_col = _ensure_variable!(data, f1, state, colkey)
            state.last_col_key = colkey
            state.last_col = new_col
            new_col
        end
    else
        if state.last_col != 0 && f1 == state.last_col_name
            state.last_col
        else
            new_col = _ensure_variable!(data, f1, state)
            state.last_col_name = f1
            state.last_col = new_col
            new_col
        end
    end
    _apply_column_pair!(data, col, f2, _parse_float(f3), nline, state)
    nfields >= 5 && _apply_column_pair!(data, col, f4, _parse_float(f5), nline, state)
end

function _read_columns_line!(data::MPSData, card::MPSCard, state::ParserState)
    return _read_columns_fields!(data, card.f1, card.f2, card.f3, card.f4, card.f5, card.nfields, card.nline, state)
end

function _apply_rhs!(data::MPSData, rowname::AbstractString, val::Float64, nline::Int, state::ParserState)
    rowkey = state.use_packed_names ? _pack_fixed_name(rowname) : UInt64(0)
    row = _conindex_get(data, rowname, rowkey, -2, state)
    if row == 0
        data.c0 = -val
    elseif row == -1
        return
    elseif row > 0
        idx = row
        rtype = data.contypes[idx]
        if rtype == RTYPE_EQUALTO
            data.lcon[idx] = val
            data.ucon[idx] = val
        elseif rtype == RTYPE_LESSTHAN
            data.ucon[idx] = val
        else
            data.lcon[idx] = val
        end
    else
        error("Unknown row $(rowname) at line $nline")
    end
end

function _read_rhs_fields!(data::MPSData, f1::SubString{String}, f2::SubString{String}, f3::SubString{String}, f4::SubString{String}, f5::SubString{String}, nfields::Int, nline::Int, state::ParserState)
    nfields < 3 && error("Line $nline contains only $nfields fields")
    if data.rhsname === nothing
        data.rhsname = String(f1)
    elseif data.rhsname != f1
        return
    end
    _apply_rhs!(data, f2, _parse_float(f3), nline, state)
    nfields >= 5 && _apply_rhs!(data, f4, _parse_float(f5), nline, state)
end

function _read_rhs_line!(data::MPSData, card::MPSCard, state::ParserState)
    return _read_rhs_fields!(data, card.f1, card.f2, card.f3, card.f4, card.f5, card.nfields, card.nline, state)
end

function _apply_range!(data::MPSData, rowname::AbstractString, val::Float64, nline::Int, state::ParserState)
    rowkey = state.use_packed_names ? _pack_fixed_name(rowname) : UInt64(0)
    row = _conindex_get(data, rowname, rowkey, -2, state)
    if row <= 0
        error("Encountered objective or unknown row $(rowname) in RANGES section (l. $nline)")
    end
    rtype = data.contypes[row]
    if rtype == RTYPE_EQUALTO
        if val >= 0.0
            data.ucon[row] += val
        else
            data.lcon[row] += val
        end
    elseif rtype == RTYPE_LESSTHAN
        data.lcon[row] = data.ucon[row] - abs(val)
    else
        data.ucon[row] = data.lcon[row] + abs(val)
    end
end

function _read_ranges_fields!(data::MPSData, f1::SubString{String}, f2::SubString{String}, f3::SubString{String}, f4::SubString{String}, f5::SubString{String}, nfields::Int, nline::Int, state::ParserState)
    nfields < 3 && error("Line $nline contains only $nfields fields")
    if data.rngname === nothing
        data.rngname = String(f1)
    elseif data.rngname != f1
        return
    end
    _apply_range!(data, f2, _parse_float(f3), nline, state)
    if nfields >= 5 && !isempty(f4)
        _apply_range!(data, f4, _parse_float(f5), nline, state)
    end
end

function _read_ranges_line!(data::MPSData, card::MPSCard, state::ParserState)
    return _read_ranges_fields!(data, card.f1, card.f2, card.f3, card.f4, card.f5, card.nfields, card.nline, state)
end

function _read_bounds_fields!(data::MPSData, f1::SubString{String}, f2::SubString{String}, f3::SubString{String}, f4::SubString{String}, nfields::Int, nline::Int, state::ParserState)
    nfields < 3 && error("Line $nline contains only $nfields fields")
    if data.bndname === nothing
        data.bndname = String(f2)
    elseif data.bndname != f2
        return
    end
    varkey = state.use_packed_names ? _pack_fixed_name(f3) : UInt64(0)
    col = _varindex_get(data, f3, varkey, 0, state)
    col == 0 && error("Unknown column $(f3)")
    btype = f1
    if btype == "FR"
        data.lvar[col] = -Inf; data.uvar[col] = Inf; return
    elseif btype == "MI"
        data.lvar[col] = -Inf; return
    elseif btype == "PL"
        data.uvar[col] = Inf; return
    elseif btype == "BV"
        data.vartypes[col] = VTYPE_BINARY; data.lvar[col] = 0.0; data.uvar[col] = 1.0; return
    end
    nfields < 4 && error("At least 4 fields required for $(btype) bounds")
    val = _parse_float(f4)
    if btype == "LO"
        data.lvar[col] = val
    elseif btype == "UP"
        data.uvar[col] = val
    elseif btype == "FX"
        data.lvar[col] = val; data.uvar[col] = val
    elseif btype == "LI"
        data.vartypes[col] = VTYPE_INTEGER; data.lvar[col] = val
    elseif btype == "UI"
        data.vartypes[col] = VTYPE_INTEGER; data.uvar[col] = val
    end
end

function _read_bounds_line!(data::MPSData, card::MPSCard, state::ParserState)
    return _read_bounds_fields!(data, card.f1, card.f2, card.f3, card.f4, card.nfields, card.nline, state)
end

function _finalize_bounds!(data::MPSData)
    for j in 1:data.nvar
        l = data.lvar[j]
        u = data.uvar[j]
        vt = data.vartypes[j]
        if isnan(l) && isnan(u)
            data.lvar[j] = 0.0
            data.uvar[j] = vt == VTYPE_MARKED ? 1.0 : Inf
        elseif isnan(l)
            data.lvar[j] = u < 0 ? -Inf : 0.0
        elseif isnan(u)
            data.uvar[j] = Inf
        end
        if vt == VTYPE_MARKED
            data.vartypes[j] = VTYPE_INTEGER
        end
    end
end

function _to_lpfiledata(data::MPSData; keep_names::Bool = false)
    sign = data.objsense == :max ? -1.0 : 1.0
    c = sign == 1.0 ? data.c : sign .* data.c
    obj_constant = sign * data.c0
    rownames = keep_names ? data.connames : nothing
    colnames = keep_names ? data.varnames : nothing
    return LPFileData(
        data.ncon,
        data.nvar,
        data.arows,
        data.acols,
        data.avals,
        c,
        data.lcon,
        data.ucon,
        data.lvar,
        data.uvar,
        obj_constant,
        rownames,
        colnames,
    )
end

function _process_header!(data::MPSData, f1::SubString{String}, f2::SubString{String}, nline::Int, state::ParserState)
    sec = _section_type(f1)
    if sec == SECTION_NAME
        data.name = isempty(f2) ? nothing : String(f2)
    elseif sec == SECTION_OBJSENSE
        state.current_section = SECTION_OBJSENSE
    elseif sec == SECTION_ROWS
        state.current_section = SECTION_ROWS
    elseif sec == SECTION_COLUMNS
        state.current_section = SECTION_COLUMNS
    elseif sec == SECTION_RHS
        state.current_section = SECTION_RHS
    elseif sec == SECTION_BOUNDS
        state.current_section = SECTION_BOUNDS
    elseif sec == SECTION_RANGES
        state.current_section = SECTION_RANGES
    elseif sec == SECTION_QUADOBJ || sec == SECTION_QMATRIX
        error("Quadratic sections are not supported by this LP reader.")
    elseif sec == SECTION_OBJECT_BOUND
        state.current_section = SECTION_OBJECT_BOUND
    elseif sec == SECTION_ENDATA
        state.endata_read = true
        return nothing
    else
        error("Unknown section header $(f1) at line $nline")
    end
    _reset_column_cache!(state)
    return nothing
end

function _process_fields!(data::MPSData, f1::SubString{String}, f2::SubString{String}, f3::SubString{String}, f4::SubString{String}, f5::SubString{String}, nfields::Int, nline::Int, state::ParserState)
    if state.current_section == SECTION_OBJSENSE
        if f1 == "MIN"
            data.objsense = :min
        elseif f1 == "MAX"
            data.objsense = :max
        end
    elseif state.current_section == SECTION_ROWS
        _read_rows_fields!(data, f1, f2, nfields, nline, state)
    elseif state.current_section == SECTION_COLUMNS
        if f2 == "'MARKER'"
            if f3 == "'INTORG'"
                state.integer_section = true
            elseif f3 == "'INTEND'"
                state.integer_section = false
            end
            _reset_column_cache!(state)
        else
            _read_columns_fields!(data, f1, f2, f3, f4, f5, nfields, nline, state)
        end
    elseif state.current_section == SECTION_RHS
        _read_rhs_fields!(data, f1, f2, f3, f4, f5, nfields, nline, state)
    elseif state.current_section == SECTION_BOUNDS
        _read_bounds_fields!(data, f1, f2, f3, f4, nfields, nline, state)
    elseif state.current_section == SECTION_RANGES
        _read_ranges_fields!(data, f1, f2, f3, f4, f5, nfields, nline, state)
    elseif state.current_section == SECTION_OBJECT_BOUND
        nothing
    end
    return nothing
end

function _process_card!(data::MPSData, card::MPSCard, state::ParserState)
    if card.isheader
        return _process_header!(data, card.f1, card.f2, card.nline, state)
    end
    return _process_fields!(data, card.f1, card.f2, card.f3, card.f4, card.f5, card.nfields, card.nline, state)
end

function _process_fixed_line!(data::MPSData, line::AbstractString, nline::Int, state::ParserState)
    len = ncodeunits(line)
    len == 0 && return nothing
    bytes = codeunits(line)
    firstb = bytes[1]

    if firstb == UInt8('*') || firstb == UInt8('&')
        return nothing
    end

    if !_isspacebyte(firstb)
        header = _parse_section_header(line, nline)
        return _process_header!(data, header.f1, header.f2, nline, state)
    end

    if state.current_section == SECTION_OBJSENSE
        f1 = _trim_range_fast(line, bytes, 2, 3, len)
        isempty(f1) && (f1 = _trim_range_fast(line, bytes, 5, 12, len))
        return _process_fields!(data, f1, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, isempty(f1) ? 0 : 1, nline, state)
    elseif state.current_section == SECTION_ROWS
        f1 = _trim_range_fast(line, bytes, 2, 3, len)
        f2 = _trim_range_fast(line, bytes, 5, 12, len)
        return _process_fields!(data, f1, f2, _EMPTY_FIELD, _EMPTY_FIELD, _EMPTY_FIELD, isempty(f1) ? 0 : (isempty(f2) ? 1 : 2), nline, state)
    elseif state.current_section == SECTION_COLUMNS
        f1s, f1e = _trim_span_fast(bytes, 5, 12, len)
        f2s, f2e = _trim_span_fast(bytes, 15, 22, len)
        f3s, f3e = _trim_span_fast(bytes, 25, 36, len)
        f4s, f4e = _trim_span_fast(bytes, 40, 47, len)
        f5s, f5e = _trim_span_fast(bytes, 50, 61, len)
        nfields = _count_spans5(f1s, f1e, f2s, f2e, f3s, f3e, f4s, f4e, f5s, f5e)
        if f2s <= f2e && _span_equals(bytes, f2s, f2e, "'MARKER'")
            if f3s <= f3e && _span_equals(bytes, f3s, f3e, "'INTORG'")
                state.integer_section = true
            elseif f3s <= f3e && _span_equals(bytes, f3s, f3e, "'INTEND'")
                state.integer_section = false
            end
            _reset_column_cache!(state)
            return nothing
        end
        return _read_columns_fixed!(data, line, bytes, f1s, f1e, f2s, f2e, f3s, f3e, f4s, f4e, f5s, f5e, nfields, nline, state)
    elseif state.current_section == SECTION_RHS || state.current_section == SECTION_RANGES
        f1s, f1e = _trim_span_fast(bytes, 5, 12, len)
        f2s, f2e = _trim_span_fast(bytes, 15, 22, len)
        f3s, f3e = _trim_span_fast(bytes, 25, 36, len)
        f4s, f4e = _trim_span_fast(bytes, 40, 47, len)
        f5s, f5e = _trim_span_fast(bytes, 50, 61, len)
        nfields = _count_spans5(f1s, f1e, f2s, f2e, f3s, f3e, f4s, f4e, f5s, f5e)
        if state.current_section == SECTION_RHS
            return _read_rhs_fixed!(data, line, bytes, f1s, f1e, f2s, f2e, f3s, f3e, f4s, f4e, f5s, f5e, nfields, nline)
        end
        return _read_ranges_fixed!(data, line, bytes, f1s, f1e, f2s, f2e, f3s, f3e, f4s, f4e, f5s, f5e, nfields, nline)
    elseif state.current_section == SECTION_BOUNDS
        f1s, f1e = _trim_span_fast(bytes, 2, 3, len)
        f2s, f2e = _trim_span_fast(bytes, 5, 12, len)
        f3s, f3e = _trim_span_fast(bytes, 15, 22, len)
        f4s, f4e = _trim_span_fast(bytes, 25, 36, len)
        nfields = _count_spans4(f1s, f1e, f2s, f2e, f3s, f3e, f4s, f4e)
        return _read_bounds_fixed!(data, line, bytes, f1s, f1e, f2s, f2e, f3s, f3e, f4s, f4e, nfields, nline)
    elseif state.current_section == SECTION_OBJECT_BOUND
        return nothing
    end

    return nothing
end

function _read_mps_native(io::IO; mpsformat::Symbol = :fixed, keep_names::Bool = false, hints::CapacityHints = _DEFAULT_CAPACITY_HINTS)
    mpsformat in (:fixed, :free) || throw(ArgumentError("Unsupported mpsformat $(mpsformat). Use :fixed or :free."))
    use_packed_names = mpsformat == :fixed
    data = _create_data(hints = hints, keep_names = keep_names, use_packed_names = use_packed_names)
    state = ParserState(SECTION_NONE, false, false, _EMPTY_FIELD, 0, 0, use_packed_names)

    for (nline, line) in enumerate(eachline(io))
        if mpsformat == :fixed
            _process_fixed_line!(data, line, nline, state)
        else
            card = _parse_free_card(line, nline)
            card.iscomment && continue
            _process_card!(data, card, state)
        end
        state.endata_read && break
    end

    state.endata_read || nothing
    _finalize_bounds!(data)
    return data
end

function _read_mps_native_text(text::String; mpsformat::Symbol = :fixed, keep_names::Bool = false, hints::CapacityHints = _DEFAULT_CAPACITY_HINTS)
    mpsformat in (:fixed, :free) || throw(ArgumentError("Unsupported mpsformat $(mpsformat). Use :fixed or :free."))
    use_packed_names = mpsformat == :fixed
    data = _create_data(hints = hints, keep_names = keep_names, use_packed_names = use_packed_names)
    state = ParserState(SECTION_NONE, false, false, _EMPTY_FIELD, 0, 0, use_packed_names)

    _scan_text_lines!(data, state, text, 0, mpsformat, false)

    state.endata_read || nothing
    _finalize_bounds!(data)
    return data
end

function _scan_text_lines!(data::MPSData, state::ParserState, text::String, start_nline::Int, mpsformat::Symbol, require_terminal_newline::Bool)
    bytes = codeunits(text)
    n = ncodeunits(text)
    start = 1
    nline = start_nline

    while start <= n
        stop = start
        while stop <= n && bytes[stop] != 0x0a
            stop += 1
        end
        if stop > n && require_terminal_newline
            return nline, start
        end

        nline += 1
        line_stop = stop - 1
        line = line_stop >= start ? SubString(text, start, line_stop) : ""
        if mpsformat == :fixed
            _process_fixed_line!(data, line, nline, state)
            state.endata_read && return nline, stop + 1
        else
            card = _parse_free_card(line, nline)
            if !card.iscomment
                _process_card!(data, card, state)
                state.endata_read && return nline, stop + 1
            end
        end
        start = stop + 1
    end

    return nline, n + 1
end

function _read_mps_native_chunked(io::IO; mpsformat::Symbol = :fixed, keep_names::Bool = false, hints::CapacityHints = _DEFAULT_CAPACITY_HINTS, chunk_bytes::Int = _STREAM_CHUNK_BYTES)
    mpsformat in (:fixed, :free) || throw(ArgumentError("Unsupported mpsformat $(mpsformat). Use :fixed or :free."))
    use_packed_names = mpsformat == :fixed
    data = _create_data(hints = hints, keep_names = keep_names, use_packed_names = use_packed_names)
    state = ParserState(SECTION_NONE, false, false, _EMPTY_FIELD, 0, 0, use_packed_names)
    buffer = Vector{UInt8}(undef, chunk_bytes)
    carry = UInt8[]
    nline = 0

    while !eof(io) && !state.endata_read
        nread = readbytes!(io, buffer, chunk_bytes)
        nread == 0 && break

        total = length(carry) + nread
        bytes = Vector{UInt8}(undef, total)
        carry_len = length(carry)
        if carry_len != 0
            copyto!(bytes, 1, carry, 1, carry_len)
        end
        copyto!(bytes, carry_len + 1, buffer, 1, nread)
        text = String(bytes)

        nline, carry_start = _scan_text_lines!(data, state, text, nline, mpsformat, true)
        if state.endata_read
            empty!(carry)
            break
        end

        if carry_start <= ncodeunits(text)
            resize!(carry, ncodeunits(text) - carry_start + 1)
            copyto!(carry, 1, codeunits(text), carry_start, length(carry))
        else
            empty!(carry)
        end
    end

    if !state.endata_read && !isempty(carry)
        text = String(copy(carry))
        _scan_text_lines!(data, state, text, nline, mpsformat, false)
    end

    state.endata_read || nothing
    _finalize_bounds!(data)
    return data
end

function read_mps(io::IO; keep_names::Bool = false, mpsformat::Symbol = :auto)
    format = mpsformat == :auto ? :free : mpsformat
    format in (:fixed, :free) || throw(ArgumentError("Unsupported mpsformat $(mpsformat). Use :auto, :fixed, or :free."))
    data = _read_mps_native(io; mpsformat = format, keep_names = keep_names)
    return _to_lpfiledata(data; keep_names = keep_names)
end

function _read_mps_file(path::String, keep_names::Bool, format::Symbol, hints::CapacityHints)
    lower_path = lowercase(path)
    if endswith(lower_path, ".mps")
        file_size = try filesize(path) catch; 0 end
        if file_size > 0 && file_size <= _READALL_THRESHOLD_BYTES
            text = read(path, String)
            data = _read_mps_native_text(text; mpsformat = format, keep_names = keep_names, hints = hints)
            return _to_lpfiledata(data; keep_names = keep_names)
        else
            return open(path, "r") do io
                data = _read_mps_native(io; mpsformat = format, keep_names = keep_names, hints = hints)
                return _to_lpfiledata(data; keep_names = keep_names)
            end
        end
    elseif endswith(lower_path, ".mps.gz")
        return open(path, "r") do raw_io
            gzip_io = GzipDecompressorStream(raw_io)
            try
                data = _read_mps_native_chunked(gzip_io; mpsformat = format, keep_names = keep_names, hints = hints)
                return _to_lpfiledata(data; keep_names = keep_names)
            finally
                close(gzip_io)
            end
        end
    end
    error("Unsupported file format for $path. Expected .mps or .mps.gz")
end

function read_mps(filename::AbstractString; keep_names::Bool = false, mpsformat::Symbol = :auto)
    path = String(filename)
    lower_path = lowercase(path)
    (endswith(lower_path, ".mps") || endswith(lower_path, ".mps.gz")) ||
        error("Unsupported file format for $path. Expected .mps or .mps.gz")

    hints = _estimate_capacities(path)
    if mpsformat == :auto
        try
            return _read_mps_file(path, keep_names, :fixed, hints)
        catch err
            if err isa ArgumentError || err isa ErrorException
                return _read_mps_file(path, keep_names, :free, hints)
            end
            rethrow(err)
        end
    end

    mpsformat in (:fixed, :free) || throw(ArgumentError("Unsupported mpsformat $(mpsformat). Use :auto, :fixed, or :free."))
    return _read_mps_file(path, keep_names, mpsformat, hints)
end