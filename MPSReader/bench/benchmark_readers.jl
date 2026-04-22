using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "src")))

using CodecZlib
using JuMP
using MPSReader
using Printf

function ensure_gzip_copy(mps_path::String)
    if endswith(lowercase(mps_path), ".mps.gz")
        return mps_path, nothing
    end
    endswith(lowercase(mps_path), ".mps") || error("Expected an .mps or .mps.gz file, got $mps_path")

    gz_path = tempname() * ".mps.gz"
    open(mps_path, "r") do src
        open(gz_path, "w") do dst
            gz = GzipCompressorStream(dst)
            try
                write(gz, read(src))
            finally
                close(gz)
            end
        end
    end
    return gz_path, gz_path
end

function bench(label::String, f::Function; repeats::Int)
    warmup_value = f()
    timings = Float64[]
    for _ in 1:repeats
        GC.gc()
        elapsed = @elapsed begin
            value = f()
            value === nothing || nothing
        end
        push!(timings, elapsed)
    end

    println(rpad(label, 28),
        @sprintf("avg = %8.4f s", sum(timings) / length(timings)),
        @sprintf("  min = %8.4f s", minimum(timings)),
        @sprintf("  max = %8.4f s", maximum(timings)))
    return warmup_value, timings
end

function main(args)
    mps_path = isempty(args) ? normpath(joinpath(@__DIR__, "..", "..", "model.mps")) : normpath(args[1])
    repeats = length(args) >= 2 ? parse(Int, args[2]) : 3

    isfile(mps_path) || error("MPS file not found: $mps_path")

    gz_path, temp_gz_path = ensure_gzip_copy(mps_path)

    println("Benchmarking readers")
    println("  plain file: ", mps_path)
    println("  gzip file:  ", gz_path)
    println("  repeats:    ", repeats)
    println("  gzip mode:  Julia gzip only")
    println("  names:      disabled for MPSReader benchmark path")
    println()

    try
        jump_plain, _ = bench("JuMP read_from_file .mps", () -> read_from_file(mps_path); repeats = repeats)
        mps_plain, _ = bench("MPSReader read_mps .mps", () -> read_mps(mps_path); repeats = repeats)
        jump_gz, _ = bench("JuMP read_from_file .gz", () -> read_from_file(gz_path); repeats = repeats)
        mps_gz, _ = bench("MPSReader read_mps .gz", () -> read_mps(gz_path); repeats = repeats)

        println()
        println("Sanity check")
        println("  JuMP plain vars: ", num_variables(jump_plain))
        println("  MPSReader plain shape: ", (mps_plain.nrow, mps_plain.ncol, length(mps_plain.avals)))
        println("  JuMP gzip vars: ", num_variables(jump_gz))
        println("  MPSReader gzip shape: ", (mps_gz.nrow, mps_gz.ncol, length(mps_gz.avals)))
    finally
        if temp_gz_path !== nothing && isfile(temp_gz_path)
            rm(temp_gz_path; force = true)
        end
    end
end

main(ARGS)