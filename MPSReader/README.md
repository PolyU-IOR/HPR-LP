# MPSReader.jl

`MPSReader.jl` is a standalone Julia package for reading large `.mps` and `.mps.gz` LP instances directly into numeric arrays.

It is intended as a lighter ingest path than `JuMP.read_from_file` when the downstream consumer only needs LP coefficient data.

The current implementation is a native Julia parser based on the reference logic in `mps_reader.cpp`. It does not depend on `QPSReader` or `MathOptInterface`.

## Features

- direct `.mps` and `.mps.gz` support,
- streamed gzip decompression through `CodecZlib`,
- native Julia fixed/free MPS parsing,
- returns numeric LP data instead of building a JuMP model,
- exposes sparse matrix construction separately.

## Example

```julia
using MPSReader

data = read_mps("instance.mps.gz")
A = sparse_matrix(data)

@show data.nrow data.ncol length(data.avals)
```

By default, `read_mps` does not preserve row or column names. That is the faster path and is intended for solver ingestion. If names are needed, call `read_mps(path; keep_names = true)`.

By default, file-based reads use `mpsformat = :auto`, which tries the fixed-format parser first and falls back to the free-format parser if needed.

## Current Scope

This package currently focuses on LP-style MPS ingest into numeric arrays using Julia's built-in gzip streaming. It does not build JuMP models and does not implement a GPU parser.

## Benchmarking Against JuMP

The package includes a simple benchmark script that compares `JuMP.read_from_file` against `MPSReader.read_mps` using Julia's built-in gzip path only.

Run it with:

```bash
julia --project=MPSReader/bench MPSReader/bench/benchmark_readers.jl
```

Or point it at a specific file and repeat count:

```bash
julia --project=MPSReader/bench MPSReader/bench/benchmark_readers.jl path/to/problem.mps 5
```

This benchmark uses Julia gzip only and does not require any external decompression tool.