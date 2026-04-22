module MPSReader

using CodecZlib
using SparseArrays

export LPFileData, read_mps, sparse_matrix

include("reader.jl")

end