using Libdl

# ================= 1. Path Configuration =================
const DEPS_DIR = @__DIR__
const PSLP_DIR = joinpath(DEPS_DIR, "PSLP")

# Check if source code has been downloaded successfully
if !isdir(PSLP_DIR)
    error("PSLP source code not found in deps/PSLP. Please check your git submodule.")
end

const SRC_DIR = joinpath(PSLP_DIR, "src")
const INC_DIR = joinpath(PSLP_DIR, "include")

# Define the output library file path
const LIB_NAME = "libPSLP.$(Libdl.dlext)"
const LIB_PATH = joinpath(DEPS_DIR, LIB_NAME)
const CMAKE_LISTS_PATH = joinpath(PSLP_DIR, "CMakeLists.txt")

function detect_pslp_version(cmake_lists_path::String)
    text = read(cmake_lists_path, String)
    parts = String[]
    for field in ("MAJOR", "MINOR", "PATCH")
        m = match(Regex("set\\(PSLP_VERSION_$(field)\\s+([0-9]+)\\)"), text)
        m === nothing && error("Could not detect PSLP_VERSION_$(field) in $cmake_lists_path")
        push!(parts, m.captures[1])
    end
    return join(parts, ".")
end

# ================= 2. Get Source File List =================
# Julia's run(Cmd) does not support *.c wildcards, so we need to manually retrieve the file list.
println("Build: Searching for source files...")

core_dir = joinpath(SRC_DIR, "core")
explorers_dir = joinpath(SRC_DIR, "explorers")

# Get core/*.c files
if isdir(core_dir)
    core_srcs = filter(x -> endswith(x, ".c"), readdir(core_dir, join=true))
else
    core_srcs = String[]
end

# Get explorers/*.c files
if isdir(explorers_dir)
    explorer_srcs = filter(x -> endswith(x, ".c"), readdir(explorers_dir, join=true))
else
    explorer_srcs = String[]
end

# Combine all source files
all_srcs = [core_srcs; explorer_srcs]

if isempty(all_srcs)
    error("No .c source files found in $SRC_DIR")
end

println("Found $(length(all_srcs)) source files.")

# ================= 3. Compilation Logic =================
println("Build: Starting compilation of PSLP...")

pslp_version = detect_pslp_version(CMAKE_LISTS_PATH)
compile_flags = String[
    "-shared",
    "-fPIC",
    "-O3",
    "-DNDEBUG",
    "-DPSLP_VERSION=\"$pslp_version\"",
]

if Sys.islinux()
    push!(compile_flags, "-D_POSIX_C_SOURCE=200809L")
    push!(compile_flags, "-pthread")
end

# Construct the GCC compilation command
# Note: $all_srcs is an array; Julia will automatically expand it into multiple file arguments, eliminating the need for *.c
compile_cmd = `gcc $(compile_flags)
               -I $INC_DIR
               -I $(joinpath(INC_DIR, "core"))
               -I $(joinpath(INC_DIR, "data_structures"))
               -I $(joinpath(INC_DIR, "explorers"))
               -I $(joinpath(INC_DIR, "PSLP"))
               $all_srcs
               -o $LIB_PATH
               -lm`

try
    run(compile_cmd)
    println("Build: Success! Library generated at $LIB_PATH")
catch e
    error("Build: Compilation failed. Error: $e")
end
