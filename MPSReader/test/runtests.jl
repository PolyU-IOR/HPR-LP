using Test
using CodecZlib
using MPSReader

const FIXTURE = joinpath(@__DIR__, "fixtures", "simple.mps")

@testset "MPSReader" begin
    @testset "Read plain MPS" begin
        data = read_mps(FIXTURE)
        @test data.nrow == 2
        @test data.ncol == 2
        @test length(data.avals) == 4
        @test data.c == [-3.0, -5.0]
        @test data.ucon == [10.0, 12.0]
        @test data.lvar == [0.0, 0.0]
        @test data.colnames === nothing

        A = sparse_matrix(data)
        @test size(A) == (2, 2)
        @test A[1, 1] == 1.0
        @test A[1, 2] == 2.0
        @test A[2, 1] == 3.0
        @test A[2, 2] == 1.0
    end

    @testset "Read gzipped MPS" begin
        mktemp() do path, io
            close(io)
            gz_path = path * ".mps.gz"
            open(gz_path, "w") do raw_io
                gz_io = GzipCompressorStream(raw_io)
                try
                    write(gz_io, read(FIXTURE, String))
                finally
                    close(gz_io)
                end
            end

            data = read_mps(gz_path)
            @test data.nrow == 2
            @test data.ncol == 2
            @test data.c == [-3.0, -5.0]
        end
    end

    @testset "Names can be kept" begin
        data = read_mps(FIXTURE; keep_names = true)
        @test data.rownames == ["C1", "C2"]
        @test data.colnames == ["X1", "X2"]
    end
end