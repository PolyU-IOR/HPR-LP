using Test
using HPRLP
using SparseArrays
using LinearAlgebra

@testset "HPRLP.jl" begin
    
    @testset "MPS File Solving" begin
        # Test solving an MPS file
        mps_file = joinpath(@__DIR__, "..", "model.mps")
        
        if isfile(mps_file)
            params = HPRLP.HPRLP_parameters()
            params.time_limit = 60
            params.stoptol = 1e-4
            params.use_gpu = false  # CPU for testing
            params.warm_up = false
            params.verbose = false  # Silent mode for tests
            
            result = HPRLP.run_single(mps_file, params)
            
            @test result.output_type == "OPTIMAL"
            # Problem: min -3x1 - 5x2, s.t. x1+2x2<=10, 3x1+x2<=12, x1,x2>=0
            # Optimal: x1=2.8, x2=3.6, objective=-26.4
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
            @test result.iter > 0
            @test result.time > 0
            @test length(result.x) == 2
        else
            @warn "MPS test file not found at $mps_file, skipping MPS test"
        end
    end
    
    @testset "Basic LP Problem - Direct API" begin
        # Same problem as MPS file:
        # min -3x1 - 5x2
        # s.t. x1 + 2x2 <= 10  (equivalent to -x1 - 2x2 >= -10)
        #      3x1 + x2 <= 12  (equivalent to -3x1 - x2 >= -12)
        #      x1 >= 0, x2 >= 0
        # Optimal: x1=2.8, x2=3.6, objective=-26.4
        
        A = sparse([-1.0 -2.0; -3.0 -1.0])
        AL = Vector{Float64}([-10.0, -12.0])
        AU = Vector{Float64}([Inf, Inf])
        c = Vector{Float64}([-3.0, -5.0])
        l = Vector{Float64}([0.0, 0.0])
        u = Vector{Float64}([Inf, Inf])
        obj_constant = 0.0
        
        params = HPRLP.HPRLP_parameters()
        params.time_limit = 60
        params.stoptol = 1e-4
        params.use_gpu = false  # CPU for testing
        params.warm_up = false
        params.verbose = false  # Silent mode for tests
        
        result = HPRLP.run_lp(A, AL, AU, c, l, u, obj_constant, params)
        
        @test result.output_type == "OPTIMAL"
        @test isapprox(result.primal_obj, -26.4, atol=1e-2)
        @test result.x[1] >= -1e-6  # x1 >= 0
        @test result.x[2] >= -1e-6  # x2 >= 0
    end
    
    @testset "JuMP Integration - Optimizer" begin
        # Test that Optimizer is exported and can be instantiated
        @test isdefined(HPRLP, :Optimizer)
        
        # Try to create an optimizer instance
        optimizer = HPRLP.Optimizer()
        @test optimizer isa HPRLP.Optimizer
    end
    
    @testset "Parameter Validation" begin
        params = HPRLP.HPRLP_parameters()
        
        # Test default values
        @test params.stoptol == 1e-4
        @test params.max_iter == typemax(Int32)
        @test params.time_limit == 3600.0
        @test params.check_iter == 150
        @test params.use_Ruiz_scaling == true
        @test params.use_Pock_Chambolle_scaling == true
        @test params.use_bc_scaling == true
        @test params.use_gpu == true
        @test params.device_number == 0
        @test params.warm_up == true
        @test params.print_frequency == -1
        @test params.verbose == true
        
        # Test parameter modification
        params.stoptol = 1e-6
        @test params.stoptol == 1e-6
        
        params.use_gpu = false
        @test params.use_gpu == false
        
        params.verbose = false
        @test params.verbose == false
    end
    
    @testset "Results Structure" begin
        # Create a simple problem to get results
        A = sparse([1.0 0.0; 0.0 1.0])
        AL = Vector{Float64}([0.0, 0.0])
        AU = Vector{Float64}([1.0, 1.0])
        c = Vector{Float64}([1.0, 1.0])
        l = Vector{Float64}([0.0, 0.0])
        u = Vector{Float64}([1.0, 1.0])
        obj_constant = 0.0
        
        params = HPRLP.HPRLP_parameters()
        params.time_limit = 60
        params.stoptol = 1e-4
        params.use_gpu = false
        params.warm_up = false
        params.verbose = false  # Silent mode for tests
        
        result = HPRLP.run_lp(A, AL, AU, c, l, u, obj_constant, params)
        
        # Test that result has all expected fields
        @test isdefined(result, :iter)
        @test isdefined(result, :time)
        @test isdefined(result, :primal_obj)
        @test isdefined(result, :residuals)
        @test isdefined(result, :gap)
        @test isdefined(result, :output_type)
        @test isdefined(result, :x)
        
        # Test types
        @test result.iter isa Int
        @test result.time isa Float64
        @test result.primal_obj isa Float64
        @test result.x isa Vector{Float64}
        @test result.output_type isa String
    end
    
    @testset "Bounded Variables LP" begin
        # min x1 + 2*x2
        # s.t. x1 + x2 = 1
        #      0 <= x1 <= 1
        #      0 <= x2 <= 1
        
        A = sparse([1.0 1.0])
        AL = Vector{Float64}([1.0])
        AU = Vector{Float64}([1.0])
        c = Vector{Float64}([1.0, 2.0])
        l = Vector{Float64}([0.0, 0.0])
        u = Vector{Float64}([1.0, 1.0])
        obj_constant = 0.0
        
        params = HPRLP.HPRLP_parameters()
        params.time_limit = 60
        params.stoptol = 1e-4
        params.use_gpu = false
        params.warm_up = false
        params.verbose = false  # Silent mode for tests
        
        result = HPRLP.run_lp(A, AL, AU, c, l, u, obj_constant, params)
        
        @test result.output_type == "OPTIMAL"
        # Optimal solution should be x1=1, x2=0, objective=1
        @test isapprox(result.primal_obj, 1.0, atol=1e-3)
        @test isapprox(result.x[1], 1.0, atol=1e-3)
        @test isapprox(result.x[2], 0.0, atol=1e-3)
    end

    @testset "GPU Parameter Validation" begin
        # Test GPU fallback when invalid device number is specified
        A = sparse([-1.0 -2.0; -3.0 -1.0])
        AL = Vector{Float64}([-10.0, -12.0])
        AU = Vector{Float64}([Inf, Inf])
        c = Vector{Float64}([-3.0, -5.0])
        l = Vector{Float64}([0.0, 0.0])
        u = Vector{Float64}([Inf, Inf])
        obj_constant = 0.0

        @testset "Invalid GPU device number fallback" begin
            params = HPRLP.HPRLP_parameters()
            params.use_gpu = true
            params.device_number = 999  # Invalid device number
            params.warm_up = false
            params.verbose = false
            
            # Should fall back to CPU without crashing
            result = HPRLP.run_lp(A, AL, AU, c, l, u, obj_constant, params)
            
            @test result.output_type == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
            # Verify that use_gpu was set to false after validation
            @test params.use_gpu == false
        end

        @testset "Negative GPU device number fallback" begin
            params = HPRLP.HPRLP_parameters()
            params.use_gpu = true
            params.device_number = -1  # Negative device number
            params.warm_up = false
            params.verbose = false
            
            # Should fall back to CPU without crashing
            result = HPRLP.run_lp(A, AL, AU, c, l, u, obj_constant, params)
            
            @test result.output_type == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
            # Verify that use_gpu was set to false after validation
            @test params.use_gpu == false
        end

        @testset "CPU execution (use_gpu=false)" begin
            params = HPRLP.HPRLP_parameters()
            params.use_gpu = false
            params.warm_up = false
            params.verbose = false
            
            result = HPRLP.run_lp(A, AL, AU, c, l, u, obj_constant, params)
            
            @test result.output_type == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
            @test params.use_gpu == false
        end
    end
end
