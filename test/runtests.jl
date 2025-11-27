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
            
            model = HPRLP.build_from_mps(mps_file, verbose=false)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
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
        
        model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
        result = HPRLP.optimize(model, params)
        
        @test result.status == "OPTIMAL"
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
        
        model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
        result = HPRLP.optimize(model, params)
        
        # Test that result has all expected fields
        @test isdefined(result, :iter)
        @test isdefined(result, :time)
        @test isdefined(result, :primal_obj)
        @test isdefined(result, :residuals)
        @test isdefined(result, :gap)
        @test isdefined(result, :status)
        @test isdefined(result, :x)
        
        # Test types
        @test result.iter isa Int
        @test result.time isa Float64
        @test result.primal_obj isa Float64
        @test result.x isa Vector{Float64}
        @test result.status isa String
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
        
        model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
        result = HPRLP.optimize(model, params)
        
        @test result.status == "OPTIMAL"
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
            model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
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
            model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
            # Verify that use_gpu was set to false after validation
            @test params.use_gpu == false
        end

        @testset "CPU execution (use_gpu=false)" begin
            params = HPRLP.HPRLP_parameters()
            params.use_gpu = false
            params.warm_up = false
            params.verbose = false
            
            model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
            @test params.use_gpu == false
        end
    end
    
    @testset "Initial Point Functionality" begin
        # Simple problem: min -3x1 - 5x2
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
        
        @testset "No initial point (baseline)" begin
            params = HPRLP.HPRLP_parameters()
            params.time_limit = 60
            params.stoptol = 1e-4
            params.use_gpu = false
            params.warm_up = false
            params.verbose = false
            
            model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
            @test isapprox(result.x[1], 2.8, atol=1e-2)
            @test isapprox(result.x[2], 3.6, atol=1e-2)
        end
        
        @testset "With optimal solution as initial point (both x and y)" begin
            # First solve to get optimal solution
            params_baseline = HPRLP.HPRLP_parameters()
            params_baseline.time_limit = 60
            params_baseline.stoptol = 1e-4
            params_baseline.use_gpu = false
            params_baseline.warm_up = false
            params_baseline.verbose = false
            
            model_baseline = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result_baseline = HPRLP.optimize(model_baseline, params_baseline)
            
            # Use the result as initial point
            params = HPRLP.HPRLP_parameters()
            params.time_limit = 60
            params.stoptol = 1e-4
            params.use_gpu = false
            params.warm_up = false
            params.verbose = false
            params.initial_x = result_baseline.x
            params.initial_y = result_baseline.y
            
            model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
            # Should converge faster with good initial point
            @test result.iter == 0
        end
        
        @testset "With only initial x" begin
            # First solve to get optimal solution
            params_baseline = HPRLP.HPRLP_parameters()
            params_baseline.time_limit = 60
            params_baseline.stoptol = 1e-4
            params_baseline.use_gpu = false
            params_baseline.warm_up = false
            params_baseline.verbose = false
            params_baseline.check_iter = 10
            
            model_baseline = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result_baseline = HPRLP.optimize(model_baseline, params_baseline)
            
            # Use only x as initial point
            params = HPRLP.HPRLP_parameters()
            params.time_limit = 60
            params.stoptol = 1e-4
            params.use_gpu = false
            params.warm_up = false
            params.verbose = false
            params.check_iter = 10
            params.initial_x = result_baseline.x
            # params.initial_y remains nothing
            
            model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
        end
        
        @testset "With only initial y" begin
            # First solve to get optimal solution
            params_baseline = HPRLP.HPRLP_parameters()
            params_baseline.time_limit = 60
            params_baseline.stoptol = 1e-4
            params_baseline.use_gpu = false
            params_baseline.warm_up = false
            params_baseline.verbose = false
            params_baseline.check_iter = 10

            model_baseline = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result_baseline = HPRLP.optimize(model_baseline, params_baseline)
            
            # Use only y as initial point
            params = HPRLP.HPRLP_parameters()
            params.time_limit = 60
            params.stoptol = 1e-4
            params.use_gpu = false
            params.warm_up = false
            params.verbose = false
            params.check_iter = 10
            # params.initial_x remains nothing
            params.initial_y = result_baseline.y
            
            model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
        end
        
        @testset "With feasible but suboptimal initial point" begin
            # Use a feasible but suboptimal starting point
            params = HPRLP.HPRLP_parameters()
            params.time_limit = 60
            params.stoptol = 1e-4
            params.use_gpu = false
            params.warm_up = false
            params.verbose = false
            params.initial_x = [1.0, 1.0]  # Feasible but not optimal
            
            model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
        end
    end
    
    @testset "Auto-Save Functionality" begin
        using HDF5
        
        # Create a simple LP problem
        A = sparse([-1.0 -2.0; -3.0 -1.0])
        AL = Vector{Float64}([-10.0, -12.0])
        AU = Vector{Float64}([Inf, Inf])
        c = Vector{Float64}([-3.0, -5.0])
        l = Vector{Float64}([0.0, 0.0])
        u = Vector{Float64}([Inf, Inf])
        obj_constant = 0.0
        
        # Test file path
        test_h5_file = joinpath(tempdir(), "test_autosave.h5")
        
        @testset "Auto-save enabled" begin
            # Clean up any existing file
            isfile(test_h5_file) && rm(test_h5_file)
            
            params = HPRLP.HPRLP_parameters()
            params.time_limit = 60
            params.stoptol = 1e-4
            params.use_gpu = false
            params.warm_up = false
            params.verbose = false
            params.auto_save = true
            params.save_filename = test_h5_file
            params.print_frequency = 10
            
            model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
            @test isfile(test_h5_file)
            
            # Read and verify HDF5 file contents
            h5open(test_h5_file, "r") do file
                # Check current state
                @test haskey(file, "current/iteration")
                @test haskey(file, "current/x_org")
                @test haskey(file, "current/y_org")
                @test haskey(file, "current/sigma")
                @test haskey(file, "current/primal_obj")
                
                # Check best state  
                @test haskey(file, "best/iteration")
                @test haskey(file, "best/x_org")
                @test haskey(file, "best/y_org")
                @test haskey(file, "best/sigma")
                
                # Verify data
                x_best = read(file, "best/x_org")
                @test length(x_best) == 2
                @test x_best isa Vector{Float64}
            end
            
            rm(test_h5_file)
        end
    end
end
