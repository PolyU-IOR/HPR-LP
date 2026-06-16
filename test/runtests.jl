using Test
using HPRLP
using JuMP
using SparseArrays
using LinearAlgebra
using Distributed

struct TestIdentityPostsolveState end

function HPRLP.run_custom_postsolve(
    ::TestIdentityPostsolveState,
    x_red::AbstractVector,
    y_red::AbstractVector,
    z_red::AbstractVector;
    presolve_params=nothing,
)
    return collect(x_red), collect(y_red), collect(z_red)
end

function make_test_params(; use_gpu::Bool=false)
    params = HPRLP.HPRLP_parameters()
    params.time_limit = 60
    params.stoptol = 1e-4
    params.use_gpu = use_gpu
    params.warm_up = false
    params.verbose = false
    params.presolve = "NONE"
    return params
end

@testset "HPRLP.jl" begin
    
    @testset "MPS File Solving" begin
        # Test solving an MPS file
        mps_file = joinpath(@__DIR__, "..", "model.mps")
        
        if isfile(mps_file)
            params = make_test_params()
            
            model = HPRLP.build_from_mps(mps_file, false)
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

    @testset "Degenerate Reduced Model Solving" begin
        A = sparse(Int32[], Int32[], Float64[], 0, 0)
        model = HPRLP.build_from_Abc(A, Float64[], Float64[], Float64[], Float64[], Float64[], 7.5)
        params = make_test_params()
        params.use_gpu = true

        result = HPRLP.solve(model, params)
        @test result.status == "OPTIMAL"
        @test result.iter == 0
        @test result.primal_obj == 7.5
        @test isempty(result.x)
        @test isempty(result.y)
        @test isempty(result.z)
        @test result.residuals == 0.0
    end

    @testset "MPS Auto Format Fallback For Long Names" begin
        mps_text = """
NAME          LONGNAMES
ROWS
 N  OBJ
 L  R1
COLUMNS
    COL000001  R1                 1
    COL000002  R1                 2
RHS
    RHS1      R1                 3
ENDATA
"""

        path = tempname() * ".mps"
        try
            open(path, "w") do io
                write(io, mps_text)
            end

            @test_throws ArgumentError HPRLP.MPSReader.read_mps(path; mpsformat=:fixed)

            lp = HPRLP.MPSReader.read_mps(path; mpsformat=:auto)
            A = sparse(lp.arows, lp.acols, lp.avals, lp.nrow, lp.ncol)

            @test lp.ncol == 2
            @test length(lp.avals) == 2
            @test nnz(A) == 2
        finally
            isfile(path) && rm(path)
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
        
        params = make_test_params()
        
        model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
        result = HPRLP.optimize(model, params)
        
        @test result.status == "OPTIMAL"
        @test isapprox(result.primal_obj, -26.4, atol=1e-2)
        @test result.x[1] >= -1e-6  # x1 >= 0
        @test result.x[2] >= -1e-6  # x2 >= 0
    end

    @testset "Solve Without Presolve" begin
        A = sparse([-1.0 -2.0; -3.0 -1.0])
        AL = Vector{Float64}([-10.0, -12.0])
        AU = Vector{Float64}([Inf, Inf])
        c = Vector{Float64}([-3.0, -5.0])
        l = Vector{Float64}([0.0, 0.0])
        u = Vector{Float64}([Inf, Inf])

        params = make_test_params()

        model = HPRLP.build_from_Abc(A, c, AL, AU, l, u)
        result = HPRLP.optimize(model, params)

        @test result.status == "OPTIMAL"
        @test isapprox(result.primal_obj, -26.4, atol=1e-2)
        @test length(result.x) == 2
        @test result.x[1] >= -1e-6
        @test result.x[2] >= -1e-6
        @test result.original_p_feas == result.reduced_p_feas
        @test result.original_d_feas == result.reduced_d_feas
        @test result.original_gap == result.reduced_gap
    end

    @testset "Presolve Failure Falls Back To Original Model" begin
        A = sparse([-1.0 -2.0; -3.0 -1.0])
        AL = Vector{Float64}([-10.0, -12.0])
        AU = Vector{Float64}([Inf, Inf])
        c = Vector{Float64}([-3.0, -5.0])
        l = Vector{Float64}([0.0, 0.0])
        u = Vector{Float64}([Inf, Inf])

        model = HPRLP.build_from_Abc(A, c, AL, AU, l, u)
        params = make_test_params()

        reduced_model, presolve_state = HPRLP.run_presolve_with_fallback("PSLP", model, params) do
            error("simulated presolver failure")
        end

        @test reduced_model === model
        @test presolve_state === nothing

        result = HPRLP.optimize(model, params)
        @test result.status == "OPTIMAL"
        @test isapprox(result.primal_obj, -26.4, atol=1e-2)
    end

    @testset "No-op Presolve Detection" begin
        A = sparse([1.0 0.0; 0.0 2.0])
        c = [3.0, 4.0]
        AL = [1.0, 2.0]
        AU = [1.0, 2.0]
        l = [0.0, 0.0]
        u = [Inf, Inf]

        original = HPRLP.build_from_Abc(A, c, AL, AU, l, u)
        identical = HPRLP.build_from_Abc(copy(A), copy(c), copy(AL), copy(AU), copy(l), copy(u))
        changed_bounds = HPRLP.build_from_Abc(copy(A), copy(c), copy(AL), [1.0, 3.0], copy(l), copy(u))

        @test HPRLP._lp_data_equal(original, identical)
        @test !HPRLP._lp_data_equal(original, changed_bounds)
    end

    @testset "Dataset Time Accounting" begin
        solve_time = HPRLP._dataset_solve_time(12.5, 10.0)
        @test solve_time == 10.0
        @test HPRLP._dataset_total_time(solve_time, 2.0, 3.5) == 15.5
        @test HPRLP._dataset_result_columns() == [
            :name,
            :iter,
            :solve_time,
            :total_time,
            :presolve_time,
            :folding_time,
            :res,
            :primal_obj,
            :status,
            :iter_4,
            :time_4,
            :iter_6,
            :time_6,
            :iter_8,
            :time_8,
        ]
    end

    @testset "Pre-main Log Omits Presolve Timing Details" begin
        A = sparse([1.0 0.0; 0.0 1.0])
        c = [1.0, 2.0]
        AL = [0.0, 0.0]
        AU = [1.0, 1.0]
        l = [0.0, 0.0]
        u = [Inf, Inf]
        model = HPRLP.build_from_Abc(A, c, AL, AU, l, u)
        params = make_test_params()
        params.verbose = true
        params.presolve = "CUSTOM"

        path = tempname()
        try
            open(path, "w") do io
                redirect_stdout(io) do
                    HPRLP.optimize(model, params)
                end
            end
            output = read(path, String)

            @test occursin("Presolve backend: CUSTOM", output)
            @test occursin("Presolved model:", output)
            @test !occursin("Presolve wall time:", output)
            @test !occursin("Presolve core time:", output)
            @test !occursin("Presolve overhead time:", output)
        finally
            rm(path, force=true)
        end
    end

    @testset "LP Folding" begin
        A = sparse([1.0 1.0; 1.0 1.0])
        AL = [1.0, 1.0]
        AU = [1.0, 1.0]
        c = [1.0, 1.0]
        l = [0.0, 0.0]
        u = [1.0, 1.0]

        model = HPRLP.build_from_Abc(A, c, AL, AU, l, u)
        @test_throws ArgumentError HPRLP.run_folding(model; tolerance=1e-8, verbose=false)

        fold_map = HPRLP.FoldingMap(2, 2, [1, 1], [1, 1], [0.5, 0.5], [0.5, 0.5])
        x, y, z = HPRLP.unfold_solution(fold_map, [0.5], [2.0], [4.0])
        @test x == [0.5, 0.5]
        @test y == [1.0, 1.0]
        @test z == [2.0, 2.0]

        params = make_test_params()
        params.folding = "NONE"
        @test_throws ArgumentError HPRLP.run_folding(model; tolerance=1e-8, verbose=false, params=params)

        HPRLP.set_folding_mode!(params, true)
        @test params.folding == "GPU"
        HPRLP.set_folding_mode!(params, false)
        @test params.folding == "NONE"
    end

    @testset "PSLP Worker Isolation Bootstrap" begin
        worker_id = HPRLP.PSLP._spawn_isolated_worker()
        try
            @test remotecall_fetch(HPRLP.PSLP._remote_ping, worker_id) == worker_id
        finally
            rmprocs(worker_id)
        end
    end

    @testset "Original KKT Metrics" begin
        A = sparse([1.0;;])
        AL = [1.0]
        AU = [1.0]
        c = [1.0]
        l = [0.0]
        u = [2.0]

        model = HPRLP.build_from_Abc(A, c, AL, AU, l, u)
        p_obj, d_obj, p_feas, d_feas, gap =
            HPRLP.compute_original_kkt_metrics(model, [1.0], [1.0], [0.0])

        @test isapprox(p_obj, 1.0, atol=1e-10)
        @test isapprox(d_obj, 1.0, atol=1e-10)
        @test isapprox(p_feas, 0.0, atol=1e-10)
        @test isapprox(d_feas, 0.0, atol=1e-10)
        @test isapprox(gap, 0.0, atol=1e-10)
        @test isempty(HPRLP.check_org_recovery_failures(p_feas, d_feas, gap, 1e-10))
    end

    @testset "Original KKT Projection With Infinite Bounds" begin
        A = sparse([1.0;;])
        AL = [-Inf]
        AU = [1.0]
        c = [0.0]
        l = [0.0]
        u = [Inf]

        model = HPRLP.build_from_Abc(A, c, AL, AU, l, u)
        p_obj, d_obj, p_feas, d_feas, gap =
            HPRLP.compute_original_kkt_metrics(model, [0.5], [2.0], [-3.0])

        @test isapprox(p_obj, 0.0, atol=1e-10)
        @test isapprox(d_obj, 0.0, atol=1e-10)
        @test isapprox(p_feas, 0.0, atol=1e-10)
        @test isapprox(d_feas, 0.0, atol=1e-10)
        @test isapprox(gap, 0.0, atol=1e-10)
        @test isempty(HPRLP.check_org_recovery_failures(p_feas, d_feas, gap, 1e-10))
    end

    @testset "Postsolve Primal Row Repair" begin
        A = sparse([1.0 1.0])
        AL = [0.0]
        AU = [0.0]
        c = [0.0, 0.0]
        l = [0.0, -Inf]
        u = [Inf, Inf]

        model = HPRLP.build_from_Abc(A, c, AL, AU, l, u)
        x = [0.0, 1.0e-3]
        changed = HPRLP._repair_primal_row_feasibility!(x, model; tol=1.0e-10)

        @test changed
        @test isapprox(dot(A[1, :], x), 0.0, atol=1.0e-12)
        @test x[1] >= l[1] - 1.0e-12
    end

    @testset "Postsolve And Unfold Share One Postprocess Log" begin
        A = sparse([1.0;;])
        AL = [1.0]
        AU = [1.0]
        c = [1.0]
        l = [0.0]
        u = [2.0]
        model = HPRLP.build_from_Abc(A, c, AL, AU, l, u)
        params = make_test_params()
        params.verbose = true

        postsolve_results = HPRLP.HPRLP_results()
        postsolve_results.status = "OPTIMAL"
        postsolve_results.x = [1.0]
        postsolve_results.y = [0.0]
        postsolve_results.z = [0.0]

        postsolve_log = tempname()
        try
            open(postsolve_log, "w") do io
                redirect_stdout(io) do
                    HPRLP.postsolve_and_validate_original_kkt!(
                        postsolve_results,
                        model,
                        TestIdentityPostsolveState(),
                        params;
                        check_original_kkt=false,
                    )
                end
            end
            output = read(postsolve_log, String)
            @test !occursin("CUSTOM POSTSOLVE", output)
            @test !occursin("Postsolve time:", output)
            @test !occursin("postsolve original KKT", output)
            @test !occursin("Original KKT check time:", output)
        finally
            rm(postsolve_log, force=true)
        end

        unfold_results = HPRLP.HPRLP_results()
        unfold_results.status = "OPTIMAL"
        unfold_results.x = [1.0]
        unfold_results.y = [0.0]
        unfold_results.z = [0.0]
        fold_map = HPRLP.FoldingMap(1, 1, [1], [1], [1.0], [1.0])

        unfold_log = tempname()
        try
            open(unfold_log, "w") do io
                redirect_stdout(io) do
                    HPRLP.unfold_results!(
                        unfold_results,
                        model,
                        fold_map,
                        params;
                        log_kkt=false,
                    )
                end
            end
            output = read(unfold_log, String)
            @test !occursin("Warning: unfolded original KKT check failed", output)
            @test unfold_results.original_d_feas > params.stoptol
        finally
            rm(unfold_log, force=true)
        end

        postprocess_log = tempname()
        try
            open(postprocess_log, "w") do io
                redirect_stdout(io) do
                    HPRLP._print_postprocess_header(
                        true,
                        1.23,
                        0.45,
                    )
                    HPRLP._print_original_kkt_report(
                        false,
                        ["dual recover failed"],
                        1.0e-6,
                        10.0,
                        12.0,
                        1.0e-8,
                        2.0e-3,
                        1.0e-4,
                        0.67,
                    )
                end
            end
            output = read(postprocess_log, String)
            @test occursin("POSTPROCESS", output)
            @test occursin("Postsolve time: 1.23 seconds", output)
            @test occursin("Unfold time: 0.45 seconds", output)
            @test occursin("Warning: original KKT check failed: dual recover failed", output)
            @test occursin("Original KKT check time: 0.67 seconds", output)
            @test !occursin("CUSTOM POSTSOLVE", output)
            @test !occursin("UNFOLD", output)
        finally
            rm(postprocess_log, force=true)
        end

        time_results = HPRLP.HPRLP_results()
        time_results.time = 7.57
        time_results.presolve_time = 0.03
        time_results.folding_time = 0.02

        time_log = tempname()
        try
            open(time_log, "w") do io
                redirect_stdout(io) do
                    HPRLP._print_final_time_summary(time_results)
                end
            end
            output = read(time_log, String)
            @test occursin("Total time: 7.62s", output)
            @test occursin("solve time = 7.57s", output)
            @test occursin("presolve time = 0.03s", output)
            @test occursin("folding time = 0.02s", output)
            @test !occursin("setup time", output)
        finally
            rm(time_log, force=true)
        end
    end

    @testset "Original KKT Error" begin
        @test max(1e-3, 2e-4, 3e-5) == 1e-3
        @test max(2e-6, 5e-5, 4e-6) == 5e-5
    end

    @testset "Original Recovery Failures" begin
        @test isempty(HPRLP.check_org_recovery_failures(1e-6, 2e-6, 3e-6, 1e-4))
        @test HPRLP.check_org_recovery_failures(2e-4, 2e-6, 3e-6, 1e-4) == ["primal recover failed"]
        @test HPRLP.check_org_recovery_failures(2e-6, 2e-4, 3e-6, 1e-4) == ["dual recover failed"]
        @test HPRLP.check_org_recovery_failures(2e-6, 2e-6, 3e-4, 1e-4) == ["dual recover failed"]
        @test HPRLP.check_org_recovery_failures(2e-4, 2e-4, 3e-6, 1e-4) == ["primal recover failed", "dual recover failed"]
    end

    @testset "Structural L1 Rewrite Enabled By Default" begin
        params_default = HPRLP.GPUPresolver.GPUBackend.PresolveParams()
        @test params_default.structural_l1_allow_main_flow_without_tape == true

        config_path = joinpath(@__DIR__, "..", "deps", "GPUPresolver", "config", "default.toml")
        config_default = HPRLP.load_gpu_presolve_params(config_path)
        @test config_default.structural_l1_allow_main_flow_without_tape == true
    end

    @testset "GPU Singleton Column Postsolve Respects Bounds" begin
        if HPRLP.GPUPresolver.CUDA.functional()
            pparams = HPRLP.GPUPresolver.GPUBackend.PresolveParams()
            pparams.max_iters = 1
            pparams.enable_close_bounds = false
            pparams.enable_empty_rows = false
            pparams.enable_singleton_rows = false
            pparams.enable_activity_checks = false
            pparams.enable_primal_propagation = false
            pparams.enable_parallel_rows = false
            pparams.enable_redundant_bounds = false
            pparams.enable_empty_cols = false
            pparams.enable_singleton_cols_eq = true
            pparams.enable_singleton_cols_dual_infer = false
            pparams.enable_doubleton_eq = false
            pparams.enable_dual_fix = false
            pparams.enable_parallel_cols = false
            pparams.row_rule_order = Symbol[]
            pparams.col_rule_order = [:singleton_cols_eq]

            problem = HPRLP.GPUPresolver.LPProblem(HPRLP.GPUPresolver.build_from_Abc(
                sparse([1.0e-4 1.0]),
                [0.0, 0.0],
                [0.0],
                [0.0],
                [0.0, -Inf],
                [Inf, Inf],
                0.0,
            ))
            result = HPRLP.GPUPresolver.run_presolve(
                problem;
                config=HPRLP.GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=pparams,
                ),
            )
            @test result.status == "OK"
            x_org, _, _ = HPRLP.GPUPresolver.run_postsolve(
                result.state,
                [1.0e-3],
                [0.0],
                [0.0],
            )
            @test x_org[1] >= -1.0e-12
            @test isapprox(x_org[1], 0.0, atol=1.0e-12)
            @test HPRLP.GPUPresolver.free_presolve_state!(result.state) === nothing
        else
            @info "CUDA is not functional; skipping GPU singleton column postsolve bound test"
        end
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
        @test params.use_Curtis_Reid_scaling == true
        @test HPRLP.CURTIS_REID_SCALING_ITERS == 20
        @test params.use_Ruiz_scaling == true
        @test params.use_Pock_Chambolle_scaling == true
        @test params.use_bc_scaling == true
        @test params.use_gpu == true
        @test params.device_number == 0
        @test params.warm_up == true
        @test params.print_frequency == -1
        @test params.verbose == true
        @test !(:print_level in fieldnames(HPRLP.HPRLP_parameters))
        @test !HPRLP.MOI.supports(HPRLP.Optimizer(), HPRLP.MOI.RawOptimizerAttribute("print_level"))
        @test params.folding == "NONE"
        @test params.folding_tolerance == 1e-8
        
        # Test parameter modification
        params.stoptol = 1e-6
        @test params.stoptol == 1e-6
        
        params.use_gpu = false
        @test params.use_gpu == false

        params.verbose = false
        @test params.verbose == false

        params.presolve = "NONE"
        @test params.presolve == "NONE"

        params.folding = "GPU"
        @test params.folding == "GPU"
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
        
        params = make_test_params()
        
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
        
        params = make_test_params()
        
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
            params = make_test_params(use_gpu=true)
            params.device_number = 999  # Invalid device number
            
            # Should fall back to CPU without crashing
            model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
            # Verify that use_gpu was set to false after validation
            @test params.use_gpu == false
        end

        @testset "Negative GPU device number fallback" begin
            params = make_test_params(use_gpu=true)
            params.device_number = -1  # Negative device number
            
            # Should fall back to CPU without crashing
            model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
            # Verify that use_gpu was set to false after validation
            @test params.use_gpu == false
        end

        @testset "CPU execution (use_gpu=false)" begin
            params = make_test_params()
            
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
            params = make_test_params()
            
            model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result = HPRLP.optimize(model, params)
            
            @test result.status == "OPTIMAL"
            @test isapprox(result.primal_obj, -26.4, atol=1e-2)
            @test isapprox(result.x[1], 2.8, atol=1e-2)
            @test isapprox(result.x[2], 3.6, atol=1e-2)
        end
        
        @testset "With optimal solution as initial point (both x and y)" begin
            # First solve to get optimal solution
            params_baseline = make_test_params()
            
            model_baseline = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result_baseline = HPRLP.optimize(model_baseline, params_baseline)
            
            # Use the result as initial point
            params = make_test_params()
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
            params_baseline = make_test_params()
            params_baseline.check_iter = 10
            
            model_baseline = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result_baseline = HPRLP.optimize(model_baseline, params_baseline)
            
            # Use only x as initial point
            params = make_test_params()
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
            params_baseline = make_test_params()
            params_baseline.check_iter = 10

            model_baseline = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
            result_baseline = HPRLP.optimize(model_baseline, params_baseline)
            
            # Use only y as initial point
            params = make_test_params()
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
            params = make_test_params()
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
            
            params = make_test_params()
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
