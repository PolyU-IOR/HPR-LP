using Test
using GPUPresolver
using SparseArrays

@testset "GPUPresolver" begin
    @testset "MPS Reader Fallback" begin
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

            @test_throws ArgumentError GPUPresolver.MPSReader.read_mps(path; mpsformat=:fixed)
            lp = GPUPresolver.MPSReader.read_mps(path; mpsformat=:auto)
            A = sparse(lp.arows, lp.acols, lp.avals, lp.nrow, lp.ncol)
            @test lp.ncol == 2
            @test nnz(A) == 2
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "Model Builders" begin
        A = sparse([-1.0 -2.0; -3.0 -1.0])
        AL = [-10.0, -12.0]
        AU = [Inf, Inf]
        c = [-3.0, -5.0]
        l = [0.0, 0.0]
        u = [Inf, Inf]

        model = GPUPresolver.build_from_Abc(A, c, AL, AU, l, u, 0.0)
        @test size(model.A) == (2, 2)
        @test length(model.c) == 2
        @test model.obj_constant == 0.0
    end

    @testset "KKT Metrics Helpers" begin
        A = sparse([1.0;;])
        AL = [1.0]
        AU = [1.0]
        c = [1.0]
        l = [0.0]
        u = [2.0]
        model = GPUPresolver.build_from_Abc(A, c, AL, AU, l, u)

        p_obj, d_obj, p_feas, d_feas, gap =
            GPUPresolver.compute_original_kkt_metrics(model, [1.0], [1.0], [0.0])
        @test isapprox(p_obj, 1.0, atol=1e-10)
        @test isapprox(d_obj, 1.0, atol=1e-10)
        @test isapprox(p_feas, 0.0, atol=1e-10)
        @test isapprox(d_feas, 0.0, atol=1e-10)
        @test isapprox(gap, 0.0, atol=1e-10)
        @test isempty(GPUPresolver.check_org_recovery_failures(p_feas, d_feas, gap, 1e-10))
    end

    @testset "Presolve Params And Scheduling" begin
                                                @test GPUPresolver.normalize_presolve_scheduler_mode(:fixed) == :fixed
        @test GPUPresolver.normalize_presolve_scheduler_mode(:tiered) == :tiered
        @test GPUPresolver.GPUBackend._qp_stats_is_diagonal(
            GPUPresolver.GPUBackend.QPresolveStats_gpu(0, 0),
        )

        pparams = GPUPresolver.GPUBackend.PresolveParams()
        @test isinf(pparams.max_time)
        @test pparams.doubleton_eq_max_fill_in_proxy == 10
        @test pparams.doubleton_eq_min_selected_ratio == 0.005
        @test !(:primal_propagation_paper_gpu in fieldnames(typeof(pparams)))
        @test !(:primal_propagation_long_row_threshold in fieldnames(typeof(pparams)))
        for max_time in (0.0, 1.0, Inf)
            pparams.max_time = max_time
            @test GPUPresolver.GPUBackend._validate_presolve_rule_orders(pparams) === nothing
        end
        pparams.max_time = -1.0
        @test_throws ErrorException GPUPresolver.GPUBackend._validate_presolve_rule_orders(pparams)
        pparams.max_time = Inf
        @test !GPUPresolver.GPUBackend._presolve_time_exceeded(nothing, pparams)
        @test !GPUPresolver.GPUBackend._presolve_time_exceeded(time() - 100.0, pparams)
        pparams.max_time = 0.0
        @test GPUPresolver.GPUBackend._presolve_time_exceeded(time() - 1.0, pparams)
        pparams.max_time = Inf

        @test pparams.gpu_presolve_scheduler == :fixed
        for scheduler in (:fixed, :tiered)
            pparams.gpu_presolve_scheduler = scheduler
            @test GPUPresolver.GPUBackend._validate_presolve_rule_orders(pparams) === nothing
        end
        pparams.gpu_presolve_scheduler = :bad_scheduler
        @test_throws ErrorException GPUPresolver.GPUBackend._validate_presolve_rule_orders(pparams)
        pparams.gpu_presolve_scheduler = :fixed

        @test GPUPresolver.GPUBackend.TIERED_LIGHT_ROW_RULES == (:singleton_rows,)
        @test GPUPresolver.GPUBackend.TIERED_LIGHT_COL_RULES == (:singleton_cols_dual_infer, :singleton_cols_eq)
        @test GPUPresolver.GPUBackend.TIERED_MEDIUM_ROW_RULES == (:activity_checks, :primal_propagation, :parallel_rows)
        @test GPUPresolver.GPUBackend.TIERED_MEDIUM_COL_RULES == (:parallel_cols, :fme_projection)
        @test GPUPresolver.GPUBackend.TIERED_HEAVY_ROW_RULES == ()
        @test GPUPresolver.GPUBackend.TIERED_HEAVY_COL_RULES == (:doubleton_eq,)

        lp_parallel_spec = GPUPresolver._rule_spec(GPUPresolver.LPPresolveKind(), :parallel_cols)
        @test lp_parallel_spec.public_name == :parallel_cols
        @test lp_parallel_spec.impl_name == :lp_parallel_cols
        @test lp_parallel_spec.execution_stage == GPUPresolver.RULE_STAGE_GPU_LOOP
        @test lp_parallel_spec.supported

        qp_empty_spec = GPUPresolver._rule_spec(GPUPresolver.QPPresolveKind(), :empty_cols)
        @test qp_empty_spec.public_name == :empty_cols
        @test qp_empty_spec.impl_name == :qp_empty_cols_gpu
        @test qp_empty_spec.execution_stage == GPUPresolver.RULE_STAGE_GPU_LOOP
        @test qp_empty_spec.supported

        qp_singleton_dual_spec = GPUPresolver._rule_spec(GPUPresolver.QPPresolveKind(), :singleton_cols_dual_infer)
        @test qp_singleton_dual_spec.public_name == :singleton_cols_dual_infer
        @test qp_singleton_dual_spec.impl_name == :qp_singleton_cols_dual_infer_gpu
        @test qp_singleton_dual_spec.execution_stage == GPUPresolver.RULE_STAGE_GPU_LOOP
        @test qp_singleton_dual_spec.supported

        qp_singleton_eq_spec = GPUPresolver._rule_spec(GPUPresolver.QPPresolveKind(), :singleton_cols_eq)
        @test qp_singleton_eq_spec.public_name == :singleton_cols_eq
        @test qp_singleton_eq_spec.impl_name == :qp_singleton_cols_eq_gpu
        @test qp_singleton_eq_spec.execution_stage == GPUPresolver.RULE_STAGE_GPU_LOOP
        @test qp_singleton_eq_spec.supported

        qp_parallel_spec = GPUPresolver._rule_spec(GPUPresolver.QPPresolveKind(), :parallel_cols)
        @test qp_parallel_spec.public_name == :parallel_cols
        @test qp_parallel_spec.impl_name == :qp_parallel_cols_gpu
        @test qp_parallel_spec.execution_stage == GPUPresolver.RULE_STAGE_GPU_LOOP
        @test qp_parallel_spec.supported

        @test !occursin("paper_gpu", read(GPUPresolver.default_config_path(), String))

        cfg_path = tempname() * ".toml"
        try
            write(cfg_path, """
[doubleton]
max_fill_in_proxy = 3
""")
            setup = GPUPresolver.load_presolve_setup(cfg_path)
            @test setup.presolve_params.doubleton_eq_max_fill_in_proxy == 3
        finally
            isfile(cfg_path) && rm(cfg_path)
        end

        cfg_path = tempname() * ".toml"
        try
            write(cfg_path, """
[doubleton]
scan = true
min_selected_per_batch = 128
min_selected_ratio = 0.05
max_batch_rounds = 64
max_time = 0.5
""")
            setup = GPUPresolver.load_presolve_setup(cfg_path)
            @test setup.presolve_params.doubleton_eq_scan == true
            @test setup.presolve_params.doubleton_eq_min_selected_per_batch == 128
            @test setup.presolve_params.doubleton_eq_min_selected_ratio == 0.05
            @test setup.presolve_params.doubleton_eq_max_batch_rounds == 64
            @test setup.presolve_params.doubleton_eq_max_time == 0.5
        finally
            isfile(cfg_path) && rm(cfg_path)
        end

        cfg_path = tempname() * ".toml"
        try
            write(cfg_path, """
[rules]
redundant_bounds = false
""")
            setup = GPUPresolver.load_presolve_setup(cfg_path)
            @test setup.presolve_params.enable_redundant_bounds == false
        finally
            isfile(cfg_path) && rm(cfg_path)
        end

        qp_doubleton_spec = GPUPresolver._rule_spec(GPUPresolver.QPPresolveKind(), :doubleton_eq)
        @test qp_doubleton_spec.public_name == :doubleton_eq
        @test qp_doubleton_spec.impl_name == :qp_doubleton_eq_gpu
        @test qp_doubleton_spec.execution_stage == GPUPresolver.RULE_STAGE_GPU_LOOP
        @test qp_doubleton_spec.supported

        qp_linear_eq_agg_spec = GPUPresolver._rule_spec(GPUPresolver.QPPresolveKind(), :linear_eq_agg)
        @test qp_linear_eq_agg_spec.public_name == :linear_eq_agg
        @test qp_linear_eq_agg_spec.impl_name == :qp_linear_eq_agg_gpu
        @test qp_linear_eq_agg_spec.execution_stage == GPUPresolver.RULE_STAGE_GPU_LOOP
        @test qp_linear_eq_agg_spec.supported

        qp_dual_fix_spec = GPUPresolver._rule_spec(GPUPresolver.QPPresolveKind(), :dual_fix)
        @test qp_dual_fix_spec.public_name == :dual_fix
        @test qp_dual_fix_spec.impl_name == :qp_dual_fix_gpu
        @test qp_dual_fix_spec.execution_stage == GPUPresolver.RULE_STAGE_GPU_LOOP
        @test qp_dual_fix_spec.supported

        qp_plan_params = GPUPresolver.GPUBackend.PresolveParams()
        qp_plan_params.enable_empty_cols = true
        qp_plan_params.enable_singleton_cols_dual_infer = false
        qp_plan_params.enable_parallel_cols = true
        qp_plan_params.enable_doubleton_eq = false
        qp_plan_params.enable_linear_eq_agg = false
        qp_plan_params.enable_dual_fix = true
        qp_plan_params.row_rule_order = [:empty_rows, :parallel_rows]
        qp_plan_params.col_rule_order = [:close_bounds, :empty_cols, :singleton_cols_dual_infer, :linear_eq_agg, :dual_fix, :parallel_cols]
        GPUPresolver._resolve_qp_native_rule_plan!(qp_plan_params)
        @test qp_plan_params.row_rule_order == [:empty_rows, :parallel_rows]
        @test qp_plan_params.col_rule_order == [:close_bounds, :empty_cols, :dual_fix, :parallel_cols]
        @test qp_plan_params.enable_empty_cols
        @test !qp_plan_params.enable_singleton_cols_dual_infer
        @test qp_plan_params.enable_dual_fix
        @test qp_plan_params.enable_parallel_cols

        withenv(
            "GPUPRESOLVER_PRESOLVE_RULE_PROFILE" => nothing,
        ) do
            @test !GPUPresolver.GPUBackend._gpu_presolver_env_enabled(
                "GPUPRESOLVER_PRESOLVE_RULE_PROFILE",
            )
        end
        withenv(
            "GPUPRESOLVER_PRESOLVE_RULE_PROFILE" => "1",
        ) do
            @test GPUPresolver.GPUBackend._gpu_presolver_env_enabled(
                "GPUPRESOLVER_PRESOLVE_RULE_PROFILE",
            )
        end

        @testset "GPU presolve warmup mode" begin
            withenv("GPUPRESOLVER_WARMUP_MODE" => nothing) do
                @test GPUPresolver._gpu_presolve_warmup_mode() == :actual
            end
            withenv("GPUPRESOLVER_WARMUP_MODE" => "tiny") do
                @test GPUPresolver._gpu_presolve_warmup_mode() == :tiny
            end
            withenv("GPUPRESOLVER_WARMUP_MODE" => "actual") do
                @test GPUPresolver._gpu_presolve_warmup_mode() == :actual
            end
            withenv("GPUPRESOLVER_WARMUP_MODE" => "off") do
                @test GPUPresolver._gpu_presolve_warmup_mode() == :off
            end
            withenv("GPUPRESOLVER_WARMUP_MODE" => "bad") do
                @test GPUPresolver._gpu_presolve_warmup_mode() == :actual
            end

            tiny_model = GPUPresolver._gpu_presolve_warmup_model(:tiny)
            @test size(tiny_model.A) == (6, 8)

            withenv("GPUPRESOLVER_REPRESENTATIVE_WARMUP_BLOCKS" => "32") do
                rep_model = GPUPresolver._gpu_presolve_warmup_model(:representative)
                @test size(rep_model.A, 1) == 128
                @test size(rep_model.A, 2) == 128
                @test nnz(rep_model.A) > nnz(tiny_model.A)
                @test any(isinf, rep_model.AL)
                @test all(isfinite, rep_model.u)
            end
            withenv("GPUPRESOLVER_COVERAGE_WARMUP_BLOCKS" => "32") do
                coverage_model = GPUPresolver._gpu_presolve_warmup_model(:coverage)
                coverage_models = GPUPresolver._gpu_presolve_warmup_models(:coverage)
                @test length(coverage_models) >= 3
                @test size(coverage_model.A) == size(first(coverage_models).A)
                @test nnz(coverage_model.A) == nnz(first(coverage_models).A)
                @test size(coverage_model.A, 1) >= 128
                @test size(coverage_model.A, 2) >= 128
                @test nnz(coverage_model.A) > nnz(tiny_model.A)
                @test any(isinf, coverage_model.AL)
                @test any(coverage_model.c .< 0.0)
                @test all(isfinite, coverage_model.u)

                suite_model = coverage_models[2]
                @test any(diff(suite_model.A.colptr) .== 0)
                @test any(diff(suite_model.A.rowval[sortperm(suite_model.A.rowval)]) .== 0)
                @test maximum(diff(SparseArrays.sparse(suite_model.A').colptr)) >= 64

                parallel_col_model = coverage_models[3]
                @test size(parallel_col_model.A) == (2, 3)
                @test parallel_col_model.A[:, 1] == parallel_col_model.A[:, 2]
                @test parallel_col_model.c[1] == parallel_col_model.c[2]
            end
        end

        config_path = tempname() * ".toml"
        try
            open(config_path, "w") do io
                write(io, """
[runtime]
backend = "GPU"
device_number = 2
verbose = false
scheduler_mode = "tiered"

[problem]
type = "LP"

[limits]
max_presolve_iters = 7
max_presolve_time = 12.5

[tolerances]
feasibility = 1e-5
bound = 2e-6
zero = 1e-9

[rules]
doubleton_eq = false
linear_eq_agg = true

[qp]
linear_eq_agg_max_support = 9
singleton_max_q_fill_ratio = "Inf"

""")
            end

            setup = GPUPresolver.load_presolve_setup(config_path)
            @test setup.problem_type == "LP"
            @test setup.config.backend == "GPU"
            @test setup.config.device_number == 2
            @test !setup.config.verbose
            @test setup.presolve_params.max_iters == 7
            @test isapprox(setup.presolve_params.max_time, 12.5)
            @test isapprox(setup.presolve_params.feasibility_tol, 1e-5)
            @test isapprox(setup.presolve_params.bound_tol, 2e-6)
            @test isapprox(setup.presolve_params.zero_tol, 1e-9)
            @test !setup.presolve_params.enable_doubleton_eq
            @test setup.presolve_params.enable_linear_eq_agg
            @test setup.presolve_params.qp_linear_eq_agg_max_support == 9
            @test isinf(setup.presolve_params.qp_singleton_max_q_fill_ratio)
            @test setup.presolve_params.gpu_presolve_scheduler == :tiered
        finally
            isfile(config_path) && rm(config_path)
        end
    end


    @testset "Unified Presolve API" begin
        A = sparse([-1.0 -2.0; -3.0 -1.0])
        AL = Vector{Float64}([-10.0, -12.0])
        AU = Vector{Float64}([Inf, Inf])
        c = Vector{Float64}([-3.0, -5.0])
        l = Vector{Float64}([0.0, 0.0])
        u = Vector{Float64}([Inf, Inf])
        model = GPUPresolver.build_from_Abc(A, c, AL, AU, l, u, 0.0)

        result_none = GPUPresolver.run_presolve(
            model;
            config=GPUPresolver.PresolveConfig(backend="NONE", verbose=false),
        )
        @test result_none.status == "SKIPPED"
        @test result_none.backend == "NONE"
        @test result_none.reduced_problem isa GPUPresolver.LPProblem
        @test size(result_none.reduced_problem.model.A) == size(model.A)
        @test isnothing(result_none.state)
        @test GPUPresolver.free_presolve_state!(result_none.state) === nothing

        Q = sparse([2.0 1.0; 1.0 4.0])
        A_qp = sparse([1.0 2.0])
        c_qp = [3.0, 5.0]
        AL_qp = [2.0]
        AU_qp = [6.0]
        l_qp = [1.0, 0.0]
        u_qp = [1.0, 10.0]
        qp = GPUPresolver.QPProblem(Q, A_qp, c_qp, AL_qp, AU_qp, l_qp, u_qp, 7.0)

        native_pparams = GPUPresolver.GPUBackend.PresolveParams()
        native_pparams.max_iters = 1
        native_pparams.enable_close_bounds = false
        native_pparams.enable_empty_rows = false
        native_pparams.enable_singleton_rows = false
        native_pparams.enable_activity_checks = false
        native_pparams.enable_primal_propagation = false
        native_pparams.enable_parallel_rows = false
        native_pparams.enable_redundant_bounds = false
        native_pparams.enable_empty_cols = false
        native_pparams.enable_singleton_cols_eq = false
        native_pparams.enable_singleton_cols_dual_infer = false
        native_pparams.enable_doubleton_eq = false
        native_pparams.enable_dual_fix = false
        native_pparams.enable_parallel_cols = false
        native_pparams.row_rule_order = Symbol[]
        native_pparams.col_rule_order = Symbol[]

        if GPUPresolver.CUDA.functional()
            delta_direct = GPUPresolver.GPUBackend._qp_build_delta_csr_direct(
                GPUPresolver.CUDA.CuArray(Int32[2, 1, 3, 2]),
                GPUPresolver.CUDA.CuArray(Int32[1, 3, 2, 3]),
                GPUPresolver.CUDA.CuArray(Float64[4.0, 2.0, -1.0, 5.0]),
                (3, 3),
            )
            delta_direct_cpu = SparseMatrixCSC(delta_direct)
            @test nnz(delta_direct_cpu) == 4
            @test isapprox(delta_direct_cpu[1, 3], 2.0, atol=1e-12)
            @test isapprox(delta_direct_cpu[2, 1], 4.0, atol=1e-12)
            @test isapprox(delta_direct_cpu[2, 3], 5.0, atol=1e-12)
            @test isapprox(delta_direct_cpu[3, 2], -1.0, atol=1e-12)

            qp_result = GPUPresolver.run_presolve(
                qp;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=native_pparams,
                ),
            )
            @test qp_result.status == "OK"
            @test qp_result.reduced_problem isa GPUPresolver.QPProblem
            qp_red = qp_result.reduced_problem
            @test size(qp_red.A) == (1, 2)
            @test size(qp_red.Q) == (2, 2)
            @test isapprox(qp_red.A[1, 1], 1.0, atol=1e-12)
            @test isapprox(qp_red.A[1, 2], 2.0, atol=1e-12)
            @test isapprox(qp_red.AL[1], 2.0, atol=1e-12)
            @test isapprox(qp_red.AU[1], 6.0, atol=1e-12)
            @test isapprox(qp_red.c[1], 3.0, atol=1e-12)
            @test isapprox(qp_red.c[2], 5.0, atol=1e-12)
            @test isapprox(qp_red.Q[1, 2], 1.0, atol=1e-12)
            @test isapprox(qp_red.obj_constant, 7.0, atol=1e-12)

            @test !isnothing(qp_result.state)
            x_org, y_org, z_org = GPUPresolver.run_postsolve(
                qp_result.state,
                [1.0, 2.5],
                [0.0],
                [0.0, 0.0],
            )
            @test length(x_org) == 2
            @test length(y_org) == 1
            @test length(z_org) == 2
            @test isapprox(x_org[1], 1.0, atol=1e-12)
            @test isapprox(x_org[2], 2.5, atol=1e-12)
            @test GPUPresolver.free_presolve_state!(qp_result.state) === nothing

            # Non-symmetric Q should be interpreted via symmetric part 0.5*(Q + Q').
            Q_nonsym = sparse([2.0 3.0; 1.0 4.0])
            qp_nonsym = GPUPresolver.QPProblem(Q_nonsym, A_qp, c_qp, AL_qp, AU_qp, l_qp, u_qp, 7.0)
            qp_nonsym_result = GPUPresolver.run_presolve(
                qp_nonsym;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=native_pparams,
                ),
            )
            @test qp_nonsym_result.status == "OK"
            qp_nonsym_red = qp_nonsym_result.reduced_problem
            @test isapprox(qp_nonsym_red.Q[1, 2], 2.0, atol=1e-12)
            @test isapprox(qp_nonsym_red.Q[2, 1], 2.0, atol=1e-12)
            @test isapprox(qp_nonsym_red.c[2], 5.0, atol=1e-12)
            @test isapprox(qp_nonsym_red.obj_constant, 7.0, atol=1e-12)
            @test GPUPresolver.free_presolve_state!(qp_nonsym_result.state) === nothing

            doubleton_pparams = GPUPresolver.GPUBackend.PresolveParams()
            doubleton_pparams.max_iters = 1
            doubleton_pparams.enable_close_bounds = false
            doubleton_pparams.enable_empty_rows = false
            doubleton_pparams.enable_singleton_rows = false
            doubleton_pparams.enable_activity_checks = false
            doubleton_pparams.enable_primal_propagation = false
            doubleton_pparams.enable_parallel_rows = false
            doubleton_pparams.enable_redundant_bounds = false
            doubleton_pparams.enable_empty_cols = false
            doubleton_pparams.enable_singleton_cols_eq = false
            doubleton_pparams.enable_singleton_cols_dual_infer = false
            doubleton_pparams.enable_doubleton_eq = true
            doubleton_pparams.enable_dual_fix = false
            doubleton_pparams.enable_parallel_cols = false
            doubleton_pparams.row_rule_order = Symbol[]
            doubleton_pparams.col_rule_order = [:doubleton_eq]
            Q_doubleton_sep = sparse([2.0 0.0; 0.0 4.0])
            qp_doubleton = GPUPresolver.QPProblem(
                Q_doubleton_sep,
                sparse([1.0 1.0]),
                c_qp,
                [2.0],
                [2.0],
                [0.0, 0.0],
                [10.0, 10.0],
                7.0,
            )
            qp_doubleton_result = GPUPresolver.run_presolve(
                qp_doubleton;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=doubleton_pparams,
                ),
            )
            @test qp_doubleton_result.status == "OK"
            qp_doubleton_red = qp_doubleton_result.reduced_problem
            @test size(qp_doubleton_red.A) == (0, 1)
            @test size(qp_doubleton_red.Q) == (1, 1)
            @test isapprox(qp_doubleton_red.Q[1, 1], 6.0, atol=1e-12)
            @test isapprox(qp_doubleton_red.c[1], -10.0, atol=1e-12)
            @test isapprox(qp_doubleton_red.obj_constant, 25.0, atol=1e-12)
            @test isapprox(qp_doubleton_red.l[1], 0.0, atol=1e-12)
            @test isapprox(qp_doubleton_red.u[1], 2.0, atol=1e-12)
            x_doubleton_org, _, _ = GPUPresolver.run_postsolve(
                qp_doubleton_result.state,
                [0.5],
                Float64[],
                [0.0],
            )
            @test length(x_doubleton_org) == 2
            @test isapprox(x_doubleton_org[1], 0.5, atol=1e-12)
            @test isapprox(x_doubleton_org[2], 1.5, atol=1e-12)
            @test GPUPresolver.free_presolve_state!(qp_doubleton_result.state) === nothing

            # QP public presolve is now a thin wrapper over the QP-native
            # scheduler.  Rules that have not been ported to QP-native are
            # ignored instead of being replayed in the API layer.
            singleton_eq_pparams = GPUPresolver.GPUBackend.PresolveParams()
            singleton_eq_pparams.max_iters = 1
            singleton_eq_pparams.enable_close_bounds = false
            singleton_eq_pparams.enable_empty_rows = false
            singleton_eq_pparams.enable_singleton_rows = false
            singleton_eq_pparams.enable_activity_checks = false
            singleton_eq_pparams.enable_primal_propagation = false
            singleton_eq_pparams.enable_parallel_rows = false
            singleton_eq_pparams.enable_redundant_bounds = false
            singleton_eq_pparams.enable_empty_cols = false
            singleton_eq_pparams.enable_singleton_cols_eq = true
            singleton_eq_pparams.enable_singleton_cols_dual_infer = false
            singleton_eq_pparams.enable_doubleton_eq = false
            singleton_eq_pparams.enable_dual_fix = false
            singleton_eq_pparams.enable_parallel_cols = false
            singleton_eq_pparams.row_rule_order = Symbol[]
            singleton_eq_pparams.col_rule_order = [:singleton_cols_eq]
            qp_eq_singleton = GPUPresolver.QPProblem(
                sparse([
                    2.0 0.0 0.0
                    0.0 4.0 0.0
                    0.0 0.0 5.0
                ]),
                sparse([1.0 2.0 -1.0]),
                [7.0, 11.0, 13.0],
                [4.0],
                [4.0],
                [-Inf, 0.0, 0.0],
                [Inf, 10.0, 10.0],
                17.0,
            )
            qp_eq_singleton_result = GPUPresolver.run_presolve(
                qp_eq_singleton;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=singleton_eq_pparams,
                ),
            )
            @test qp_eq_singleton_result.status == "OK"
            qp_eq_singleton_red = qp_eq_singleton_result.reduced_problem
            @test size(qp_eq_singleton_red.A) == (0, 2)
            @test size(qp_eq_singleton_red.Q) == (2, 2)
            @test isapprox(qp_eq_singleton_red.Q[1, 1], 12.0, atol=1e-12)
            @test isapprox(qp_eq_singleton_red.Q[1, 2], -4.0, atol=1e-12)
            @test isapprox(qp_eq_singleton_red.Q[2, 1], -4.0, atol=1e-12)
            @test isapprox(qp_eq_singleton_red.Q[2, 2], 7.0, atol=1e-12)
            @test isapprox(qp_eq_singleton_red.c[1], -19.0, atol=1e-12)
            @test isapprox(qp_eq_singleton_red.c[2], 28.0, atol=1e-12)
            @test isapprox(qp_eq_singleton_red.obj_constant, 61.0, atol=1e-12)
            x_eq_singleton_org, _, _ = GPUPresolver.run_postsolve(
                qp_eq_singleton_result.state,
                [1.0, 2.0],
                Float64[],
                [0.0, 0.0],
            )
            @test length(x_eq_singleton_org) == 3
            @test isapprox(x_eq_singleton_org[1], 4.0, atol=1e-12)
            @test isapprox(x_eq_singleton_org[2], 1.0, atol=1e-12)
            @test isapprox(x_eq_singleton_org[3], 2.0, atol=1e-12)
            @test GPUPresolver.free_presolve_state!(qp_eq_singleton_result.state) === nothing

            singleton_eq_qdiag_zero_pparams = deepcopy(singleton_eq_pparams)
            singleton_eq_qdiag_zero_pparams.qp_singleton_cols_eq_require_qdiag_zero = true
            qp_eq_singleton_qdiag_blocked_result = GPUPresolver.run_presolve(
                qp_eq_singleton;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=singleton_eq_qdiag_zero_pparams,
                ),
            )
            @test qp_eq_singleton_qdiag_blocked_result.status == "OK"
            @test size(qp_eq_singleton_qdiag_blocked_result.reduced_problem.A) == (1, 3)
            @test size(qp_eq_singleton_qdiag_blocked_result.reduced_problem.Q) == (3, 3)
            @test GPUPresolver.free_presolve_state!(qp_eq_singleton_qdiag_blocked_result.state) === nothing

            singleton_eq_fill_guard_pparams = deepcopy(singleton_eq_pparams)
            singleton_eq_fill_guard_pparams.qp_singleton_max_q_fill_abs = 1
            qp_eq_singleton_fill_guard_result = GPUPresolver.run_presolve(
                qp_eq_singleton;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=singleton_eq_fill_guard_pparams,
                ),
            )
            @test qp_eq_singleton_fill_guard_result.status == "OK"
            @test size(qp_eq_singleton_fill_guard_result.reduced_problem.A) == (1, 3)
            @test size(qp_eq_singleton_fill_guard_result.reduced_problem.Q) == (3, 3)
            @test GPUPresolver.free_presolve_state!(qp_eq_singleton_fill_guard_result.state) === nothing

            singleton_eq_fill_ratio_pparams = deepcopy(singleton_eq_pparams)
            singleton_eq_fill_ratio_pparams.qp_singleton_max_q_fill_ratio = 0.5
            qp_eq_singleton_fill_ratio_result = GPUPresolver.run_presolve(
                qp_eq_singleton;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=singleton_eq_fill_ratio_pparams,
                ),
            )
            @test qp_eq_singleton_fill_ratio_result.status == "OK"
            @test size(qp_eq_singleton_fill_ratio_result.reduced_problem.A) == (1, 3)
            @test size(qp_eq_singleton_fill_ratio_result.reduced_problem.Q) == (3, 3)
            @test GPUPresolver.free_presolve_state!(qp_eq_singleton_fill_ratio_result.state) === nothing

            dual_infer_pparams = GPUPresolver.GPUBackend.PresolveParams()
            dual_infer_pparams.max_iters = 1
            dual_infer_pparams.enable_close_bounds = false
            dual_infer_pparams.enable_empty_rows = false
            dual_infer_pparams.enable_singleton_rows = false
            dual_infer_pparams.enable_activity_checks = false
            dual_infer_pparams.enable_primal_propagation = false
            dual_infer_pparams.enable_parallel_rows = false
            dual_infer_pparams.enable_redundant_bounds = false
            dual_infer_pparams.enable_empty_cols = false
            dual_infer_pparams.enable_singleton_cols_eq = false
            dual_infer_pparams.enable_singleton_cols_dual_infer = true
            dual_infer_pparams.enable_doubleton_eq = false
            dual_infer_pparams.enable_dual_fix = false
            dual_infer_pparams.enable_parallel_cols = false
            dual_infer_pparams.row_rule_order = Symbol[]
            dual_infer_pparams.col_rule_order = [:singleton_cols_dual_infer]
            qp_ineq_singleton = GPUPresolver.QPProblem(
                spdiagm(0 => [2.0, 4.0]),
                sparse([1.0 1.0]),
                [10.0, 0.0],
                [4.0],
                [6.0],
                [0.0, 0.0],
                [10.0, 1.0],
                0.0,
            )
            qp_ineq_singleton_result = GPUPresolver.run_presolve(
                qp_ineq_singleton;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=dual_infer_pparams,
                ),
            )
            @test qp_ineq_singleton_result.status == "OK"
            @test size(qp_ineq_singleton_result.reduced_problem.A) == (1, 2)
            @test size(qp_ineq_singleton_result.reduced_problem.Q) == (2, 2)
            @test isapprox(qp_ineq_singleton_result.reduced_problem.AL[1], 4.0, atol=1e-12)
            @test isapprox(qp_ineq_singleton_result.reduced_problem.AU[1], 4.0, atol=1e-12)
            x_ineq_singleton_org, y_ineq_singleton_org, z_ineq_singleton_org = GPUPresolver.run_postsolve(
                qp_ineq_singleton_result.state,
                [3.0, 1.0],
                [0.0],
                [0.0, 0.0],
            )
            @test isapprox(x_ineq_singleton_org[1], 3.0, atol=1e-12)
            @test isapprox(x_ineq_singleton_org[2], 1.0, atol=1e-12)
            @test length(y_ineq_singleton_org) == 1
            @test length(z_ineq_singleton_org) == 2
            @test GPUPresolver.free_presolve_state!(qp_ineq_singleton_result.state) === nothing

            qp_ineq_singleton_coupled = GPUPresolver.QPProblem(
                sparse([2.0 1.0; 1.0 4.0]),
                sparse([1.0 1.0]),
                [10.0, 0.0],
                [4.0],
                [6.0],
                [0.0, 0.0],
                [10.0, 1.0],
                0.0,
            )
            qp_ineq_singleton_coupled_result = GPUPresolver.run_presolve(
                qp_ineq_singleton_coupled;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=dual_infer_pparams,
                ),
            )
            @test qp_ineq_singleton_coupled_result.status == "OK"
            @test isapprox(qp_ineq_singleton_coupled_result.reduced_problem.AL[1], 4.0, atol=1e-12)
            @test isapprox(qp_ineq_singleton_coupled_result.reduced_problem.AU[1], 6.0, atol=1e-12)
            @test GPUPresolver.free_presolve_state!(qp_ineq_singleton_coupled_result.state) === nothing

            qp_ineq_singleton_boundary = GPUPresolver.QPProblem(
                spdiagm(0 => [2.0, 4.0]),
                sparse([1.0 1.0]),
                [10.0, 0.0],
                [4.0],
                [6.0],
                [3.0, 0.0],
                [6.0, 1.0],
                0.0,
            )
            qp_ineq_singleton_boundary_result = GPUPresolver.run_presolve(
                qp_ineq_singleton_boundary;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=dual_infer_pparams,
                ),
            )
            @test qp_ineq_singleton_boundary_result.status == "OK"
            @test isapprox(qp_ineq_singleton_boundary_result.reduced_problem.AL[1], 4.0, atol=1e-12)
            @test isapprox(qp_ineq_singleton_boundary_result.reduced_problem.AU[1], 4.0, atol=1e-12)
            @test GPUPresolver.free_presolve_state!(qp_ineq_singleton_boundary_result.state) === nothing

            empty_pparams = GPUPresolver.GPUBackend.PresolveParams()
            empty_pparams.max_iters = 1
            empty_pparams.enable_close_bounds = false
            empty_pparams.enable_empty_rows = false
            empty_pparams.enable_singleton_rows = false
            empty_pparams.enable_activity_checks = false
            empty_pparams.enable_primal_propagation = false
            empty_pparams.enable_parallel_rows = false
            empty_pparams.enable_redundant_bounds = false
            empty_pparams.enable_empty_cols = true
            empty_pparams.enable_singleton_cols_eq = false
            empty_pparams.enable_singleton_cols_dual_infer = false
            empty_pparams.enable_doubleton_eq = false
            empty_pparams.enable_dual_fix = false
            empty_pparams.enable_parallel_cols = false
            empty_pparams.row_rule_order = Symbol[]
            empty_pparams.col_rule_order = [:empty_cols]

            # Separable empty column: x2 is independent and has minimizer 2.
            A_empty = sparse([1.0 0.0])
            Q_empty = spdiagm(0 => [2.0, 4.0])
            qp_empty = GPUPresolver.QPProblem(
                Q_empty,
                A_empty,
                [0.0, -8.0],
                [1.0],
                [1.0],
                [0.0, 0.0],
                [10.0, 10.0],
                5.0,
            )
            qp_empty_result = GPUPresolver.run_presolve(
                qp_empty;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=empty_pparams,
                ),
            )
            @test qp_empty_result.status == "OK"
            qp_empty_red = qp_empty_result.reduced_problem
            @test size(qp_empty_red.A) == (1, 1)
            @test size(qp_empty_red.Q) == (1, 1)
            @test isapprox(qp_empty_red.Q[1, 1], 2.0, atol=1e-12)
            @test isapprox(qp_empty_red.c[1], 0.0, atol=1e-12)
            @test isapprox(qp_empty_red.obj_constant, -3.0, atol=1e-12)
            x_empty_org, _, _ = GPUPresolver.run_postsolve(
                qp_empty_result.state,
                [3.0],
                [0.0],
                [0.0],
            )
            @test isapprox(x_empty_org[1], 3.0, atol=1e-12)
            @test isapprox(x_empty_org[2], 2.0, atol=1e-12)
            @test GPUPresolver.free_presolve_state!(qp_empty_result.state) === nothing

            # Coupled empty column must be skipped because Q[1,2] is live.
            qp_empty_coupled = GPUPresolver.QPProblem(
                sparse([2.0 1.0; 1.0 4.0]),
                A_empty,
                [0.0, -8.0],
                [1.0],
                [1.0],
                [0.0, 0.0],
                [10.0, 10.0],
                5.0,
            )
            qp_empty_coupled_result = GPUPresolver.run_presolve(
                qp_empty_coupled;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=empty_pparams,
                ),
            )
            @test qp_empty_coupled_result.status == "OK"
            @test size(qp_empty_coupled_result.reduced_problem.A) == (1, 2)
            @test size(qp_empty_coupled_result.reduced_problem.Q) == (2, 2)
            @test GPUPresolver.free_presolve_state!(qp_empty_coupled_result.state) === nothing

            # Concave separable empty column with finite bounds is fixed to the
            # better endpoint.
            qp_empty_concave = GPUPresolver.QPProblem(
                spdiagm(0 => [2.0, -2.0]),
                A_empty,
                [0.0, 0.0],
                [1.0],
                [1.0],
                [0.0, -1.0],
                [10.0, 3.0],
                5.0,
            )
            qp_empty_concave_result = GPUPresolver.run_presolve(
                qp_empty_concave;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=empty_pparams,
                ),
            )
            @test qp_empty_concave_result.status == "OK"
            @test isapprox(qp_empty_concave_result.reduced_problem.obj_constant, -4.0, atol=1e-12)
            x_empty_concave_org, _, _ = GPUPresolver.run_postsolve(
                qp_empty_concave_result.state,
                [3.0],
                [0.0],
                [0.0],
            )
            @test isapprox(x_empty_concave_org[2], 3.0, atol=1e-12)
            @test GPUPresolver.free_presolve_state!(qp_empty_concave_result.state) === nothing

            qp_empty_unbounded = GPUPresolver.QPProblem(
                spdiagm(0 => [2.0, 0.0]),
                A_empty,
                [0.0, -1.0],
                [1.0],
                [1.0],
                [0.0, 0.0],
                [10.0, Inf],
                0.0,
            )
            @test_throws ErrorException GPUPresolver.run_presolve(
                qp_empty_unbounded;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=empty_pparams,
                ),
            )

            qp_native_gpu = GPUPresolver.setup_gpu_qp_model(
                Q_empty,
                A_empty,
                [0.0, -8.0],
                [1.0],
                [1.0],
                [0.0, 0.0],
                [10.0, 10.0],
                5.0;
                verbose=false,
            )
            qp_native_red, qp_native_rec = GPUPresolver.GPUBackend.presolve_gpu(
                qp_native_gpu;
                presolve_params=empty_pparams,
                verbose=false,
            )
            Q_native_cpu, A_native_cpu, c_native_cpu, AL_native_cpu, AU_native_cpu, l_native_cpu, u_native_cpu, c0_native_cpu =
                GPUPresolver.copy_qp_model_to_cpu(qp_native_red)
            @test size(A_native_cpu) == (1, 1)
            @test size(Q_native_cpu) == (1, 1)
            @test isapprox(Q_native_cpu[1, 1], 2.0, atol=1e-12)
            @test isapprox(c_native_cpu[1], 0.0, atol=1e-12)
            @test isapprox(AL_native_cpu[1], 1.0, atol=1e-12)
            @test isapprox(AU_native_cpu[1], 1.0, atol=1e-12)
            @test isapprox(l_native_cpu[1], 0.0, atol=1e-12)
            @test isapprox(u_native_cpu[1], 10.0, atol=1e-12)
            @test isapprox(c0_native_cpu, -3.0, atol=1e-12)
            @test Int(qp_native_rec.n1) == 1
            @test get(qp_native_rec.rule_counters, :empty_cols, 0) == 1

            qp_native_doubleton_gpu = GPUPresolver.setup_gpu_qp_model(
                Q_doubleton_sep,
                sparse([1.0 1.0]),
                c_qp,
                [2.0],
                [2.0],
                [0.0, 0.0],
                [10.0, 10.0],
                7.0;
                verbose=false,
            )
            qp_native_doubleton_red, qp_native_doubleton_rec = GPUPresolver.GPUBackend.presolve_gpu(
                qp_native_doubleton_gpu;
                presolve_params=doubleton_pparams,
                verbose=false,
            )
            Q_native_doubleton_cpu, A_native_doubleton_cpu, c_native_doubleton_cpu, AL_native_doubleton_cpu, AU_native_doubleton_cpu, l_native_doubleton_cpu, u_native_doubleton_cpu, c0_native_doubleton_cpu =
                GPUPresolver.copy_qp_model_to_cpu(qp_native_doubleton_red)
            @test size(A_native_doubleton_cpu) == (0, 1)
            @test size(Q_native_doubleton_cpu) == (1, 1)
            @test isapprox(Q_native_doubleton_cpu[1, 1], 6.0, atol=1e-12)
            @test isapprox(c_native_doubleton_cpu[1], -10.0, atol=1e-12)
            @test length(AL_native_doubleton_cpu) == 0
            @test length(AU_native_doubleton_cpu) == 0
            @test isapprox(l_native_doubleton_cpu[1], 0.0, atol=1e-12)
            @test isapprox(u_native_doubleton_cpu[1], 2.0, atol=1e-12)
            @test isapprox(c0_native_doubleton_cpu, 25.0, atol=1e-12)
            @test Int(qp_native_doubleton_rec.m1) == 0
            @test Int(qp_native_doubleton_rec.n1) == 1
            @test get(qp_native_doubleton_rec.rule_counters, :doubleton_eq, 0) == 1

            doubleton_qdiag_zero_pparams = deepcopy(doubleton_pparams)
            doubleton_qdiag_zero_pparams.qp_doubleton_eq_require_qdiag_zero = true
            qp_doubleton_qdiag_blocked_result = GPUPresolver.run_presolve(
                qp_doubleton;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=doubleton_qdiag_zero_pparams,
                ),
            )
            @test qp_doubleton_qdiag_blocked_result.status == "OK"
            @test size(qp_doubleton_qdiag_blocked_result.reduced_problem.A) == (1, 2)
            @test size(qp_doubleton_qdiag_blocked_result.reduced_problem.Q) == (2, 2)
            @test GPUPresolver.free_presolve_state!(qp_doubleton_qdiag_blocked_result.state) === nothing

            guarded_doubleton_pparams = deepcopy(doubleton_pparams)
            guarded_doubleton_pparams.qp_doubleton_max_q_fill_abs = 0
            qp_doubleton_fill_guard_gpu = GPUPresolver.setup_gpu_qp_model(
                sparse([
                    2.0 0.0 0.0
                    0.0 4.0 1.0
                    0.0 1.0 3.0
                ]),
                sparse([1.0 1.0 0.0]),
                [3.0, 5.0, 0.0],
                [2.0],
                [2.0],
                [0.0, 0.0, -10.0],
                [10.0, 10.0, 10.0],
                7.0;
                verbose=false,
            )
            qp_doubleton_fill_guard_red, qp_doubleton_fill_guard_rec = GPUPresolver.GPUBackend.presolve_gpu(
                qp_doubleton_fill_guard_gpu;
                presolve_params=guarded_doubleton_pparams,
                verbose=false,
            )
            Q_fill_guard_cpu, A_fill_guard_cpu, _, _, _, _, _, c0_fill_guard_cpu =
                GPUPresolver.copy_qp_model_to_cpu(qp_doubleton_fill_guard_red)
            @test size(A_fill_guard_cpu) == (1, 3)
            @test size(Q_fill_guard_cpu) == (3, 3)
            @test isapprox(c0_fill_guard_cpu, 7.0, atol=1e-12)
            @test Int(qp_doubleton_fill_guard_rec.m1) == 1
            @test Int(qp_doubleton_fill_guard_rec.n1) == 3
            @test get(qp_doubleton_fill_guard_rec.rule_counters, :doubleton_eq, 0) == 0

            qp_doubleton_select_next_gpu = GPUPresolver.setup_gpu_qp_model(
                sparse([
                    2.0 0.0 0.0 0.0 0.0
                    0.0 4.0 0.0 0.0 1.0
                    0.0 0.0 3.0 0.0 0.0
                    0.0 0.0 0.0 2.0 0.0
                    0.0 1.0 0.0 0.0 1.0
                ]),
                sparse([
                    1.0 1.0 0.0 0.0 0.0
                    0.0 0.0 1.0 1.0 0.0
                ]),
                [3.0, 5.0, 1.0, 0.0, 0.0],
                [2.0, 1.0],
                [2.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, -10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
                7.0;
                verbose=false,
            )
            qp_doubleton_select_next_red, qp_doubleton_select_next_rec = GPUPresolver.GPUBackend.presolve_gpu(
                qp_doubleton_select_next_gpu;
                presolve_params=guarded_doubleton_pparams,
                verbose=false,
            )
            Q_select_next_cpu, A_select_next_cpu, c_select_next_cpu, AL_select_next_cpu, AU_select_next_cpu, l_select_next_cpu, u_select_next_cpu, c0_select_next_cpu =
                GPUPresolver.copy_qp_model_to_cpu(qp_doubleton_select_next_red)
            @test size(A_select_next_cpu) == (1, 4)
            @test size(Q_select_next_cpu) == (4, 4)
            @test isapprox(A_select_next_cpu[1, 1], 1.0, atol=1e-12)
            @test isapprox(A_select_next_cpu[1, 2], 1.0, atol=1e-12)
            @test isapprox(A_select_next_cpu[1, 3], 0.0, atol=1e-12)
            @test isapprox(A_select_next_cpu[1, 4], 0.0, atol=1e-12)
            @test isapprox(Q_select_next_cpu[3, 3], 5.0, atol=1e-12)
            @test isapprox(c_select_next_cpu[3], -1.0, atol=1e-12)
            @test isapprox(AL_select_next_cpu[1], 2.0, atol=1e-12)
            @test isapprox(AU_select_next_cpu[1], 2.0, atol=1e-12)
            @test isapprox(l_select_next_cpu[3], 0.0, atol=1e-12)
            @test isapprox(u_select_next_cpu[3], 1.0, atol=1e-12)
            @test isapprox(c0_select_next_cpu, 8.0, atol=1e-12)
            @test Int(qp_doubleton_select_next_rec.m1) == 1
            @test Int(qp_doubleton_select_next_rec.n1) == 4
            @test get(qp_doubleton_select_next_rec.rule_counters, :doubleton_eq, 0) == 1

            qp_native_doubleton_batch_gpu = GPUPresolver.setup_gpu_qp_model(
                sparse([
                    2.0 0.0 0.0 0.0
                    0.0 4.0 0.0 0.0
                    0.0 0.0 3.0 0.0
                    0.0 0.0 0.0 2.0
                ]),
                sparse([
                    1.0 1.0 0.0 0.0
                    0.0 0.0 1.0 1.0
                ]),
                [3.0, 5.0, 1.0, 2.0],
                [2.0, 1.0],
                [2.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0, 10.0],
                7.0;
                verbose=false,
            )
            qp_native_doubleton_batch_red, qp_native_doubleton_batch_rec = GPUPresolver.GPUBackend.presolve_gpu(
                qp_native_doubleton_batch_gpu;
                presolve_params=doubleton_pparams,
                verbose=false,
            )
            Q_native_doubleton_batch_cpu, A_native_doubleton_batch_cpu, c_native_doubleton_batch_cpu, AL_native_doubleton_batch_cpu, AU_native_doubleton_batch_cpu, l_native_doubleton_batch_cpu, u_native_doubleton_batch_cpu, c0_native_doubleton_batch_cpu =
                GPUPresolver.copy_qp_model_to_cpu(qp_native_doubleton_batch_red)
            @test size(A_native_doubleton_batch_cpu) == (0, 2)
            @test size(Q_native_doubleton_batch_cpu) == (2, 2)
            @test isapprox(Q_native_doubleton_batch_cpu[1, 1], 6.0, atol=1e-12)
            @test isapprox(Q_native_doubleton_batch_cpu[2, 2], 5.0, atol=1e-12)
            @test isapprox(c_native_doubleton_batch_cpu[1], -10.0, atol=1e-12)
            @test isapprox(c_native_doubleton_batch_cpu[2], -3.0, atol=1e-12)
            @test length(AL_native_doubleton_batch_cpu) == 0
            @test length(AU_native_doubleton_batch_cpu) == 0
            @test isapprox(l_native_doubleton_batch_cpu[1], 0.0, atol=1e-12)
            @test isapprox(u_native_doubleton_batch_cpu[1], 2.0, atol=1e-12)
            @test isapprox(l_native_doubleton_batch_cpu[2], 0.0, atol=1e-12)
            @test isapprox(u_native_doubleton_batch_cpu[2], 1.0, atol=1e-12)
            @test isapprox(c0_native_doubleton_batch_cpu, 28.0, atol=1e-12)
            @test Int(qp_native_doubleton_batch_rec.m1) == 0
            @test Int(qp_native_doubleton_batch_rec.n1) == 2
            @test get(qp_native_doubleton_batch_rec.rule_counters, :doubleton_eq, 0) == 2

            chained_qp_pparams = deepcopy(doubleton_pparams)
            chained_qp_pparams.enable_empty_cols = true
            chained_qp_pparams.col_rule_order = [:doubleton_eq, :empty_cols]
            qp_native_chain_gpu = GPUPresolver.setup_gpu_qp_model(
                Q_doubleton_sep,
                sparse([1.0 1.0]),
                c_qp,
                [2.0],
                [2.0],
                [0.0, 0.0],
                [10.0, 10.0],
                7.0;
                verbose=false,
            )
            qp_native_chain_red, qp_native_chain_rec = GPUPresolver.GPUBackend.presolve_gpu(
                qp_native_chain_gpu;
                presolve_params=chained_qp_pparams,
                verbose=false,
            )
            Q_native_chain_cpu, A_native_chain_cpu, c_native_chain_cpu, AL_native_chain_cpu, AU_native_chain_cpu, l_native_chain_cpu, u_native_chain_cpu, c0_native_chain_cpu =
                GPUPresolver.copy_qp_model_to_cpu(qp_native_chain_red)
            @test size(A_native_chain_cpu) == (0, 0)
            @test size(Q_native_chain_cpu) == (0, 0)
            @test isempty(c_native_chain_cpu)
            @test isempty(AL_native_chain_cpu)
            @test isempty(AU_native_chain_cpu)
            @test isempty(l_native_chain_cpu)
            @test isempty(u_native_chain_cpu)
            @test isapprox(c0_native_chain_cpu, 50.0 / 3.0, atol=1e-12)
            @test Int(qp_native_chain_rec.m1) == 0
            @test Int(qp_native_chain_rec.n1) == 0
            @test get(qp_native_chain_rec.rule_counters, :doubleton_eq, 0) == 1
            @test get(qp_native_chain_rec.rule_counters, :empty_cols, 0) == 1

            chained_qp_tiered_pparams = deepcopy(chained_qp_pparams)
            chained_qp_tiered_pparams.gpu_presolve_scheduler = :tiered
            qp_native_chain_tiered_gpu = GPUPresolver.setup_gpu_qp_model(
                Q_doubleton_sep,
                sparse([1.0 1.0]),
                c_qp,
                [2.0],
                [2.0],
                [0.0, 0.0],
                [10.0, 10.0],
                7.0;
                verbose=false,
            )
            qp_native_chain_tiered_red, qp_native_chain_tiered_rec = GPUPresolver.GPUBackend.presolve_gpu(
                qp_native_chain_tiered_gpu;
                presolve_params=chained_qp_tiered_pparams,
                verbose=false,
            )
            Q_native_chain_tiered_cpu, A_native_chain_tiered_cpu, c_native_chain_tiered_cpu, AL_native_chain_tiered_cpu, AU_native_chain_tiered_cpu, l_native_chain_tiered_cpu, u_native_chain_tiered_cpu, c0_native_chain_tiered_cpu =
                GPUPresolver.copy_qp_model_to_cpu(qp_native_chain_tiered_red)
            @test size(A_native_chain_tiered_cpu) == (0, 0)
            @test size(Q_native_chain_tiered_cpu) == (0, 0)
            @test isempty(c_native_chain_tiered_cpu)
            @test isempty(AL_native_chain_tiered_cpu)
            @test isempty(AU_native_chain_tiered_cpu)
            @test isempty(l_native_chain_tiered_cpu)
            @test isempty(u_native_chain_tiered_cpu)
            @test isapprox(c0_native_chain_tiered_cpu, 50.0 / 3.0, atol=1e-12)
            @test Int(qp_native_chain_tiered_rec.m1) == 0
            @test Int(qp_native_chain_tiered_rec.n1) == 0
            @test get(qp_native_chain_tiered_rec.rule_counters, :doubleton_eq, 0) == 1
            @test get(qp_native_chain_tiered_rec.rule_counters, :empty_cols, 0) == 1

            linear_eq_agg_pparams = GPUPresolver.GPUBackend.PresolveParams()
            linear_eq_agg_pparams.max_iters = 1
            linear_eq_agg_pparams.enable_close_bounds = false
            linear_eq_agg_pparams.enable_empty_rows = false
            linear_eq_agg_pparams.enable_singleton_rows = false
            linear_eq_agg_pparams.enable_activity_checks = false
            linear_eq_agg_pparams.enable_primal_propagation = false
            linear_eq_agg_pparams.enable_parallel_rows = false
            linear_eq_agg_pparams.enable_redundant_bounds = false
            linear_eq_agg_pparams.enable_empty_cols = false
            linear_eq_agg_pparams.enable_singleton_cols_eq = false
            linear_eq_agg_pparams.enable_singleton_cols_dual_infer = false
            linear_eq_agg_pparams.enable_doubleton_eq = false
            linear_eq_agg_pparams.enable_linear_eq_agg = true
            linear_eq_agg_pparams.enable_dual_fix = false
            linear_eq_agg_pparams.enable_parallel_cols = false
            linear_eq_agg_pparams.row_rule_order = Symbol[]
            linear_eq_agg_pparams.col_rule_order = [:linear_eq_agg]

            qp_linear_eq_agg = GPUPresolver.QPProblem(
                spdiagm(0 => [0.0, 2.0, 3.0, 1.0]),
                sparse([
                    1.0 1.0 1.0 0.0
                    1.0 0.0 0.0 1.0
                ]),
                [0.0, 7.0, 11.0, 13.0],
                [3.0, 1.0],
                [3.0, 1.0],
                [-Inf, 0.0, 0.0, 0.0],
                [Inf, 10.0, 10.0, 10.0],
                17.0,
            )
            qp_linear_eq_agg_result = GPUPresolver.run_presolve(
                qp_linear_eq_agg;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=linear_eq_agg_pparams,
                ),
            )
            @test qp_linear_eq_agg_result.status == "OK"
            qp_linear_eq_agg_red = qp_linear_eq_agg_result.reduced_problem
            @test size(qp_linear_eq_agg_red.A) == (1, 3)
            @test size(qp_linear_eq_agg_red.Q) == (3, 3)
            @test isapprox(qp_linear_eq_agg_red.A[1, 1], -1.0, atol=1e-12)
            @test isapprox(qp_linear_eq_agg_red.A[1, 2], -1.0, atol=1e-12)
            @test isapprox(qp_linear_eq_agg_red.A[1, 3], 1.0, atol=1e-12)
            @test isapprox(qp_linear_eq_agg_red.AL[1], -2.0, atol=1e-12)
            @test isapprox(qp_linear_eq_agg_red.AU[1], -2.0, atol=1e-12)
            @test isapprox(qp_linear_eq_agg_red.c[1], 7.0, atol=1e-12)
            @test isapprox(qp_linear_eq_agg_red.c[2], 11.0, atol=1e-12)
            @test isapprox(qp_linear_eq_agg_red.c[3], 13.0, atol=1e-12)
            @test isapprox(qp_linear_eq_agg_red.obj_constant, 17.0, atol=1e-12)
            x_linear_eq_agg_org, _, _ = GPUPresolver.run_postsolve(
                qp_linear_eq_agg_result.state,
                [1.0, 2.0, 1.0],
                [0.0],
                [0.0, 0.0, 0.0],
            )
            @test length(x_linear_eq_agg_org) == 4
            @test isapprox(x_linear_eq_agg_org[1], 0.0, atol=1e-12)
            @test isapprox(x_linear_eq_agg_org[2], 1.0, atol=1e-12)
            @test isapprox(x_linear_eq_agg_org[3], 2.0, atol=1e-12)
            @test isapprox(x_linear_eq_agg_org[4], 1.0, atol=1e-12)
            @test get(qp_linear_eq_agg_result.state.inner.record.rule_counters, :linear_eq_agg, 0) == 1
            @test GPUPresolver.free_presolve_state!(qp_linear_eq_agg_result.state) === nothing

            dual_fix_pparams = GPUPresolver.GPUBackend.PresolveParams()
            dual_fix_pparams.max_iters = 1
            dual_fix_pparams.enable_close_bounds = false
            dual_fix_pparams.enable_empty_rows = false
            dual_fix_pparams.enable_singleton_rows = false
            dual_fix_pparams.enable_activity_checks = false
            dual_fix_pparams.enable_primal_propagation = false
            dual_fix_pparams.enable_parallel_rows = false
            dual_fix_pparams.enable_redundant_bounds = false
            dual_fix_pparams.enable_empty_cols = false
            dual_fix_pparams.enable_singleton_cols_eq = false
            dual_fix_pparams.enable_singleton_cols_dual_infer = false
            dual_fix_pparams.enable_doubleton_eq = false
            dual_fix_pparams.enable_dual_fix = true
            dual_fix_pparams.enable_parallel_cols = false
            dual_fix_pparams.row_rule_order = Symbol[]
            dual_fix_pparams.col_rule_order = [:dual_fix]

            qp_dual_fix = GPUPresolver.QPProblem(
                spdiagm(0 => [2.0, 4.0]),
                sparse([1.0 -1.0]),
                [1.0, 5.0],
                [4.0],
                [Inf],
                [0.0, -1.0],
                [10.0, 5.0],
                1.5,
            )
            qp_dual_fix_result = GPUPresolver.run_presolve(
                qp_dual_fix;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=dual_fix_pparams,
                ),
            )
            @test qp_dual_fix_result.status == "OK"
            qp_dual_fix_red = qp_dual_fix_result.reduced_problem
            @test size(qp_dual_fix_red.A) == (1, 1)
            @test size(qp_dual_fix_red.Q) == (1, 1)
            @test isapprox(qp_dual_fix_red.Q[1, 1], 2.0, atol=1e-12)
            @test isapprox(qp_dual_fix_red.A[1, 1], 1.0, atol=1e-12)
            @test isapprox(qp_dual_fix_red.AL[1], 3.0, atol=1e-12)
            @test qp_dual_fix_red.AU[1] == Inf
            @test isapprox(qp_dual_fix_red.obj_constant, -1.5, atol=1e-12)
            x_dual_fix_org, _, _ = GPUPresolver.run_postsolve(
                qp_dual_fix_result.state,
                [5.0],
                [0.0],
                [0.0],
            )
            @test isapprox(x_dual_fix_org[1], 5.0, atol=1e-12)
            @test isapprox(x_dual_fix_org[2], -1.0, atol=1e-12)
            @test GPUPresolver.free_presolve_state!(qp_dual_fix_result.state) === nothing

            qp_native_dual_gpu = GPUPresolver.setup_gpu_qp_model(
                spdiagm(0 => [2.0, 4.0]),
                sparse([1.0 -1.0]),
                [1.0, 5.0],
                [4.0],
                [Inf],
                [0.0, -1.0],
                [10.0, 5.0],
                1.5;
                verbose=false,
            )
            qp_native_dual_red, qp_native_dual_rec = GPUPresolver.GPUBackend.presolve_gpu(
                qp_native_dual_gpu;
                presolve_params=dual_fix_pparams,
                verbose=false,
            )
            Q_native_dual_cpu, A_native_dual_cpu, c_native_dual_cpu, AL_native_dual_cpu, AU_native_dual_cpu, l_native_dual_cpu, u_native_dual_cpu, c0_native_dual_cpu =
                GPUPresolver.copy_qp_model_to_cpu(qp_native_dual_red)
            @test size(A_native_dual_cpu) == (1, 1)
            @test size(Q_native_dual_cpu) == (1, 1)
            @test isapprox(Q_native_dual_cpu[1, 1], 2.0, atol=1e-12)
            @test isapprox(A_native_dual_cpu[1, 1], 1.0, atol=1e-12)
            @test isapprox(c_native_dual_cpu[1], 1.0, atol=1e-12)
            @test isapprox(AL_native_dual_cpu[1], 3.0, atol=1e-12)
            @test AU_native_dual_cpu[1] == Inf
            @test isapprox(l_native_dual_cpu[1], 0.0, atol=1e-12)
            @test isapprox(u_native_dual_cpu[1], 10.0, atol=1e-12)
            @test isapprox(c0_native_dual_cpu, -1.5, atol=1e-12)
            @test Int(qp_native_dual_rec.n1) == 1
            @test get(qp_native_dual_rec.rule_counters, :dual_fix, 0) == 1

            qp_dual_fix_coupled = GPUPresolver.QPProblem(
                sparse([2.0 1.0; 1.0 4.0]),
                sparse([1.0 -1.0]),
                [1.0, 5.0],
                [4.0],
                [Inf],
                [0.0, -1.0],
                [10.0, 5.0],
                1.5,
            )
            qp_dual_fix_coupled_result = GPUPresolver.run_presolve(
                qp_dual_fix_coupled;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=dual_fix_pparams,
                ),
            )
            @test qp_dual_fix_coupled_result.status == "OK"
            @test size(qp_dual_fix_coupled_result.reduced_problem.A) == (1, 2)
            @test size(qp_dual_fix_coupled_result.reduced_problem.Q) == (2, 2)
            @test GPUPresolver.free_presolve_state!(qp_dual_fix_coupled_result.state) === nothing

            qp_dual_fix_unbounded = GPUPresolver.QPProblem(
                spdiagm(0 => [2.0, 0.0]),
                sparse([1.0 1.0]),
                [1.0, -1.0],
                [4.0],
                [Inf],
                [0.0, 0.0],
                [10.0, Inf],
                0.0,
            )
            @test_throws ErrorException GPUPresolver.run_presolve(
                qp_dual_fix_unbounded;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=dual_fix_pparams,
                ),
            )

            parallel_pparams = GPUPresolver.GPUBackend.PresolveParams()
            parallel_pparams.max_iters = 1
            parallel_pparams.enable_close_bounds = false
            parallel_pparams.enable_empty_rows = false
            parallel_pparams.enable_singleton_rows = false
            parallel_pparams.enable_activity_checks = false
            parallel_pparams.enable_primal_propagation = false
            parallel_pparams.enable_parallel_rows = false
            parallel_pparams.enable_redundant_bounds = false
            parallel_pparams.enable_empty_cols = false
            parallel_pparams.enable_singleton_cols_eq = false
            parallel_pparams.enable_singleton_cols_dual_infer = false
            parallel_pparams.enable_doubleton_eq = false
            parallel_pparams.enable_dual_fix = false
            parallel_pparams.enable_parallel_cols = true
            parallel_pparams.row_rule_order = Symbol[]
            parallel_pparams.col_rule_order = [:parallel_cols]

            # Exact QP parallel columns: A[:,2] = 2A[:,1], c2 = 2c1, and
            # Q[:,2] = 2Q[:,1], so the objective depends only on y = x1 + 2x2.
            qp_parallel = GPUPresolver.QPProblem(
                sparse([3.0 6.0; 6.0 12.0]),
                sparse([1.0 2.0]),
                [1.0, 2.0],
                [0.0],
                [16.0],
                [0.0, 0.0],
                [10.0, 3.0],
                4.0,
            )
            qp_parallel_result = GPUPresolver.run_presolve(
                qp_parallel;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=parallel_pparams,
                ),
            )
            @test qp_parallel_result.status == "OK"
            qp_parallel_red = qp_parallel_result.reduced_problem
            @test size(qp_parallel_red.A) == (1, 1)
            @test size(qp_parallel_red.Q) == (1, 1)
            @test isapprox(qp_parallel_red.A[1, 1], 1.0, atol=1e-12)
            @test isapprox(qp_parallel_red.Q[1, 1], 3.0, atol=1e-12)
            @test isapprox(qp_parallel_red.c[1], 1.0, atol=1e-12)
            @test isapprox(qp_parallel_red.l[1], 0.0, atol=1e-12)
            @test isapprox(qp_parallel_red.u[1], 16.0, atol=1e-12)
            x_parallel_org, _, _ = GPUPresolver.run_postsolve(
                qp_parallel_result.state,
                [8.0],
                [0.0],
                [0.0],
            )
            @test isapprox(x_parallel_org[1] + 2.0 * x_parallel_org[2], 8.0, atol=1e-12)
            @test 0.0 <= x_parallel_org[1] <= 10.0
            @test 0.0 <= x_parallel_org[2] <= 3.0
            @test GPUPresolver.free_presolve_state!(qp_parallel_result.state) === nothing

            qp_native_parallel_gpu = GPUPresolver.setup_gpu_qp_model(
                sparse([3.0 6.0; 6.0 12.0]),
                sparse([1.0 2.0]),
                [1.0, 2.0],
                [0.0],
                [16.0],
                [0.0, 0.0],
                [10.0, 3.0],
                4.0;
                verbose=false,
            )
            qp_native_parallel_red, qp_native_parallel_rec = GPUPresolver.GPUBackend.presolve_gpu(
                qp_native_parallel_gpu;
                presolve_params=parallel_pparams,
                verbose=false,
            )
            Q_native_parallel_cpu, A_native_parallel_cpu, c_native_parallel_cpu, AL_native_parallel_cpu, AU_native_parallel_cpu, l_native_parallel_cpu, u_native_parallel_cpu, c0_native_parallel_cpu =
                GPUPresolver.copy_qp_model_to_cpu(qp_native_parallel_red)
            @test size(A_native_parallel_cpu) == (1, 1)
            @test size(Q_native_parallel_cpu) == (1, 1)
            @test isapprox(A_native_parallel_cpu[1, 1], 1.0, atol=1e-12)
            @test isapprox(Q_native_parallel_cpu[1, 1], 3.0, atol=1e-12)
            @test isapprox(c_native_parallel_cpu[1], 1.0, atol=1e-12)
            @test isapprox(AL_native_parallel_cpu[1], 0.0, atol=1e-12)
            @test isapprox(AU_native_parallel_cpu[1], 16.0, atol=1e-12)
            @test isapprox(l_native_parallel_cpu[1], 0.0, atol=1e-12)
            @test isapprox(u_native_parallel_cpu[1], 16.0, atol=1e-12)
            @test isapprox(c0_native_parallel_cpu, 4.0, atol=1e-12)
            @test Int(qp_native_parallel_rec.n1) == 1
            @test get(qp_native_parallel_rec.rule_counters, :parallel_cols, 0) == 1

            # Diagonal Q with nonzero curvature is not an exact aggregate
            # objective, so the QP parallel-column pass must skip it.
            qp_parallel_diag = GPUPresolver.QPProblem(
                spdiagm(0 => [3.0, 12.0]),
                sparse([1.0 2.0]),
                [1.0, 2.0],
                [0.0],
                [16.0],
                [0.0, 0.0],
                [10.0, 3.0],
                4.0,
            )
            qp_parallel_diag_result = GPUPresolver.run_presolve(
                qp_parallel_diag;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=parallel_pparams,
                ),
            )
            @test qp_parallel_diag_result.status == "OK"
            @test size(qp_parallel_diag_result.reduced_problem.A) == (1, 2)
            @test size(qp_parallel_diag_result.reduced_problem.Q) == (2, 2)
            @test GPUPresolver.free_presolve_state!(qp_parallel_diag_result.state) === nothing
        else
            qp_result = GPUPresolver.run_presolve(
                qp;
                config=GPUPresolver.PresolveConfig(
                    backend="GPU",
                    verbose=false,
                    presolve_params=pparams,
                ),
            )
            @test qp_result.status == "PRESOLVE_FAILED"
            @test qp_result.reduced_problem === qp
            @test isnothing(qp_result.state)
        end
    end

    @testset "One-Call MPS Presolve API" begin
        mps_text = """
NAME          MINI
ROWS
 N  OBJ
 L  R1
COLUMNS
    X1        OBJ               1
    X1        R1                1
RHS
    RHS1      R1                1
BOUNDS
 LO BND1      X1                0
 UP BND1      X1                2
ENDATA
"""
        lp_path = tempname() * ".mps"
        try
            open(lp_path, "w") do io
                write(io, mps_text)
            end

            res_lp = GPUPresolver.run_presolve(
                lp_path;
                problem_type=:LP,
                config=GPUPresolver.PresolveConfig(backend="NONE", verbose=false),
            )
            @test res_lp.status == "SKIPPED"
            @test res_lp.reduced_problem isa GPUPresolver.LPProblem
            @test size(res_lp.reduced_problem.model.A) == (1, 1)
            @test GPUPresolver.free_presolve_state!(res_lp.state) === nothing

            res_qp = GPUPresolver.run_presolve(
                lp_path;
                problem_type="QP",
                config=GPUPresolver.PresolveConfig(backend="NONE", verbose=false),
            )
            @test res_qp.status == "SKIPPED"
            @test res_qp.reduced_problem isa GPUPresolver.QPProblem
            @test size(res_qp.reduced_problem.A) == (1, 1)
            @test size(res_qp.reduced_problem.Q) == (1, 1)
            @test GPUPresolver.free_presolve_state!(res_qp.state) === nothing
        finally
            isfile(lp_path) && rm(lp_path)
        end
    end

    @testset "LP Doubleton Batch Prunes Structural Zeros" begin
        A = sparse([
            1.0 1.0
            1.0 1.0
        ])
        AL = [1.0, -Inf]
        AU = [1.0, 2.0]
        c = [0.0, 0.0]
        l = [0.0, 0.0]
        u = [10.0, 10.0]
        model = GPUPresolver.build_from_Abc(A, c, AL, AU, l, u, 0.0)

        pparams = GPUPresolver.GPUBackend.PresolveParams()
        pparams.max_iters = 1
        pparams.enable_close_bounds = false
        pparams.enable_empty_rows = false
        pparams.enable_singleton_rows = false
        pparams.enable_activity_checks = false
        pparams.enable_primal_propagation = false
        pparams.enable_parallel_rows = false
        pparams.enable_redundant_bounds = false
        pparams.enable_empty_cols = false
        pparams.enable_singleton_cols_eq = false
        pparams.enable_singleton_cols_dual_infer = false
        pparams.enable_doubleton_eq = true
        pparams.enable_dual_fix = false
        pparams.enable_parallel_cols = false
        pparams.row_rule_order = Symbol[]
        pparams.col_rule_order = [:doubleton_eq]

        result = GPUPresolver.run_presolve(
            GPUPresolver.LPProblem(model);
            config=GPUPresolver.PresolveConfig(
                backend="GPU",
                verbose=false,
                presolve_params=pparams,
            ),
        )
        @test result.status == "OK"

        reduced = result.reduced_problem.model
        @test size(reduced.A) == (1, 1)
        @test size(reduced.AT) == (1, 1)
        @test nnz(reduced.A) == 0
        @test nnz(reduced.AT) == 0
        @test reduced.AT == sparse(transpose(reduced.A))

        vals_A = findnz(reduced.A)[3]
        vals_AT = findnz(reduced.AT)[3]
        @test all(abs.(vals_A) .> pparams.zero_tol)
        @test all(abs.(vals_AT) .> pparams.zero_tol)
        @test GPUPresolver.free_presolve_state!(result.state) === nothing
    end
end
