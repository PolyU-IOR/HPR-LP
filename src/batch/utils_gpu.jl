function _as_float64_matrix(name::AbstractString, X::AbstractMatrix{<:Real})
	return Matrix{Float64}(X)
end

function _as_float64_vector(name::AbstractString, x::AbstractVector{<:Real})
	return Vector{Float64}(x)
end

function _validate_batched_dimensions(
	A,
	C::AbstractMatrix,
	AL::AbstractMatrix,
	AU::AbstractMatrix,
	L::AbstractMatrix,
	U::AbstractMatrix,
	obj_constants::AbstractVector,
)
	m, n = size(A)
	size(C, 1) == n || throw(ArgumentError("C must have size n x B; expected first dimension $(n), got $(size(C, 1))."))
	B = size(C, 2)
	size(L) == (n, B) || throw(ArgumentError("L must have size $(n) x $(B); got $(size(L))."))
	size(U) == (n, B) || throw(ArgumentError("U must have size $(n) x $(B); got $(size(U))."))
	size(AL) == (m, B) || throw(ArgumentError("AL must have size $(m) x $(B); got $(size(AL))."))
	size(AU) == (m, B) || throw(ArgumentError("AU must have size $(m) x $(B); got $(size(AU))."))
	length(obj_constants) == B || throw(ArgumentError("obj_constants must have length $(B); got $(length(obj_constants))."))
	return m, n, B
end

function _column_norms_gpu(X::CuMatrix{Float64})
	return vec(sqrt.(sum(abs2.(X); dims=1)))
end

function _batched_bound_norms_gpu(AL::CuMatrix{Float64}, AU::CuMatrix{Float64})
	AL_ninf = ifelse.(AL .== -Inf, 0.0, AL)
	AU_ninf = ifelse.(AU .== Inf, 0.0, AU)
	return vec(sqrt.(sum(max.(abs.(AL_ninf), abs.(AU_ninf)).^2; dims=1)))
end

function build_batched_shared_matrix_gpu(A::Union{SparseMatrixCSC,Matrix}, params::HPRLP_parameters)
	params.use_gpu || throw(ArgumentError("build_batched_shared_matrix_gpu requires params.use_gpu=true."))
	validate_gpu_parameters!(params)

	A_sparse = A isa Matrix ? sparse(A) : A
	A_sparse = SparseMatrixCSC{Float64,Int32}(A_sparse)
	m, n = size(A_sparse)

	CUDA.device!(params.device_number)
	shared = BatchedSharedMatrix_gpu(
		CuSparseMatrixCSR(copy(A_sparse)),
		CuSparseMatrixCSR(transpose(A_sparse)),
		m,
		n,
		CUDA.ones(Float64, m),
		CUDA.ones(Float64, n),
		NaN,
	)

	dummy_model = LP_info_gpu(
		shared.A,
		shared.AT,
		CUDA.zeros(Float64, n),
		CUDA.zeros(Float64, m),
		CUDA.zeros(Float64, m),
		CUDA.zeros(Float64, n),
		CUDA.zeros(Float64, n),
		0.0,
		Int32(0),
		CUDA.zeros(Int32, n),
	)
	scaling_info = scaling_gpu!(
		dummy_model,
		params.use_Curtis_Reid_scaling,
		params.use_Ruiz_scaling,
		params.use_Pock_Chambolle_scaling,
		false,
	)
	shared.row_norm = scaling_info.row_norm
	shared.col_norm = scaling_info.col_norm
	shared.lambda_max = compute_batched_shared_lambda_max_gpu(shared, params)
	CUDA.synchronize()
	return shared
end

function compute_batched_shared_lambda_max_gpu(shared::BatchedSharedMatrix_gpu, params::HPRLP_parameters)
	dummy_model = LP_info_gpu(
		shared.A,
		shared.AT,
		CUDA.zeros(Float64, shared.n),
		CUDA.zeros(Float64, shared.m),
		CUDA.zeros(Float64, shared.m),
		CUDA.zeros(Float64, shared.n),
		CUDA.zeros(Float64, shared.n),
		0.0,
		Int32(0),
		CUDA.zeros(Int32, shared.n),
	)
	dummy_scaling = Scaling_info_gpu(
		CUDA.zeros(Float64, shared.n),
		CUDA.zeros(Float64, shared.n),
		shared.row_norm,
		shared.col_norm,
		1.0,
		1.0,
		1.0,
		1.0,
		1.0,
		1.0,
	)
	lambda_params = HPRLP_parameters()
	lambda_params.use_gpu = true
	lambda_params.CUSPARSE_spmv = params.CUSPARSE_spmv
	lambda_params.verbose = params.verbose
	lambda_params.initial_x = nothing
	lambda_params.initial_y = nothing
	ws = allocate_workspace_gpu(dummy_model, dummy_scaling, lambda_params)
	compute_maximum_eigenvalue!(dummy_model, ws, lambda_params)
	return ws.lambda_max
end

function build_batched_lp_gpu(
	shared::BatchedSharedMatrix_gpu,
	C::AbstractMatrix{<:Real},
	AL::AbstractMatrix{<:Real},
	AU::AbstractMatrix{<:Real},
	L::AbstractMatrix{<:Real},
	U::AbstractMatrix{<:Real};
	obj_constants::AbstractVector{<:Real}=zeros(size(C, 2)),
	use_bc_scaling::Bool=true,
)
	_validate_batched_dimensions(shared.A, C, AL, AU, L, U, obj_constants)

	C_gpu = CuArray(_as_float64_matrix("C", C))
	AL_gpu = CuArray(_as_float64_matrix("AL", AL))
	AU_gpu = CuArray(_as_float64_matrix("AU", AU))
	L_gpu = CuArray(_as_float64_matrix("L", L))
	U_gpu = CuArray(_as_float64_matrix("U", U))
	obj_gpu = CuArray(_as_float64_vector("obj_constants", obj_constants))

	norm_b_org = 1.0 .+ _batched_bound_norms_gpu(AL_gpu, AU_gpu)
	norm_c_org = 1.0 .+ _column_norms_gpu(C_gpu)

	AL_gpu ./= reshape(shared.row_norm, :, 1)
	AU_gpu ./= reshape(shared.row_norm, :, 1)
	C_gpu ./= reshape(shared.col_norm, :, 1)
	L_gpu .*= reshape(shared.col_norm, :, 1)
	U_gpu .*= reshape(shared.col_norm, :, 1)

	B = size(C_gpu, 2)
	b_scale = CUDA.ones(Float64, B)
	c_scale = CUDA.ones(Float64, B)
	if use_bc_scaling
		b_scale .= 1.0 .+ _batched_bound_norms_gpu(AL_gpu, AU_gpu)
		c_scale .= 1.0 .+ _column_norms_gpu(C_gpu)
		AL_gpu ./= reshape(b_scale, 1, :)
		AU_gpu ./= reshape(b_scale, 1, :)
		C_gpu ./= reshape(c_scale, 1, :)
		L_gpu ./= reshape(b_scale, 1, :)
		U_gpu ./= reshape(b_scale, 1, :)
	end

	norm_b = _batched_bound_norms_gpu(AL_gpu, AU_gpu)
	norm_c = _column_norms_gpu(C_gpu)

	AL_gpu .= ifelse.(AL_gpu .== -Inf, -1.0e100, AL_gpu)
	AU_gpu .= ifelse.(AU_gpu .== Inf, 1.0e100, AU_gpu)
	L_gpu .= ifelse.(L_gpu .== -Inf, -1.0e100, L_gpu)
	U_gpu .= ifelse.(U_gpu .== Inf, 1.0e100, U_gpu)
	CUDA.synchronize()

	batch = BatchedLPData_gpu(C_gpu, AL_gpu, AU_gpu, L_gpu, U_gpu, obj_gpu, B)
	scaling = BatchedScalingInfo_gpu(
		shared.row_norm,
		shared.col_norm,
		b_scale,
		c_scale,
		norm_b,
		norm_c,
		norm_b_org,
		norm_c_org,
	)
	return batch, scaling
end
