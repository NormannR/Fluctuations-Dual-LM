module EconModel

	using SymEngine
	using SparseSym
	using SparseArrays
	using LinearAlgebra
	using Kamenik

	export Model, Order1, Order2, Order3, solve_order1!, solve_order2!, solve_order3!, chainrule, subs_param, Aζ_k, perm_1

	struct Model{T <: AbstractFloat}
		calibrated::Dict{Symbol, T}
	    ss_y::Dict{Symbol, T}
	    ss_x::Dict{Symbol, T}
	    params::Vector{Basic}
	    yp::Vector{Basic}
	    y::Vector{Basic}
	    xp::Vector{Basic}
	    x::Vector{Basic}
	    Eq::Vector{Basic}
	    ζ::SparseMatrixCSC{Basic}
	    η::SparseMatrixCSC{T}
		η_sgu::SparseMatrixCSC{T}
	    n_f::Int
	    n_f_sgu::Int
	    n_x::Int
	    n_x_sgu::Int
	    n_y::Int
	    ind_f_sgu::UnitRange{Int}
	    ind_yp::UnitRange{Int}
	    ind_y::UnitRange{Int}
	    ind_xp::UnitRange{Int}
	    ind_xp_sgu::UnitRange{Int}
	    ind_x::UnitRange{Int}
	    ind_x_sgu::UnitRange{Int}
	end

	mutable struct Order1{T<:AbstractFloat}
		M::Model{T}
		f_v::SparseMatrixCSC{T}
		g_x::SparseMatrixCSC{T}
		g_x_sgu::SparseMatrixCSC{T}
		h_x::SparseMatrixCSC{T}
		h_x_sgu::SparseMatrixCSC{T}
		eu::Vector{Bool}
		params::Vector{T}
	end

	mutable struct Order2{T<:AbstractFloat}
		O1::Order1{T}
		f_vv::SparseMatrixCSC{T}
		Vx0::SparseMatrixCSC{T}
		Vx1::SparseMatrixCSC{T}
		g_xx::SparseMatrixCSC{T}
		g_xx_sgu::SparseMatrixCSC{T}
		g_σσ_sgu::SparseVector{T}
		h_xx::SparseMatrixCSC{T}
		h_xx_sgu::SparseMatrixCSC{T}
		h_σσ_sgu::SparseVector{T}
		select_xx::Vector{Int}
		select_σσ::Int
		Z_21::SparseMatrixCSC{T}
		Z_22::SparseVector{T}
	end

	function Order1(M::Model, f_v::SparseMatrixCSC{Basic}, params::Vector{T}) where T<:AbstractFloat
		g_x = spzeros(M.n_y, M.n_x)
		g_x_sgu = spzeros(M.n_y, M.n_x_sgu)
		h_x = spzeros(M.n_x, M.n_x)
		h_x_sgu = spzeros(M.n_x_sgu, M.n_x_sgu)
		eu = [false, false]
		return Order1(M, subs_param(f_v, params, M), g_x, g_x_sgu, h_x, h_x_sgu, eu, params)
	end

	function Order2(O1::Order1, f_vv::SparseMatrixCSC{Basic})

		n_x = O1.M.n_x
		n_y = O1.M.n_y
		n_x_sgu = O1.M.n_x_sgu

		Vx0 = spzeros(2*n_x+2*n_y, n_x)
		Vx1 = spzeros(2*n_x+2*n_y, n_x)

		ind = LinearIndices((n_x,n_x))
		select_xx = ind[1:end-1, 1:end-1][:]
		Z_21 = spI(n_x^2)[:,select_xx]
		select_σσ = ind[end, end]
		Z_22 = spI(n_x^2)[:,select_σσ]

		g_xx = spzeros(n_y, n_x^2)
		g_xx_sgu = spzeros(n_y, n_x_sgu^2)
		h_xx = spzeros(n_x, n_x^2)
		h_xx_sgu = spzeros(n_x_sgu, n_x_sgu^2)
		g_σσ_sgu = spzeros(n_y)
		h_σσ_sgu = spzeros(n_x_sgu)

		return Order2(O1, subs_param(f_vv, O1.params, O1.M), Vx0, Vx1, g_xx, g_xx_sgu, g_σσ_sgu, h_xx, h_xx_sgu, h_σσ_sgu, select_xx, select_σσ, Z_21, Z_22)

	end

	mutable struct Order3{T <: AbstractFloat}
		O2::Order2{T}
		f_vvv::SparseMatrixCSC{T}
		Vxx0::SparseMatrixCSC{T}
		Vxx1::SparseMatrixCSC{T}
		g_xxx::SparseMatrixCSC{T}
		g_xxx_sgu::SparseMatrixCSC{T}
		g_xσσ_sgu::SparseMatrixCSC{T}
		g_σσσ_sgu::SparseVector{T}
		h_xxx::SparseMatrixCSC{T}
		h_xxx_sgu::SparseMatrixCSC{T}
		h_xσσ_sgu::SparseMatrixCSC{T}
		h_σσσ_sgu::SparseVector{T}
		select_xxx::Vector{Int}
		select_xσσ::Vector{Int}
		select_σσσ::Int
		Z_31::SparseMatrixCSC{T}
		Z_32::SparseMatrixCSC{T}
		Z_33::SparseVector{T}
		Ω₁::SparseMatrixCSC{T}
	end

	function Order3(O2::Order2, f_vvv::SparseMatrixCSC{Basic})

		M = O2.O1.M

		n_x = O2.O1.M.n_x
		n_y = O2.O1.M.n_y
		n_x_sgu = O2.O1.M.n_x_sgu
		n_f_sgu = O2.O1.M.n_f_sgu
		n_f = O2.O1.M.n_f
		η_sgu = O2.O1.M.η_sgu

		params = O2.O1.params

	    ind = LinearIndices((n_x,n_x,n_x))
	    select_xxx = ind[1:end-1, 1:end-1, 1:end-1][:]
	    Z_31 = spI(n_x^3)[:, select_xxx]

	    select_xσσ = ind[1:end-1, end, end][:]
	    Z_32 = spI(n_x^3)[:, select_xσσ]

	    ind = LinearIndices((n_x,n_x,n_x))
	    select_σσσ = ind[end, end, end]
	    Z_33 = spI(n_x^3)[:, select_σσσ]

	    spI_n_x = spI(n_x^3)
	    Ω₁ = spI_n_x[:, permutedims(ind, [3, 1, 2])[:]] + spI_n_x[:, permutedims(ind, [1, 3, 2])[:]] + spI_n_x[:, permutedims(ind, [1, 2, 3])[:]]

		g_xxx = spzeros(n_y, n_x^3)
		h_xxx = spzeros(n_x, n_x^3)
		g_xxx_sgu = spzeros(n_y, n_x_sgu^3)
		h_xxx_sgu = spzeros(n_x_sgu, n_x_sgu^3)
		g_xσσ_sgu = spzeros(n_y, n_x_sgu)
		h_xσσ_sgu = spzeros(n_x_sgu, n_x_sgu)
		g_σσσ_sgu = spzeros(n_y)
		h_σσσ_sgu = spzeros(n_x_sgu)

		Vxx0 = spzeros(2*n_x+2*n_y, n_x^2)
		Vxx1 = spzeros(2*n_x+2*n_y, n_x^2)

		return Order3(O2, subs_param(f_vvv, params, M), Vxx0, Vxx1, g_xxx, g_xxx_sgu, g_xσσ_sgu, g_σσσ_sgu, h_xxx, h_xxx_sgu, h_xσσ_sgu, h_σσσ_sgu, select_xxx, select_xσσ, select_σσσ, Z_31, Z_32, Z_33, Ω₁)

	end

	function solve_order1!(O1::Order1)

		A = hcat(O1.f_v[O1.M.ind_f_sgu, O1.M.ind_xp_sgu], O1.f_v[O1.M.ind_f_sgu, O1.M.ind_yp])
		B = -hcat(O1.f_v[O1.M.ind_f_sgu, O1.M.ind_x_sgu], O1.f_v[O1.M.ind_f_sgu, O1.M.ind_y])

		O1.g_x_sgu, O1.h_x_sgu, O1.eu = solve_eig(Array(A), Array(B), O1.M.n_x_sgu)

		if all(O1.eu)
			O1.g_x[:, 1:O1.M.n_x_sgu] .= sparse(O1.g_x_sgu)
			O1.h_x[1:O1.M.n_x_sgu, 1:O1.M.n_x_sgu] .= sparse(O1.h_x_sgu)
			O1.h_x[end,end] = 1.
		end

	end

	function solve_eig(A::Array{T,2}, B::Array{T,2}, n_x::Int) where T<: AbstractFloat

		F = eigen(B,A)
		perm = sortperm(abs.(F.values))
		V = F.vectors[:,perm]
		D = F.values[perm]
		m = findlast(abs.(D) .< 1)
		eu = [true,true]

		if m > n_x
			eu[2] = false
			println("WARNING: the equilibrium is not unique !")
		elseif m < n_x
			eu[1] = false
			println("WARNING: the equilibrium does not exist !")
		end

		if all(eu)
			h_x = V[1:m,1:m]*Diagonal(D)[1:m,1:m]*inv(V[1:m,1:m])
			g_x = V[m+1:end,1:m]*inv(V[1:m,1:m])
		end

		return (sparse(g_x), sparse(h_x), eu)

	end

	function solve_order2!(O2::Order2, Eϵ::Dict)

		M = O2.O1.M

		n_x = O2.O1.M.n_x
		n_y = O2.O1.M.n_y
		n_x_sgu = O2.O1.M.n_x_sgu
		n_f_sgu = O2.O1.M.n_f_sgu
		n_f = O2.O1.M.n_f
		η_sgu = O2.O1.M.η_sgu

		ind_f_sgu = O2.O1.M.ind_f_sgu
		ind_yp = O2.O1.M.ind_yp
		ind_y = O2.O1.M.ind_y
		ind_xp = O2.O1.M.ind_xp
		ind_xp_sgu = O2.O1.M.ind_xp_sgu
		ind_x = O2.O1.M.ind_x
		ind_x_sgu = O2.O1.M.ind_x_sgu

		g_x = O2.O1.g_x
		h_x = O2.O1.h_x
		g_x_sgu = O2.O1.g_x_sgu
		h_x_sgu = O2.O1.h_x_sgu
		f_v = O2.O1.f_v

		O2.Vx0[ind_yp,:] .= g_x*h_x
		O2.Vx0[ind_y,:] .= g_x
		O2.Vx0[ind_xp,:] .= h_x
		O2.Vx0[ind_x,:] .= I(n_x)
		O2.Vx1[ind_yp,:] .= g_x
		O2.Vx1[ind_xp,:] .= I(n_x)

		X = spzeros(n_f, n_x^2)

		Evx2 = kron_pow(O2.Vx0,2) + kron_pow(O2.Vx1,2)*Aζ_k(spI(n_x), 2,  M, Eϵ)
		A2 = O2.f_vv*Evx2

		#Bloc xx

		A = hcat(f_v[ind_f_sgu, ind_xp_sgu] + f_v[ind_f_sgu, ind_yp]*g_x_sgu, f_v[ind_f_sgu, ind_y])
		B = spzeros(n_x_sgu + n_y, n_y + n_x_sgu)
		B[:,n_x_sgu+1:end] .= f_v[ind_f_sgu, ind_yp]
		D = -A2[ind_f_sgu, :]*O2.Z_21
		C = h_x_sgu
		sol_xx = kamenik_solve(Array(A), Array(B), Array(C), D, 2)

		println("Error on sol_xx:")
		println(maximum(abs.(A*sol_xx + B*sol_xx*kron_pow(C, 2) - D)))

		O2.h_xx_sgu .= sparse(sol_xx[1:n_x_sgu,:])
		O2.g_xx_sgu .= sparse(sol_xx[n_x_sgu+1:end,:])

		X[1:n_x_sgu, O2.select_xx] .= O2.h_xx_sgu
		X[n_x+1:end, O2.select_xx] .= O2.g_xx_sgu

		#Bloc σσ

		A[1:n_f_sgu,n_x_sgu+1:end] .= f_v[ind_f_sgu,ind_y] + f_v[ind_f_sgu, ind_yp]

		sol_σσ = -Array(A) \ Array(f_v[ind_f_sgu,ind_yp] * O2.g_xx_sgu * kron_pow(η_sgu, 2) * Eϵ[2] + A2[ind_f_sgu,:] * O2.Z_22)

		O2.h_σσ_sgu .= sparse(sol_σσ[1:n_x_sgu])
		O2.g_σσ_sgu .= sparse(sol_σσ[n_x_sgu+1:end])

		X[1:n_x_sgu,O2.select_σσ] .= O2.h_σσ_sgu
		X[n_x+1:end,O2.select_σσ] .= O2.g_σσ_sgu

		O2.h_xx .= X[1:n_x,:]
		O2.g_xx .= X[n_x+1:end,:]

	end

	function solve_order3!(O3, Eϵ)

		M = O3.O2.O1.M

		n_x = O3.O2.O1.M.n_x
		n_y = O3.O2.O1.M.n_y
		n_x_sgu = O3.O2.O1.M.n_x_sgu
		n_f_sgu = O3.O2.O1.M.n_f_sgu
		n_f = O3.O2.O1.M.n_f
		η_sgu = O3.O2.O1.M.η_sgu

		ind_f_sgu = O3.O2.O1.M.ind_f_sgu
		ind_yp = O3.O2.O1.M.ind_yp
		ind_y = O3.O2.O1.M.ind_y
		ind_xp = O3.O2.O1.M.ind_xp
		ind_xp_sgu = O3.O2.O1.M.ind_xp_sgu
		ind_x = O3.O2.O1.M.ind_x
		ind_x_sgu = O3.O2.O1.M.ind_x_sgu

		g_x = O3.O2.O1.g_x
		h_x = O3.O2.O1.h_x
		g_x_sgu = O3.O2.O1.g_x_sgu
		h_x_sgu = O3.O2.O1.h_x_sgu
		f_v = O3.O2.O1.f_v

		g_xx = O3.O2.g_xx
		h_xx = O3.O2.h_xx
		f_vv = O3.O2.f_vv
		Vx0 = O3.O2.Vx0
		Vx1 = O3.O2.Vx1


		X = spzeros(n_f, n_x^3)

		O3.Vxx0[1:n_y,:] .= g_xx*kron_pow(h_x,2) + g_x*h_xx
	    O3.Vxx0[n_y+1:2*n_y,:] .= g_xx
	    O3.Vxx0[2*n_y+1:2*n_y+n_x,:] .= h_xx
	    O3.Vxx1[1:n_y,:] .= g_xx

	    Evx3 = kron_pow(Vx0,3) + perm_1(Vx1, Vx0, 2, M, Eϵ, sto=true)
	    Eζ2_hx = kron(Aζ_k(spI(n_x), 2, M, Eϵ), h_x)
	    P = kron(spI(n_x), S(n_x, n_x))
	    E_vx_vxx = kron(Vx0,O3.Vxx0) + kron(Vx1,O3.Vxx1)*(Eζ2_hx + P*Eζ2_hx*P) + kron(Vx0, O3.Vxx1)*kron(spI(n_x), Aζ_k(spI(n_x), 2, M, Eϵ))
	    f_yp = f_v[:,ind_yp]
	    A3 = O3.f_vvv*Evx3 + f_vv*E_vx_vxx*O3.Ω₁ + f_yp*g_xx*kron(h_x, h_xx)*O3.Ω₁
	    B3 = kron_pow(h_x, 3) + perm_1(spI(n_x), h_x, 2, M, Eϵ, sto=true)

	    # Bloc xxx
	    A = hcat(f_v[ind_f_sgu,ind_xp_sgu] + f_v[ind_f_sgu,ind_yp]*g_x_sgu, f_v[ind_f_sgu,ind_y])
	    B = zeros(n_x_sgu+n_y,n_y+n_x_sgu)
	    B[:,n_x_sgu+1:end] .= f_v[ind_f_sgu,ind_yp]
	    D = -A3[ind_f_sgu,:]*O3.Z_31
	    C = h_x_sgu
	    sol_xxx = kamenik_solve(Array(A), Array(B), Array(C), D, 3)

		println("Error on sol_xxx :")
		println(maximum(abs.(A*sol_xxx+B*sol_xxx*kron_pow(C,3)-D)))

	    O3.h_xxx_sgu .= sparse(sol_xxx[1:n_x_sgu,:])
	    O3.g_xxx_sgu .= sparse(sol_xxx[n_x_sgu+1:end,:])

	    X[1:n_x_sgu, O3.select_xxx] .= O3.h_xxx_sgu
	    X[n_x+1:end, O3.select_xxx] .= O3.g_xxx_sgu

	    # Bloc xσσ
	    D = -A3[ind_f_sgu,:]*O3.Z_32 - f_v[ind_f_sgu, ind_yp]*O3.g_xxx_sgu*kron(kron_pow(η_sgu, 2)*Eϵ[2], h_x_sgu)

	    sol_xσσ = kamenik_solve(Array(A), Array(B), Array(C), D, 1)

		println("Error on sol_xσσ :")
		println(maximum(abs.(A*sol_xσσ+B*sol_xσσ*C-D)))


	    O3.h_xσσ_sgu .= sparse(sol_xσσ[1:n_x_sgu,:])
	    O3.g_xσσ_sgu .= sparse(sol_xσσ[n_x_sgu+1:end,:])

	    X[1:n_x_sgu, O3.select_xσσ] .= O3.h_xσσ_sgu
	    X[n_x+1:end, O3.select_xσσ] .= O3.g_xσσ_sgu

	    #Bloc σσσ

	    A[1:n_f_sgu,n_x_sgu+1:end] .= A[1:n_f_sgu,n_x_sgu+1:end] + f_v[ind_f_sgu,ind_yp]

	    sol_σσσ = Array(A)\Array(-A3[ind_f_sgu,:]*O3.Z_33)
	    O3.h_σσσ_sgu .= sparse(sol_σσσ[1:n_x_sgu])
	    O3.g_σσσ_sgu .= sparse(sol_σσσ[n_x_sgu+1:end])

	    X[1:n_x_sgu,O3.select_σσσ] .= O3.h_σσσ_sgu
	    X[n_x+1:end,O3.select_σσσ] .= O3.g_σσσ_sgu

	    O3.h_xxx .= X[1:n_x,:]
	    O3.g_xxx .= X[n_x+1:end,:]

	end

	function chainrule(M::Model)

		v = vcat(M.yp, M.y, M.xp, M.x)
		n_v = size(v, 1)
		v_ss = zeros(n_v)

	    f_v = vector_jacobian(sparse_md(M.Eq),v)
	    f_vv = vector_jacobian(f_v,v)
	    f_vvv = vector_jacobian(f_vv,v)

	    f_v_ss = subs_array(f_v, v, v_ss)
	    f_vv_ss = subs_array(f_vv, v, v_ss)
	    f_vvv_ss = subs_array(f_vvv, v, v_ss)

	    f_v_ss_mat = to_matrix_csc(f_v_ss)
	    f_vv_ss_mat = to_matrix_csc(f_vv_ss)
	    f_vvv_ss_mat = to_matrix_csc(f_vvv_ss)

	    return (f_v_ss_mat, f_vv_ss_mat, f_vvv_ss_mat)

	end

	function subs_param(f::SparseMatrixCSC{Basic}, param::Vector{T}, M::Model) where T <:AbstractFloat
	    f0 = subs_array(f, M.params, param)
	    return SparseMatrixCSC(f.m, f.n, f0.colptr, f0.rowval, Float64.(f0.nzval))
	end

	function Aζ_k(A::SparseMatrixCSC, k::Int64, M::Model, Eϵ::Dict)
	    output = spzeros(size(A,1)^k,M.n_x^k)
	    output[:,end] .= kron_pow(A*M.η, k)*Eϵ[k]
	    return output
	end

	function perm_1(A::SparseMatrixCSC, B::SparseMatrixCSC, n::Int64, M::Model, Eϵ::Dict ; sto=false)
	    m_A = A.m
	    n_A = A.n
	    m_B = B.m
	    n_B = B.n
	    dim = m_A^n*n_B
	    if sto
	        base = kron(Aζ_k(A, n, M, Eϵ), B)
	    else
	        base = kron(O1.kron_pow(A, n), B)
	    end
	    P = sum([S(m_A^(n-k),m_A^k*m_B)*base*S(n_A^k*n_B, n_A^(n-k)) for k in 0:n])
	    return P
	end


end
