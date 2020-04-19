module SMC_aux

using Distributed
using SharedArrays
using LinearAlgebra
using SymEngine
using Distributions
using LaTeXStrings

import QuantEcon.solve_discrete_lyapunov

export LinMod, SymMod, init_EstLinMod, likelihood!, init_EstSMC, EstLinMod, EstSMC, solve!, state_space!, logprior

"A structure for linearized models."
struct LinMod{T<: AbstractFloat}
	calibrated::Dict{Symbol,T}
	ss_x::Dict{Symbol,T}
	ss_y::Dict{Symbol,T}
	n_ϵ::Int
	n_param::Int
	param_names::Vector{String}
	set_lin_mod!::Function
	state_space!::Function
end

"A structure for estimated linearized models."
mutable struct EstLinMod{T<: AbstractFloat}
	model::LinMod{T}
	prior::Vector{Distribution}
	Y::Array{T,2}
	bound::Array{T,2}
	param::Vector{T}
	G::Array{T,2}
	R::Array{T,2}
	Z::Array{T,2}
	g_x::Array{T,2}
	h_x::Array{T,2}
	eu::Vector{Bool}
	f_y_tp1::Array{T,2}
	f_y_t::Array{T,2}
	f_x_tp1::Array{T,2}
	f_x_t::Array{T,2}
end

"A structure for SMC estimation."
mutable struct EstSMC{T<: AbstractFloat}
    EstM::EstLinMod{T}
    n_ϕ::Int
    n_part::Int
    λ::T
    c::T
    acpt::T
    trgt::T
    ϕ::Vector{T}
    draws::SharedArray{T}
    weights::SharedArray{T}
    const_vec::Vector{T}
    id::SharedArray{Int}
    loglh::SharedArray{Float64}
    logpost::SharedArray{Float64}
    n_resamp::Int
    μ::Vector{T}
    σ::Vector{T}
    c_vec::Vector{T}
    ESS_vec::Vector{T}
    acpt_vec::Vector{T}
    rsmp_vec::Vector{T}
    acpt_part::SharedArray{T}
end

"Initializes a structure for SMC estimation."
function init_EstSMC(M::EstLinMod; n_ϕ=100, n_part=1000, λ=2., c=0.5, acpt=0.25, trgt=0.25)

	T = eltype(values(M.model.calibrated))

	ϕ = ((0:(n_ϕ-1))/(n_ϕ-1)).^λ

    draws = SharedArray{T}((n_part, M.model.n_param))
    weights   = zeros(n_part, n_ϕ)
    const_vec    = zeros(n_ϕ)
    id = SharedArray{Int64}((n_part))
    loglh   = SharedArray{T}((n_part))
    logpost = SharedArray{T}((n_part))
    n_resamp = 0

    μ = zeros(M.model.n_param)
    σ = zeros(M.model.n_param)

    c_vec    = zeros(n_ϕ)
    ESS_vec  = zeros(n_ϕ)
    acpt_vec = zeros(n_ϕ)
    rsmp_vec = zeros(n_ϕ)
    acpt_part = SharedArray{T}((n_part))

    return EstSMC{T}(M, n_ϕ, n_part, λ, c, acpt, trgt, ϕ, draws, weights, const_vec, id, loglh, logpost, n_resamp, μ, σ, c_vec, ESS_vec, acpt_vec, rsmp_vec, acpt_part)

end

"Initializes an element of the EstLinMod class"
function init_EstLinMod(prior, Y, bound, M::LinMod)

	n_x = length(M.ss_x)
	n_y = length(M.ss_y)
	n_eq = n_x + n_y
	f_x_t = zeros(n_eq,n_x)
	f_x_tp1 = zeros(n_eq,n_x)
	f_y_t = zeros(n_eq,n_y)
	f_y_tp1 = zeros(n_eq,n_y)
	g_x = zeros(n_y, n_x)
	h_x = zeros(n_x, n_x)
	eu = [true,true]

	G = zeros(n_x, n_x)
	R = zeros(n_x, M.n_ϵ)
	Z = zeros(M.n_ϵ, n_x)

	param = zeros(M.n_param)

	return EstLinMod{eltype(G)}(M, prior, Y, bound, param, G, R, Z, g_x, h_x, eu, f_y_tp1, f_y_t, f_x_tp1, f_x_t)

end

"A structure for symbolic models."
struct SymMod{T<: AbstractFloat}
	calibrated::Dict{Symbol,T}
	ss_x::Dict{Symbol,T}
	ss_y::Dict{Symbol,T}
	sym_mod::Function
end

"Solves the model for given parameters."
function solve!(g_x, h_x, eu, f_y_tp1, f_y_t, f_x_tp1, f_x_t, param, M)

	M.set_lin_mod!(f_y_tp1, f_y_t, f_x_tp1, f_x_t, param)

	A = hcat(f_x_tp1, f_y_tp1)
	B = -hcat(f_x_t, f_y_t)

	solve_eig!(g_x, h_x, eu, A, B, M)

end
solve!(M::EstLinMod) = solve!(M.g_x, M.h_x, M.eu, M.f_y_tp1, M.f_y_t, M.f_x_tp1, M.f_x_t, M.param, M.model)
state_space!(M::EstLinMod) = M.model.state_space!(M.G, M.R, M.Z, M.g_x, M.h_x)

"Solves a generalized eigenvalue problem."
function solve_eig!(g_x, h_x, eu, A, B, M::LinMod)

	fill!(g_x, 0.)
	fill!(h_x, 0.)
	fill!(eu, true)

	F = eigen(B, A)
	perm = sortperm(abs.(F.values))
	V = F.vectors[:,perm]
	D = F.values[perm]
	m = findlast(abs.(D) .< 1)

	n_x = length(M.ss_x)
	n_y = length(M.ss_y)

	if m > n_x
		eu[2] = false
		# println("WARNING: the equilibrium is not unique !")
		# println(m)
	elseif m < n_x
		eu[1] = false
		# println("WARNING: the equilibrium does not exist !")
	end


	if all(eu)
		h_x .= real.(V[1:m,1:m]*Diagonal(D)[1:m,1:m]*inv(V[1:m,1:m]))
		g_x .= real.(V[m+1:end,1:m]*inv(V[1:m,1:m]))
	end

end

"Computes the log pdf of prior distributions at given values of parameters."
function logprior(M::EstLinMod)
	prior = [logpdf(M.prior[i],M.param[i]) for i in 1:length(M.param)]
	if any(isnan, prior) | any(isinf, prior)
        flag=false
        lprior=NaN
    else
        flag=true
        lprior=sum(prior)
    end
    return (lprior,flag)
end

"Computes the log posterior, log likelihood of the data and log prior."
function likelihood!(x, ϕ, M::EstLinMod)
    M.param .= x
    outbound = (M.param .< M.bound[:,1]) .| (M.param .> M.bound[:,2])
    if any(outbound)
        lpost = -Inf
        lY = -Inf
        lprior = -Inf
		flag = false
    else
		solve!(M)
		state_space!(M)
    	if !all(M.eu)
			lpost = -Inf
	        lY = -Inf
	        lprior = -Inf
			flag = false
        else
            # Initialize Kalman filter
            T,n = size(M.Y)
        	ss = size(M.G,1)
            pshat = solve_discrete_lyapunov(M.G, M.R*M.R')
        	shat = zeros(ss,1)
            lht = zeros(T)
			flag = true
			i = 1
			while (i<T+1) & (flag)
				shat,pshat,lht[i],_,go_on = kffast(M.Y[i,:], M.Z, shat, pshat, M.G, M.R)
				i += 1
			end
			if flag
	            lY = -((T*n*0.5)*(log(2*pi))) + sum(lht)
	            lprior,_ = logprior(M)
	            lpost = ϕ*lY + lprior
				flag = flag & !(any(isnan.([lpost, lY, lprior])))
			else
				lpost = -Inf
				lY = -Inf
				lprior = -Inf
			end
        end
    end
    return (lpost, lY, lprior, flag)
end

"Carries out an update step of the Kalman filter."
function kffast(y,Z,s,P,T,R)
	#Forecasting
	fors = T*s
	forP = T*P*T'+R*R'
	fory = Z*fors
	forV = Z*forP*Z'
	#Updating
	C = cholesky(Hermitian(forV); check = false)
	if issuccess(C)
		z = C.L\(y-fory)
		x = C.U\z
		M = forP'*Z'
		sqrtinvforV = inv(C.L)
		invforV = sqrtinvforV'*sqrtinvforV
		ups = fors + M*x
		upP = forP - M*invforV*M'
		#log-Likelihood
		lh = -.5*(y - fory)'*invforV*(y - fory) .- sum(log.(diag(C.L)))
		return (ups, upP, lh[1], fory, true)
	else
		return (fors, forP, -Inf, fory, false)
	end
end

end
