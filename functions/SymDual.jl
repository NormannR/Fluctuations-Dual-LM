module SymDual

	using SymEngine
	using NLsolve
	using Distributions
	using Macros
	using EconModel
	using SpecialFunctions
	using LinearAlgebra
	using Macros
	using SparseArrays

	export sym_dual, @spec_funs

	@def spec_funs begin
		Ez = exp(σᶻ^2/2)
		Φ(x) = 0.5*(1+erf(x/sqrt(2)))
		CDF(x) = Φ(log(x)/σᶻ)
		PDF(x) = exp(-0.5*log(x)^2/σᶻ^2)/(x*σᶻ*sqrt(2*pi))
		I(x) = exp(σᶻ^2/2)*Φ((σᶻ^2-log(x))/σᶻ)
		int(x) = I(x) - x*(1-CDF(x))
	end

	"Calibrates the model without indexing of F and b on the wages."
	function calibrate()

		σᶻ = 0.2
		η = 0.6
		σ = 0.6
		β = 0.99
		nfsn = 0.15
		ssξ = 0.6
		ξ = 0.03
		gsY = 0.2
		μfsμ = 0.7
		ρᶠ = 4/9
		ρᵇ = 0.4
		q = 0.7
		ϵ = 6.
		μ = ϵ/(ϵ-1)
		ϕ = 1/μ
		n = 0.67

		u = 1-n
		nf = nfsn*n
		np = n-nf
		s = ssξ*ξ
		δ = ξ*(1-nfsn)*μfsμ/(nfsn*(1-μfsμ))
		e = u+δ*nf+ξ*np
		μᵖ = ξ*np/e
		μᶠ = δ*nf/e

		@spec_funs

		R = ones(1)
		function solve_zp!(R, x)
			R[1] = ξ - s - (1-s)*CDF(x[1])
		end

		sol = nlsolve( solve_zp!, [0.6376100147942086], show_trace=true, ftol=1e-12)
		zp = sol.zero[1]
		F = ρᶠ*ϕ*(η*I(zp)/(1-CDF(zp)) + (1-η)*(zp + β*(1-s)*int(zp)))/(1-ρᶠ*(1-β*(1-s)))

		rU = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp)
		zc = zp + F/ϕ

		function solve_ρ!(R,x)

			ρ = x[1]

			zf = (rU - ρ*ϕ*(1-δ)*β*Ez)/(ρ*ϕ*(1-β*(1-δ)))
			zs = (zc-ρ*zf)/(1-ρ)

			R[1] = μfsμ - (CDF(zs)-CDF(zf))/(1-CDF(zf))

		end

		R = ones(1)
		sol = nlsolve( solve_ρ!, [0.975], show_trace=true, ftol=1e-12, iterations=20)
		ρ = sol.zero[1]

		zf = (rU - ρ*ϕ*(1-δ)*β*Ez)/(ρ*ϕ*(1-β*(1-δ)))
		zs = (zc-ρ*zf)/(1-ρ)
		p = μᵖ/(1-CDF(zs))
		θ = p/q

		m = q*θ^σ
		γ = (1-η)*ϕ*q*(int(zs) + ρ*(int(zf)-int(zs)))
		b = rU - η*γ*θ/(1-η)

		return @save σᶻ β σ η ϵ F b s δ ρ m γ gsY

	end

	@def SS begin

		ϕ = (ϵ-1)/ϵ
	    rU = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp)
	    θ = (rU - b)/(η*γ/(1-η))
	    ξ = s + (1-s)*CDF(zp)

	    zc = zp + F/ϕ
	    zf = (rU - ρ*ϕ*(1-δ)*β*Ez)/(ρ*ϕ*(1-β*(1-δ)))
	    zs = (zc-ρ*zf)/(1-ρ)

	    q = γ/((1-η)*ϕ*(int(max(zs,zc)) + ρ*(int(zf)-int(max(zs,zf)))))

	end

	"Computes the steady state of the model"
	function steady_state(cal::Dict{Symbol,Float64} ; x0 = [0.6376100147942086])

		@load cal σᶻ β σ η ϵ F b s δ ρ m γ gsY

		@spec_funs

		function SS!(R,x)

			zp = x[1]

			@SS

			R[1] = θ - (q/m)^(-1/σ)

		end

		R = ones(1)
		sol = nlsolve( SS!, x0, show_trace=true, ftol=1e-12, iterations=20)
		zp = sol.zero[1]

		@SS

		p = θ*q
		μᵖ = p*(1-CDF(max(zs,zc)))
		μᶠ = p*(CDF(max(zs,zf))-CDF(zf))
		np = δ*μᵖ/(ξ*μᶠ*(1-δ) + δ*μᵖ*(1-ξ) + ξ*δ)
		nf = ξ*μᶠ/(ξ*μᶠ*(1-δ) + δ*μᵖ*(1-ξ) + ξ*δ)
		u = 1-np-nf
		e = u + ξ*np + δ*nf

		R = 1/β
	    v = θ*e
		Y = np*(1-s)*I(zp) + p*e*I(max(zs,zc)) + ρ*(nf*(1-δ)*Ez + p*e*(I(zf)-I(max(zs,zf))))
		g = gsY*Y
		c = Y-g-γ*v
		A = 1.
		π = 1.
		π_s = 1.
		F_A = A
		Δ = 1.
		ϵ_A = 0.
		ϵ_μ = 0.
		ϵ_g = 0.
		ϵ_m = 0.
		ω_A = 0.
		ω_μ = 0.
		ω_g = 0.
		ω_m = 0.

		ss_x = @save np nf R g A Δ ϵ_A ϵ_g ϵ_μ ϵ_m ω_A ω_g ω_m ω_μ
		ss_y = @save c zp zf θ ϕ v Y zs π π_s F_A

		return (ss_x, ss_y)

	end

	"Computes the symbolic system of equilibrium equations"
	function sym_mod(cal, ss_x, ss_y)

		Eq = Vector{SymEngine.Basic}(undef, 32)

		@vars np_t_hat nf_t_hat R_t_hat g_t_hat A_t_hat Δ_t_hat σ_A_t_hat σ_μ_t_hat σ_g_t_hat σ_m_t_hat ϵ_A_tp1 ϵ_μ_tp1 ϵ_g_tp1 ϵ_m_tp1 ω_A_tp1 ω_μ_tp1 ω_g_tp1 ω_m_tp1 χ_tp1
		@vars np_tm1_hat nf_tm1_hat R_tm1_hat g_tm1_hat A_tm1_hat Δ_tm1_hat σ_A_tm1_hat σ_μ_tm1_hat σ_g_tm1_hat σ_m_tm1_hat ϵ_A_t ϵ_μ_t ϵ_g_t ϵ_m_t ω_A_t ω_μ_t ω_g_t ω_m_t χ_t
		@vars c_t_hat zp_t_hat zf_t_hat θ_t_hat ϕ_t_hat v_t_hat Y_t_hat zs_t_hat π_t_hat π_s_t_hat F_A_t_hat P_n_t_hat P_d_t_hat
		@vars c_tp1_hat zp_tp1_hat zf_tp1_hat θ_tp1_hat ϕ_tp1_hat v_tp1_hat Y_tp1_hat zs_tp1_hat π_tp1_hat π_s_tp1_hat F_A_tp1_hat P_n_tp1_hat P_d_tp1_hat
		@vars ρ_A ρ_g ρ_R ρ_π ρ_y ψ σ_A σ_μ σ_g σ_m λ_A λ_μ λ_g λ_m τ_A τ_μ τ_g τ_m

		@load cal σᶻ β σ η ϵ F b s δ ρ m γ gsY
		@load_cat ss ss_x np nf R g A Δ ϵ_A ϵ_μ ϵ_g ϵ_m ω_A ω_μ ω_g ω_m
		@load_cat ss ss_y c zp zf θ ϕ v Y zs π π_s F_A

		P_n_ss = Y_ss*ϕ_ss/(1-β*ψ)
		P_d_ss = Y_ss/(1-β*ψ)

		np_t = np_ss*exp(np_t_hat)
		nf_t = nf_ss*exp(nf_t_hat)
		R_t = R_ss*exp(R_t_hat)
		g_t = g_ss*exp(g_t_hat)
		A_t = A_ss*exp(A_t_hat)
		Δ_t = Δ_ss*exp(Δ_t_hat)
		np_tm1 = np_ss*exp(np_tm1_hat)
		nf_tm1 = nf_ss*exp(nf_tm1_hat)
		R_tm1 = R_ss*exp(R_tm1_hat)
		g_tm1 = g_ss*exp(g_tm1_hat)
		A_tm1 = A_ss*exp(A_tm1_hat)
		Δ_tm1 = Δ_ss*exp(Δ_tm1_hat)
		c_t = c_ss*exp(c_t_hat)
		zp_t = zp_ss*exp(zp_t_hat)
		zf_t = zf_ss*exp(zf_t_hat)
		θ_t = θ_ss*exp(θ_t_hat)
		ϕ_t = ϕ_ss*exp(ϕ_t_hat)
		v_t = v_ss*exp(v_t_hat)
		Y_t = Y_ss*exp(Y_t_hat)
		zs_t = zs_ss*exp(zs_t_hat)
		π_t = π_ss*exp(π_t_hat)
		π_s_t = π_s_ss*exp(π_s_t_hat)
		F_A_t = F_A_ss*exp(F_A_t_hat)
		c_tp1 = c_ss*exp(c_tp1_hat)
		zp_tp1 = zp_ss*exp(zp_tp1_hat)
		zf_tp1 = zf_ss*exp(zf_tp1_hat)
		θ_tp1 = θ_ss*exp(θ_tp1_hat)
		ϕ_tp1 = ϕ_ss*exp(ϕ_tp1_hat)
		v_tp1 = v_ss*exp(v_tp1_hat)
		Y_tp1 = Y_ss*exp(Y_tp1_hat)
		zs_tp1 = zs_ss*exp(zs_tp1_hat)
		π_tp1 = π_ss*exp(π_tp1_hat)
		π_s_tp1 = π_s_ss*exp(π_s_tp1_hat)
		F_A_tp1 = F_A_ss*exp(F_A_tp1_hat)
		σ_g_tm1 = σ_g*exp(σ_g_tm1_hat)
		σ_A_tm1 = σ_A*exp(σ_A_tm1_hat)
		σ_μ_tm1 = σ_μ*exp(σ_μ_tm1_hat)
		σ_m_tm1 = σ_m*exp(σ_m_tm1_hat)
		σ_g_t = σ_g*exp(σ_g_t_hat)
		σ_A_t = σ_A*exp(σ_A_t_hat)
		σ_μ_t = σ_μ*exp(σ_μ_t_hat)
		σ_m_t = σ_m*exp(σ_m_t_hat)
		P_n_t = P_n_ss*exp(P_n_t_hat)
		P_n_tp1 = P_n_ss*exp(P_n_tp1_hat)
		P_d_t = P_d_ss*exp(P_d_t_hat)
		P_d_tp1 = P_d_ss*exp(P_d_tp1_hat)

		@spec_funs

		ξ_t = s + (1-s)*CDF(zp_t)
		q_t = m*θ_t^(-σ)
		β_t_tp1 = β*c_t/c_tp1

		Eq[1] = -c_ss*(1/c_t - β*R_t/(π_tp1*c_tp1))

		Eq[2] = log(A_t) - ρ_A*log(A_tm1) - σ_A_t*ϵ_A_t

		Eq[3] = log(g_t/g_ss) - ρ_g*log(g_tm1/g_ss) - σ_g_t*ϵ_g_t

		Eq[4] = log(R_t/R_ss) - ρ_R*log(R_tm1/R_ss) - (1-ρ_R)*(ρ_π*log(π_tp1) + ρ_y*log(Y_t/Y_ss)) - σ_m_t*ϵ_m_t

		Eq[5] = log(σ_A_t/σ_A) - λ_A*log(σ_A_tm1/σ_A) - τ_A*ω_A_t

		Eq[6] = log(σ_μ_t/σ_μ) - λ_μ*log(σ_μ_tm1/σ_A) - τ_μ*ω_μ_t

		Eq[7] = log(σ_g_t/σ_g) - λ_g*log(σ_g_tm1/σ_g) - τ_g*ω_g_t

		Eq[8] = log(σ_m_t/σ_m) - λ_m*log(σ_m_tm1/σ_m) - τ_m*ω_m_t

		Eq[9] = (Y_t - c_t - g_t - γ*v_t)/Y_ss

		Eq[10] = (v_t - θ_t*(1-(1-ξ_t)*np_tm1-(1-δ)*nf_tm1))/v_ss

		Eq[11] = (np_t - (1-ξ_t)*np_tm1 - v_t*q_t*(1-CDF(zs_t)))/np_ss

		Eq[12] = (nf_t - (1-δ)*nf_tm1 - v_t*q_t*(CDF(zs_t)-CDF(zf_t)))/nf_ss

		Eq[13] = (Y_t*Δ_t - ( (1-s)*A_t*I(zp_t)*np_tm1 + ρ*(1-δ)*Ez*A_t*nf_tm1 + A_t*v_t*q_t*I(zs_t) + ρ*q_t*v_t*A_t*(I(zf_t)-I(zs_t)) ))/Y_ss

		Eq[14] = (1-ρ)*zs_t - zp_t - F/(A_t*ϕ_t) + ρ*zf_t

		Eq[15] =  γ/((1-η)*A_t*ϕ_t*q_t) - ((1-ρ)*int(zs_t) + ρ*int(zf_t))

		Eq[16] = A_t*ϕ_t*zp_t - b + F - β_t_tp1*(1-s)*F + β_t_tp1*F_A_tp1*ϕ_tp1*(1-s)*int(zp_tp1) - η*γ*θ_t/(1-η)

		Eq[17] = ρ*A_t*ϕ_t*zf_t - b + ρ*β_t_tp1*(1-δ)*F_A_tp1*ϕ_tp1*(Ez-zf_tp1) - η*γ*θ_t/(1-η)

		Eq[18] = A_t_hat - F_A_t_hat

		Eq[19] = π_t - (ψ + (1-ψ)*π_s_t^(1-ϵ))^(1/(1-ϵ)) - σ_μ_t*ϵ_μ_t

	    Eq[20] = P_n_t - Y_t*ϕ_t - β_t_tp1*ψ*π_tp1^ϵ*P_n_tp1

	    Eq[21] = P_d_t - Y_t - β_t_tp1*ψ*π_tp1^(ϵ-1)*P_d_tp1

	    Eq[22] = π_s_t - π_t*(ϵ/(ϵ-1))*P_n_t/P_d_t

  		Eq[23] = Δ_t - (1-ψ)*(π_s_t/π_t)^(-ϵ) - ψ*Δ_tm1*π_t^ϵ

		Eq[24] = ϵ_A_tp1

		Eq[25] = ϵ_μ_tp1

		Eq[26] = ϵ_g_tp1

		Eq[27] = ϵ_m_tp1

		Eq[28] = ω_A_tp1

		Eq[29] = ω_μ_tp1

		Eq[30] = ω_g_tp1

		Eq[31] = ω_m_tp1

		Eq[32] = χ_tp1 - χ_t

		xp = [np_t_hat, nf_t_hat, R_t_hat, g_t_hat, A_t_hat, Δ_t_hat, σ_A_t_hat, σ_μ_t_hat, σ_g_t_hat, σ_m_t_hat, ϵ_A_tp1, ϵ_μ_tp1, ϵ_g_tp1, ϵ_m_tp1, ω_A_tp1, ω_μ_tp1, ω_g_tp1, ω_m_tp1, χ_tp1]
		x = [np_tm1_hat, nf_tm1_hat, R_tm1_hat, g_tm1_hat, A_tm1_hat, Δ_tm1_hat, σ_A_tm1_hat, σ_μ_tm1_hat, σ_g_tm1_hat, σ_m_tm1_hat, ϵ_A_t, ϵ_μ_t, ϵ_g_t, ϵ_m_t, ω_A_t, ω_μ_t, ω_g_t, ω_m_t, χ_t]
		yp = [c_tp1_hat, zp_tp1_hat, zf_tp1_hat, θ_tp1_hat, ϕ_tp1_hat, v_tp1_hat, Y_tp1_hat, zs_tp1_hat, π_tp1_hat, π_s_tp1_hat, F_A_tp1_hat, P_n_tp1_hat, P_d_tp1_hat]
		y = [c_t_hat, zp_t_hat, zf_t_hat, θ_t_hat, ϕ_t_hat, v_t_hat, Y_t_hat, zs_t_hat, π_t_hat, π_s_t_hat, F_A_t_hat, P_n_t_hat, P_d_t_hat]
		param = [ρ_A, ρ_g, ρ_R, ρ_π, ρ_y, ψ, σ_A, σ_μ, σ_g, σ_m, λ_A, λ_μ, λ_g, λ_m, τ_A, τ_μ, τ_g, τ_m]
		ϵ = [ϵ_A_t, ϵ_μ_t, ϵ_g_t, ϵ_m_t, ω_A_t, ω_μ_t, ω_g_t, ω_m_t]

		n_x = size(xp, 1)
		ζ = spzeros(Basic, n_x, n_x)
		ζ[11:18, end] = ϵ

		return (xp, x, yp, y, param, ζ, Eq)

	end

	function sym_dual()

		cal = calibrate()
		ss_x, ss_y = steady_state(cal)
		xp, x, yp, y, params, ζ, Eq = sym_mod(cal, ss_x, ss_y)

		n_x = length(xp)
		n_y = length(yp)

		n_f = n_x+n_y
		n_x1 = size(1:10,1)
		n_x2 = 0
		n_ϵ = size(11:18,1)
		n_x_sgu = n_x1+n_ϵ
		n_f_sgu = n_f-1

		η = spzeros(n_x,n_ϵ)
		η[11:18,:] .= I(n_ϵ)
		η_sgu = η[1:n_x_sgu, :]

		ind_yp = 1:n_y
		ind_y = n_y+1:2*n_y
		ind_xp = 2*n_y+1:2*n_y+n_x
		ind_x = 2*n_y+n_x+1:2*n_y+2*n_x
		ind_xp_sgu = ind_xp[1:end-1]
		ind_x_sgu = ind_x[1:end-1]
		ind_f_sgu = 1:n_f_sgu

		return Model( cal, ss_y, ss_x, params, yp, y, xp, x, Eq, ζ, η, η_sgu, n_f, n_f_sgu, n_x, n_x_sgu, n_y, ind_f_sgu, ind_yp, ind_y, ind_xp, ind_xp_sgu, ind_x, ind_x_sgu )

	end

end
