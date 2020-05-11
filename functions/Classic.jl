module Classic

using SymEngine
using NLsolve
using Distributions
using Macros
using SMC_aux
using SpecialFunctions
using LinearAlgebra

export lin_classic, sym_classic, @spec_funs

@def spec_funs begin
	Ez = exp(σᶻ^2/2)
	Φ(x) = 0.5*(1+erf(x/sqrt(2)))
	CDF(x) = Φ(log(x)/σᶻ)
	PDF(x) = exp(-0.5*log(x)^2/σᶻ^2)/(x*σᶻ*sqrt(2*pi))
	I(x) = exp(σᶻ^2/2)*Φ((σᶻ^2-log(x))/σᶻ)
	int(x) = I(x) - x*(1-CDF(x))
end
# %%
"Calibrates the model"
function calibrate()

	σᶻ = 0.2
	η = 0.6
	σ = 0.6
	β = 0.99
	μᵖ = 0.25
	ssξ = 0.5
	gsY = 0.2
	ρᶠ = 4/9
	q = 0.9
	ρᵇ = 0.4
	ϵ = 6
	μ = ϵ/(ϵ-1)
	ϕ = 1/μ
	n = 0.9

	u = 1-n
	e = u/(1-μᵖ)
	ξ = μᵖ*e/n
	s = ssξ*ξ

	@spec_funs

	R = ones(1)
	function solve_zp!(R, x)
		R[1] = ξ - s - (1-s)*CDF(x[1])
	end

	sol = nlsolve( solve_zp!, [0.6599821461285086], show_trace=true, ftol=1e-12)
	zp = sol.zero[1]
	F = ρᶠ*ϕ*(η*I(zp)/(1-CDF(zp)) + (1-η)*(zp + β*(1-s)*int(zp)))/(1-ρᶠ*(1-β*(1-s)))

	rU = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp)
	zc = zp + F/ϕ

	p = μᵖ/(1-CDF(zc))
	θ = p/q

	m = q*θ^σ
	γ = (1-η)*ϕ*q*int(zc)
	b = rU - η*γ*θ/(1-η)
	wp = η*ϕ*((1-ξ)*I(zp)/(1-CDF(zp)) + ξ*I(zc)/(1-CDF(zc))) - η*β*(1-s)*F + (1-ξ)*η*F + (1-η)*rU
	h = b - ρᵇ*wp

	return @save σᶻ β σ η ϵ F b s m γ gsY

end
# %%
@def SS begin

	ϕ = (ϵ-1)/ϵ
    rU = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp)
    θ = (rU - b)/(η*γ/(1-η))
    ξ = s + (1-s)*CDF(zp)

    zc = zp + F/ϕ
	q = γ/((1-η)*ϕ*int(zc))

end
# %%
"Computes the steady state of the model"
function steady_state(cal::Dict{Symbol,Float64})

	@load cal σᶻ β σ η ϵ F b s m γ gsY

	@spec_funs

	function SS!(R,x)

		zp = x[1]

		@SS

	    R[1] = θ - (q/m)^(-1/σ)

	end

	R = ones(1)
	sol = nlsolve( SS!, [0.6599821461285086], show_trace=true, ftol=1e-12, iterations=20)
	zp = sol.zero[1]

	@SS

	p = θ*q
	μ = p*(1-CDF(zc))
	np = μ/(ξ + (1-ξ)*μ)
	u = 1-np
	R = 1/β
	e = u+ξ*np
    v = θ*e
	Y = np*((1-ξ)*I(zp)/(1-CDF(zp)) + ξ*I(zc)/(1-CDF(zc)))
	g = gsY*Y
	c = Y-g-γ*v
	A = 1
	π = 1
	F_A = A
	ϵ_A = 0
	ϵ_μ = 0
	ϵ_g = 0
	ϵ_m = 0

	ss_x = @save np R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	ss_y = @save c zp θ ϕ v Y zc π F_A

	return (ss_x, ss_y)

end
# %%
"Computes the log-linearized system of equilibrium equations"
function set_lin_mod!(f_y_tp1, f_y_t, f_x_tp1, f_x_t, param, cal, ss_x, ss_y)

	fill!(f_y_tp1, 0.)
	fill!(f_y_t, 0.)
	fill!(f_x_tp1, 0.)
	fill!(f_x_t, 0.)

	@load cal σᶻ β σ η ϵ F b s m γ gsY
	@spec_funs
	@load_vec param ρ_A ρ_g ρ_R ρ_π ρ_y ψ σ_A σ_μ σ_g σ_m
	@index np R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	@index c zp θ ϕ v Y zc π F_A
	@load_cat ss ss_x np R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	@load_cat ss ss_y c zp θ ϕ v Y zc π F_A

	#Euler equation

	f_y_t[1,c] = 1.
	f_y_tp1[1,c] = -1.
	f_x_tp1[1,R] = 1.
	f_y_tp1[1,π] = -1.

	# Phillips curve

	f_y_t[2,π] = 1.
	f_y_tp1[2,π] = -β
	f_y_t[2,ϕ] = -(1-β*ψ)*(1-ψ)/ψ
	f_x_t[2,ϵ_μ] = -σ_μ

	# Productivity shock

	f_x_tp1[3,A] = 1.
	f_x_t[3,A] = -ρ_A
	f_x_t[3,ϵ_A] = -σ_A

	# Government spending shock

	f_x_tp1[4,g] = 1.
	f_x_t[4,g] = -ρ_g
	f_x_t[4,ϵ_g] = -σ_g

	# Interest rate rule

	f_x_tp1[5,R] = 1.
	f_x_t[5,R] = -ρ_R
	f_y_tp1[5,π] = -(1-ρ_R)*ρ_π
	f_y_t[5,Y] = -(1-ρ_R)*ρ_y
	f_x_t[5,ϵ_m] = -σ_m

	# Ressource constraint

	f_y_t[6,Y] = 1
	f_y_t[6,c] = -c_ss/Y_ss
	f_x_tp1[6,g] = -g_ss/Y_ss
	f_y_t[6,v] = -γ*v_ss/Y_ss

	#Definition of v

	f_y_t[7,v] = 1.
	f_y_t[7,θ] = -1.
	f_x_t[7,np] = +(1-s)*(1-CDF(zp_ss))*θ_ss*np_ss/v_ss
	f_y_t[7,zp] = -(1-s)*θ_ss*PDF(zp_ss)*zp_ss*np_ss/v_ss

	#Permanent employment

	f_x_tp1[8,np] = 1
	f_x_t[8,np] = -(1-s)*(1-CDF(zp_ss))
	f_y_t[8,zp] = (1-s)*zp_ss*PDF(zp_ss)
	f_y_t[8,v] = -(1-CDF(zc_ss))*m*θ_ss^(-σ)*v_ss/np_ss
	f_y_t[8,θ] = σ*(1-CDF(zc_ss))*m*θ_ss^(-σ)*v_ss/np_ss
	f_y_t[8,zc] = zc_ss*PDF(zc_ss)*m*θ_ss^(-σ)*v_ss/np_ss

	#Production function

	f_y_t[9,Y] = 1.
	f_x_tp1[9,A] = -1.
	f_x_t[9,np] = -(1-s)*I(zp_ss)*np_ss/Y_ss
	f_y_t[9,zp] = (1-s)*np_ss*zp_ss^2*PDF(zp_ss)/Y_ss
	f_y_t[9,zc] = zc_ss^2*PDF(zc_ss)*v_ss*m*θ_ss^(-σ)/Y_ss
	f_y_t[9,v] = -I(zc_ss)*v_ss*m*θ_ss^(-σ)/Y_ss
	f_y_t[9,θ] = σ*I(zc_ss)*v_ss*m*θ_ss^(-σ)/Y_ss

	#Definition of zs

	f_y_t[10,zc] = zc_ss
	f_y_t[10,zp] = -zp_ss
	f_y_t[10,ϕ] = F/ϕ_ss
	f_x_tp1[10,A] = F/ϕ_ss

	#Job creation condition

	f_x_tp1[11,A] = -γ/((1-η)*ϕ_ss*m*θ_ss^(-σ))
	f_y_t[11,ϕ] = -γ/((1-η)*ϕ_ss*m*θ_ss^(-σ))
	f_y_t[11,θ] = σ*γ/((1-η)*ϕ_ss*m*θ_ss^(-σ))
	f_y_t[11,zc] = (1-CDF(zc_ss))*zc_ss

	#Definition of zp

	f_x_tp1[12,A] = A_ss*ϕ_ss*zp_ss
	f_y_t[12,zp] = A_ss*ϕ_ss*zp_ss
	f_y_t[12,ϕ] = A_ss*ϕ_ss*zp_ss
	f_y_t[12,c] = β*(1-s)*(A_ss*ϕ_ss*int(zp_ss) - F)
	f_y_tp1[12,c] = -f_y_t[12,c]
	f_y_tp1[12,F_A] = β*(1-s)*A_ss*ϕ_ss*int(zp_ss)
	f_y_tp1[12,ϕ] = f_y_tp1[12,F_A]
	f_y_tp1[12,zp] = -β*(1-s)*(1-CDF(zp_ss))*A_ss*ϕ_ss*zp_ss
	f_y_t[12,θ] = -η*γ*θ_ss/(1-η)

	#Relationship between F_A in controls and A in states

	f_x_tp1[13,A] = 1.
	f_y_t[13,F_A] = -1.

	#Shocks

	f_x_tp1[14,ϵ_A] = 1.
	f_x_tp1[15,ϵ_g] = 1.
	f_x_tp1[16,ϵ_μ] = 1.
	f_x_tp1[17,ϵ_m] = 1.


end
# %%
function state_space!(G, R, Z, g_x, h_x, ss_x)
	G .= h_x
	R[end-4+1:end,:] .= I(4)

	@index np R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	@index c zp θ ϕ v Y zc π F_A

	Z[1,:] .= g_x[Y, :]
	Z[2,:] .= g_x[π, :]
	Z[3,:] .= h_x[R, :]
	Z[4,:] .= h_x[np, :]
end
# %%
"Computes the symbolic system of equilibrium equations"
function sym_mod(param, cal, ss_x, ss_y)

	n_x = length(ss_x)
	n_y = length(ss_y)

	Eq = Vector{SymEngine.Basic}(undef,n_x+n_y)

	@vars np_t_hat R_t_hat g_t_hat A_t_hat ϵ_A_tp1 ϵ_μ_tp1 ϵ_g_tp1 ϵ_m_tp1
	@vars np_tm1_hat R_tm1_hat g_tm1_hat A_tm1_hat ϵ_A_t ϵ_μ_t ϵ_g_t ϵ_m_t
	@vars c_t_hat zp_t_hat θ_t_hat ϕ_t_hat v_t_hat Y_t_hat zc_t_hat π_t_hat F_A_t_hat
	@vars c_tp1_hat zp_tp1_hat θ_tp1_hat ϕ_tp1_hat v_tp1_hat Y_tp1_hat zc_tp1_hat π_tp1_hat F_A_tp1_hat

	@load cal σᶻ β σ η ϵ ρᶠ h ρᵇ s m γ gsY
	@load_vec param ρ_A ρ_g ρ_R ρ_π ρ_y ψ σ_A σ_μ σ_g σ_m
	@load_cat ss ss_x np R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	@load_cat ss ss_y c zp θ ϕ F b wp v Y zc π F_A

	np_t = np_ss*exp(np_t_hat)
	R_t = R_ss*exp(R_t_hat)
	g_t = g_ss*exp(g_t_hat)
	A_t = A_ss*exp(A_t_hat)
	np_tm1 = np_ss*exp(np_tm1_hat)
	R_tm1 = R_ss*exp(R_tm1_hat)
	g_tm1 = g_ss*exp(g_tm1_hat)
	A_tm1 = A_ss*exp(A_tm1_hat)
	c_t = c_ss*exp(c_t_hat)
	zp_t = zp_ss*exp(zp_t_hat)
	θ_t = θ_ss*exp(θ_t_hat)
	ϕ_t = ϕ_ss*exp(ϕ_t_hat)
	v_t = v_ss*exp(v_t_hat)
	Y_t = Y_ss*exp(Y_t_hat)
	zc_t = zc_ss*exp(zc_t_hat)
	π_t = π_ss*exp(π_t_hat)
	F_A_t = F_A_ss*exp(F_A_t_hat)
	c_tp1 = c_ss*exp(c_tp1_hat)
	zp_tp1 = zp_ss*exp(zp_tp1_hat)
	θ_tp1 = θ_ss*exp(θ_tp1_hat)
	ϕ_tp1 = ϕ_ss*exp(ϕ_tp1_hat)
	v_tp1 = v_ss*exp(v_tp1_hat)
	Y_tp1 = Y_ss*exp(Y_tp1_hat)
	zc_tp1 = zc_ss*exp(zc_tp1_hat)
	π_tp1 = π_ss*exp(π_tp1_hat)
	F_A_tp1 = F_A_ss*exp(F_A_tp1_hat)

	@spec_funs

	ξ_t = s + (1-s)*CDF(zp_t)
	q_t = m*θ_t^(-σ)
	β_t_tp1 = β*c_t/c_tp1

	Eq[1] = -c_ss*(1/c_t - β*R_t/(π_tp1*c_tp1))

	Eq[2] = π_t_hat - β*π_tp1_hat - (1-β*ψ)*(1-ψ)*ϕ_t_hat/ψ - σ_μ*ϵ_μ_t

	Eq[3] = log(A_t) - ρ_A*log(A_tm1) - σ_A*ϵ_A_t

	Eq[4] = log(g_t/g_ss) - ρ_g*log(g_tm1/g_ss) - σ_g*ϵ_g_t

	Eq[5] = log(R_t/R_ss) - ρ_R*log(R_tm1/R_ss) - (1-ρ_R)*(ρ_π*log(π_tp1) + ρ_y*log(Y_t/Y_ss)) - σ_m*ϵ_m_t

	Eq[6] = (Y_t - c_t - g_t - γ*v_t)/Y_ss

	Eq[7] = (v_t - θ_t*(1-(1-ξ_t)*np_tm1))/v_ss

	Eq[8] = (np_t - (1-ξ_t)*np_tm1 - v_t*q_t*(1-CDF(zc_t)))/np_ss

	Eq[9] = (Y_t - ( (1-s)*A_t*I(zp_t)*np_tm1 + A_t*v_t*q_t*I(zc_t) ))/Y_ss

	Eq[10] = zc_t - zp_t - F_t/(A_t*ϕ_t)

	Eq[11] =  γ/((1-η)*A_t*ϕ_t*q_t) - int(zc_t)

	Eq[12] = A_t*ϕ_t*zp_t - b_t + F_t - β_t_tp1*(1-s)*F_tp1 + β_t_tp1*F_A_tp1*ϕ_tp1*(1-s)*int(zp_tp1) - η*γ*θ_t/(1-η)

	Eq[13] = A_t_hat - F_A_t_hat

	Eq[14] = ϵ_A_tp1

	Eq[15] = ϵ_g_tp1

	Eq[16] = ϵ_μ_tp1

	Eq[17] = ϵ_m_tp1

	xp = [np_t_hat, R_t_hat, g_t_hat, A_t_hat, ϵ_A_tp1, ϵ_μ_tp1, ϵ_g_tp1, ϵ_m_tp1]
	x = [np_tm1_hat, R_tm1_hat, g_tm1_hat, A_tm1_hat, ϵ_A_t, ϵ_μ_t, ϵ_g_t, ϵ_m_t]
	yp = [c_tp1_hat, zp_tp1_hat, θ_tp1_hat, ϕ_tp1_hat, v_tp1_hat, Y_tp1_hat, zc_tp1_hat, π_tp1_hat, F_A_tp1_hat]
	y = [c_t_hat, zp_t_hat, θ_t_hat, ϕ_t_hat, v_t_hat, Y_t_hat, zc_t_hat, π_t_hat, F_A_t_hat]

	return (xp, x, yp, y, Eq)

end
# %%
"Defines the linear model with a classic labor market"
function lin_classic()
	cal = calibrate()
	ss_x, ss_y = steady_state(cal)
	param_names = ["ρ_A", "ρ_g", "ρ_R", "ρ_π", "ρ_y", "ψ", "σ_A", "σ_μ", "σ_g", "σ_m"]
	return LinMod{eltype(values(cal))}(cal, ss_x, ss_y, 4, 10, param_names, (f_y_tp1, f_y_t, f_x_tp1, f_x_t, param)->set_lin_mod!(f_y_tp1, f_y_t, f_x_tp1, f_x_t, param, cal, ss_x, ss_y), (G, R, Z, g_x, h_x)->state_space!(G, R, Z, g_x, h_x, ss_x))
end
# %%
"Defines the symbolic model with a Classic Labor Market"
function sym_classic()
	cal = calibrate()
	ss_x, ss_y = steady_state(cal)
	return SymMod{eltype(values(cal))}(cal, ss_x, ss_y, param -> sym_mod(param, cal, ss_x, ss_y))
end

end
