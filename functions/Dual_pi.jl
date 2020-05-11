module Dual_pi

using SymEngine
using NLsolve
using Distributions
using Macros
using SMC_aux
using SpecialFunctions
using LinearAlgebra

export lin_dual, sym_dual, @spec_funs, steady_state, calibrate

@def spec_funs begin
	Ez = exp(σᶻ^2/2)
	Φ(x) = 0.5*(1+erf(x/sqrt(2)))
	CDF(x) = Φ(log(max(x,0.))/σᶻ)
	PDF(x) = exp(-0.5*log(max(x,0.))^2/σᶻ^2)/(x*σᶻ*sqrt(2*pi))
	I(x) = exp(σᶻ^2/2)*Φ((σᶻ^2-log(max(x,0.)))/σᶻ)
	int(x) = I(x) - x*(1-CDF(x))
end
# %%
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
	sol = nlsolve( solve_ρ!, [0.97870543587480888], show_trace=true, ftol=1e-12, iterations=20)
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
# %%
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
# %%
"Computes the steady state of the model"
function steady_state(cal::Dict{Symbol,Float64} ; x0 = [0.68])

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
	A = 1
	π = 1
	F_A = A
	ϵ_A = 0
	ϵ_μ = 0
	ϵ_g = 0
	ϵ_m = 0

	hc = 0
	subsp = 0
	Πᶠ = 0

	ss_x = @save np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	ss_y = @save c zp zf θ ϕ v Y zs π F_A zc hc subsp Πᶠ

	return (ss_x, ss_y)

end
# %%
"Computes the log-linearized system of equilibrium equations"
function set_lin_mod!(f_y_tp1, f_y_t, f_x_tp1, f_x_t, param, cal, ss_x, ss_y)

	fill!(f_y_tp1, 0.)
	fill!(f_y_t, 0.)
	fill!(f_x_tp1, 0.)
	fill!(f_x_t, 0.)

	@load cal σᶻ β σ η ϵ F b s δ ρ m γ gsY
	@spec_funs
	@load_vec param ρ_A ρ_g ρ_R ρ_π ρ_y ψ σ_A σ_μ σ_g σ_m
	@index np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	@index c zp zf θ ϕ v Y zs π F_A zc hc subsp Πᶠ
	@load_cat ss ss_x np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	@load_cat ss ss_y c zp zf θ ϕ v Y zs π F_A zc hc subsp Πᶠ

	κ = (1-β*ψ)*(1-ψ)/ψ
	q_ss = m*θ_ss^(-σ)
	μᵖ_ss = q_ss*(1-CDF(zs_ss))
	μᶠ_ss = q_ss*(CDF(zs_ss)-CDF(zf_ss))

	#Euler equation

	f_y_t[1,c] = 1.
	f_y_tp1[1,c] = -1.
	f_x_tp1[1,R] = 1.
	f_y_tp1[1,π] = -1.

	# Phillips curve

	f_y_t[2,π] = 1.
	f_y_tp1[2,π] = -β
	f_y_t[2,ϕ] = -κ
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
	f_x_t[7,nf] = +(1-δ)*θ_ss*nf_ss/v_ss
	f_y_t[7,zp] = -(1-s)*θ_ss*PDF(zp_ss)*zp_ss*np_ss/v_ss

	#Permanent employment

	f_x_tp1[8,np] = 1
	f_x_t[8,np] = -(1-s)*(1-CDF(zp_ss))
	f_y_t[8,zp] = (1-s)*zp_ss*PDF(zp_ss)
	f_y_t[8,v] = -(1-CDF(zs_ss))*m*θ_ss^(-σ)*v_ss/np_ss
	f_y_t[8,θ] = σ*(1-CDF(zs_ss))*m*θ_ss^(-σ)*v_ss/np_ss
	f_y_t[8,zs] = zs_ss*PDF(zs_ss)*m*θ_ss^(-σ)*v_ss/np_ss

	#Temporary employment

	f_x_tp1[9,nf] = 1
	f_x_t[9,nf] = -(1-δ)
	f_y_t[9,v] = -(CDF(zs_ss)-CDF(zf_ss))*m*θ_ss^(-σ)*v_ss/nf_ss
	f_y_t[9,θ] = σ*(CDF(zs_ss)-CDF(zf_ss))*m*θ_ss^(-σ)*v_ss/nf_ss
	f_y_t[9,zs] = -zs_ss*PDF(zs_ss)*m*θ_ss^(-σ)*v_ss/nf_ss
	f_y_t[9,zf] = zf_ss*PDF(zf_ss)*m*θ_ss^(-σ)*v_ss/nf_ss

	#Production function

	f_y_t[10,Y] = 1.
	f_x_tp1[10,A] = -1.
	f_x_t[10,np] = -(1-s)*I(zp_ss)*np_ss/Y_ss
	f_x_t[10,nf] = -ρ*(1-δ)*Ez*nf_ss/Y_ss
	f_y_t[10,zp] = (1-s)*np_ss*zp_ss^2*PDF(zp_ss)/Y_ss
	f_y_t[10,zs] = (1-ρ)*zs_ss^2*PDF(zs_ss)*v_ss*m*θ_ss^(-σ)/Y_ss
	f_y_t[10,zf] = ρ*zf_ss^2*PDF(zf_ss)*v_ss*m*θ_ss^(-σ)/Y_ss
	f_y_t[10,v] = -(I(zs_ss) + ρ*(I(zf_ss) - I(zs_ss)))*v_ss*m*θ_ss^(-σ)/Y_ss
	f_y_t[10,θ] = σ*(I(zs_ss) + ρ*(I(zf_ss) - I(zs_ss)))*v_ss*m*θ_ss^(-σ)/Y_ss

	#Definition of zs

	f_y_t[11,zs] = (1-ρ)*zs_ss
	f_y_t[11,zp] = -zp_ss
	f_y_t[11,zf] = ρ*zf_ss
	f_y_t[11,ϕ] = F/ϕ_ss
	f_x_tp1[11,A] = F/ϕ_ss

	#Job creation condition

	f_x_tp1[12,A] = -γ/((1-η)*ϕ_ss*m*θ_ss^(-σ))
	f_y_t[12,ϕ] = -γ/((1-η)*ϕ_ss*m*θ_ss^(-σ))
	f_y_t[12,θ] = σ*γ/((1-η)*ϕ_ss*m*θ_ss^(-σ))
	f_y_t[12,zs] = (1-ρ)*(1-CDF(zs_ss))*zs_ss
	f_y_t[12,zf] = ρ*(1-CDF(zf_ss))*zf_ss

	#Definition of zp

	f_x_tp1[13,A] = A_ss*ϕ_ss*zp_ss
	f_y_t[13,zp] = A_ss*ϕ_ss*zp_ss
	f_y_t[13,ϕ] = A_ss*ϕ_ss*zp_ss
	f_y_t[13,c] = β*(1-s)*(A_ss*ϕ_ss*int(zp_ss) - F)
	f_y_tp1[13,c] = -f_y_t[13,c]
	f_y_tp1[13,F_A] = β*(1-s)*A_ss*ϕ_ss*int(zp_ss)
	f_y_tp1[13,ϕ] = f_y_tp1[13,F_A]
	f_y_tp1[13,zp] = -β*(1-s)*(1-CDF(zp_ss))*A_ss*ϕ_ss*zp_ss
	f_y_t[13,θ] = -η*γ*θ_ss/(1-η)

	#Definition of zf

	f_x_tp1[14,A] = ρ*A_ss*ϕ_ss*zf_ss
	f_y_t[14,ϕ] = ρ*A_ss*ϕ_ss*zf_ss
	f_y_t[14,zf] = ρ*A_ss*ϕ_ss*zf_ss
	f_y_tp1[14,ϕ] = ρ*β*(1-δ)*A_ss*ϕ_ss*(Ez - zf_ss)
	f_y_tp1[14,F_A] = f_y_tp1[14,ϕ]
	f_y_tp1[14,c] = - f_y_tp1[14,ϕ]
	f_y_t[14,c] = f_y_tp1[14,ϕ]
	f_y_tp1[14,zf] = -ρ*β*(1-δ)*A_ss*ϕ_ss*zf_ss
	f_y_t[14,θ] = -η*γ*θ_ss/(1-η)

	#Relationship between F_A in controls and A in states

	f_x_tp1[15,A] = 1.
	f_y_t[15,F_A] = -1.

	#Shocks

	f_x_tp1[16,ϵ_A] = 1.
	f_x_tp1[17,ϵ_g] = 1.
	f_x_tp1[18,ϵ_μ] = 1.
	f_x_tp1[19,ϵ_m] = 1.

	#Inflation components

	f_y_t[20,zc] = zc_ss
	f_y_t[20,zp] = -zp_ss
	f_y_t[20,ϕ] = F/ϕ_ss
	f_x_tp1[20,A] = F/ϕ_ss

	f_y_t[21,hc] = 1.
	f_y_t[21,θ] = -κ*σ
	f_y_tp1[21,hc] = -β

	f_y_t[22,subsp] = 1.
	# f_y_t[22,zc] = -κ*(1-η)*ϕ_ss*μᵖ_ss*zc_ss/γ
	f_y_t[22,zs] = -κ*(1-η)*ϕ_ss*μᵖ_ss*(1-ρ)*zs_ss/γ
	f_y_tp1[22,subsp] = -β

	f_y_t[23,Πᶠ] = 1.
	f_y_t[23,zf] = -κ*(1-η)*ϕ_ss*(μᶠ_ss+μᵖ_ss)*ρ*zf_ss/γ
	f_y_tp1[23,Πᶠ] = -β

end


# %%
function state_space!(G, R, Z, g_x, h_x, ss_x)
	G .= h_x
	R[end-4+1:end,:] .= I(4)

	@index np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	@index c zp zf θ ϕ v Y zs π F_A
	@load_cat ss ss_x np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m

	Z[1,:] .= g_x[Y, :]
	Z[2,:] .= g_x[π, :]
	Z[3,:] .= h_x[R, :]
	Z[4,:] .= (h_x[np, :]*np_ss + h_x[nf, :]*nf_ss)/(np_ss + nf_ss)
end
# %%
"Defines the linear model with a Dual Labor Market"
function lin_dual()
	cal = calibrate()
	ss_x, ss_y = steady_state(cal)
	param_names = ["ρ_A", "ρ_g", "ρ_R", "ρ_π", "ρ_y", "ψ", "σ_A", "σ_μ", "σ_g", "σ_m"]
	return LinMod{eltype(values(cal))}(cal, ss_x, ss_y, 4, 10, param_names, (f_y_tp1, f_y_t, f_x_tp1, f_x_t, param)->set_lin_mod!(f_y_tp1, f_y_t, f_x_tp1, f_x_t, param, cal, ss_x, ss_y), (G, R, Z, g_x, h_x)->state_space!(G, R, Z, g_x, h_x, ss_x))
end
# %%
"Defines the linear model with a Dual Labor Market and an initial guess to solve for the steady state."
function lin_dual(cal)
	ss_x, ss_y = steady_state(cal)
	param_names = ["ρ_A", "ρ_g", "ρ_R", "ρ_π", "ρ_y", "ψ", "σ_A", "σ_μ", "σ_g", "σ_m"]
	return LinMod{eltype(values(cal))}(cal, ss_x, ss_y, 4, 10, param_names, (f_y_tp1, f_y_t, f_x_tp1, f_x_t, param)->set_lin_mod!(f_y_tp1, f_y_t, f_x_tp1, f_x_t, param, cal, ss_x, ss_y), (G, R, Z, g_x, h_x)->state_space!(G, R, Z, g_x, h_x, ss_x))
end

end
