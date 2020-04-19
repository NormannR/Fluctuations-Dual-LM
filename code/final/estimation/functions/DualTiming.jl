module DualTiming

using SymEngine
using NLsolve
using Distributions
using Macros
using SMC
using SpecialFunctions

export lin_dual, sym_dual

@def spec_funs begin
	Ez = exp(σᶻ^2/2)
	Φ(x) = 0.5*(1+erf(x/sqrt(2)))
	CDF(x) = Φ(log(x)/σᶻ)
	PDF(x) = exp(-0.5*log(x)^2/σᶻ^2)/(x*σᶻ*sqrt(2*pi))
	I(x) = exp(σᶻ^2/2)*Φ((σᶻ^2-log(x))/σᶻ)
	int(x) = I(x) - x*(1-CDF(x))
end

"Calibrates the model"
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
	n = 0.74

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

		R[1] = μfsμ*(1-CDF(zf)) - (CDF(zs)-CDF(zf))

	end

	R = ones(1)
	sol = nlsolve( solve_ρ!, [0.9751933921922066], show_trace=true, ftol=1e-12, iterations=20)
	ρ = sol.zero[1]

	zf = (rU - ρ*ϕ*(1-δ)*β*Ez)/(ρ*ϕ*(1-β*(1-δ)))
	zs = (zc-ρ*zf)/(1-ρ)
	p = μᵖ/(1-CDF(zs))
	θ = p/q

	m = q*θ^σ
	γ = (1-η)*ϕ*q*(int(zs) + ρ*(int(zf)-int(zs)))
	b = rU - η*γ*θ/(1-η)
	wf = η*ϕ*ρ*((1-δ)*Ez + δ*(I(zf)-I(zs))/(CDF(zs)-CDF(zf))) + (1-η)*rU
	wp = η*ϕ*((1-ξ)*I(zp)/(1-CDF(zp)) + ξ*I(zs)/(1-CDF(zs))) - η*β*(1-s)*F + (1-ξ)*η*F + (1-η)*rU
	w = (1-nfsn)*wp + nfsn*wf
	h = b - ρᵇ*w

	return @save σᶻ β σ η ϵ ρᶠ h ρᵇ s δ ρ m γ gsY

end

@def SS begin
	ϕ = (ϵ-1)/ϵ
	q = m*θ^(-σ)
	p = θ*q
	rU = b + η*γ*θ/(1-η)
	F = ρᶠ*(η*ϕ*I(zp)/(1-CDF(zp)) + (1-η)*rU)/(1-ρᶠ*η*(1-β*(1-s)))
	zc = zp + F/ϕ
	zf = (rU - ρ*ϕ*(1-δ)*β*Ez)/(ρ*ϕ*(1-β*(1-δ)))
	zs = (zc-ρ*zf)/(1-ρ)

	ξ = s + (1-s)*CDF(zp)

	wf = η*ϕ*ρ*((1-δ)*Ez + δ*(I(zf)-I(zs))/(CDF(zs)-CDF(zf))) + (1-η)*rU
	wp = η*ϕ*((1-ξ)*I(zp)/(1-CDF(zp)) + ξ*I(zs)/(1-CDF(zs))) - η*β*(1-s)*F + (1-ξ)*η*F + (1-η)*rU

	μᵖ = p*(1-CDF(zs))
	μᶠ = p*(CDF(zs)-CDF(zf))
	np = δ*μᵖ/(ξ*μᶠ*(1-δ) + δ*μᵖ*(1-ξ) + ξ*δ)
	nf = ξ*μᶠ/(ξ*μᶠ*(1-δ) + δ*μᵖ*(1-ξ) + ξ*δ)
	u = 1-np-nf

	w = (np*wp + nf*wf)/(np + nf)
end

"Computes the steady state of the model"
function steady_state(cal::Dict{Symbol,Float64})

	@load cal σᶻ β σ η ϵ ρᶠ h ρᵇ s δ ρ m γ gsY

	@spec_funs

	function SS!(R,x)

		zp = x[1]
		θ = x[2]
		b = x[3]

		@SS

	    R[1] = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp) - rU
		R[2] = γ/((1-η)*ϕ*q)  - (int(zs) + ρ*(int(zf)-int(zs)))
		R[3] = b - (ρᵇ*w + h)

	end

	R = ones(3)
	sol = nlsolve( SS!, [0.6376100147942086, 0.8404147116717076, 0.8369693343226298], show_trace=true, ftol=1e-12, iterations=20)
	zp = sol.zero[1]
	θ = sol.zero[2]
	b = sol.zero[3]

	@SS

	R = 1/β
	e = u+ξ*np+δ*nf
    v = θ*e
	Y = np*((1-ξ)*I(zp)/(1-CDF(zp)) + ξ*I(zs)/(1-CDF(zs))) + nf*ρ*((1-δ)*Ez + δ*(I(zf)-I(zs))/(CDF(zs)-CDF(zf)))
	g = gsY*Y
	c = Y-g-γ*v
	A = 1
	π = 1
	F_π = π

	ss_x = @save np nf R g A
	ss_y = @save c zp zf θ ϕ F b wp wf v Y zs π F_π

	return (ss_x, ss_y)

end


"Computes the log-linearized system of equilibrium equations"
function set_lin_mod!(f_y_tp1, f_y_t, f_x_tp1, f_x_t, param, cal, ss_x, ss_y)

	fill!(f_y_tp1, 0.)
	fill!(f_y_t, 0.)
	fill!(f_x_tp1, 0.)
	fill!(f_x_t, 0.)

	@load cal σᶻ β σ η ϵ ρᶠ h ρᵇ s δ ρ m γ gsY
	@spec_funs
	@load_vec param ρ_A ρ_g ρ_R ρ_π ρ_y ψ σ_A σ_μ σ_g σ_m
	@index np nf R g A
	@index c zp zf θ ϕ F b wp wf v Y zs π F_π
	@load_cat ss ss_x np nf R g A
	@load_cat ss ss_y c zp zf θ ϕ F b wp wf v Y zs π F_π

	#Euler equation

	f_y_t[1,c] = 1.
	f_y_tp1[1,c] = -1.
	f_x_t[1,R] = 1.
	f_y_tp1[1,π] = -1.

	# Phillips curve

	f_y_tp1[2,π] = 1.
	f_y_tp1[2,F_π] = -β
	f_y_tp1[2,ϕ] = -(1-β*ψ)*(1-ψ)/ψ

	# Productivity shock

	f_x_tp1[3,A] = 1.
	f_x_t[3,A] = -ρ_A

	# Government spending shock

	f_x_tp1[4,g] = 1.
	f_x_t[4,g] = -ρ_g

	# Interest rate rule

	f_x_tp1[5,R] = 1.
	f_x_t[5,R] = -ρ_R
	f_y_tp1[5,F_π] = -(1-ρ_R)*ρ_π
	f_y_tp1[5,Y] = -(1-ρ_R)*ρ_y

	# Ressource constraint

	f_y_t[6,Y] = 1
	f_y_t[6,c] = -c_ss/Y_ss
	f_x_t[6,g] = -g_ss/Y_ss
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
	f_x_t[10,A] = -1.
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
	f_y_t[11,F] = -F_ss/ϕ_ss
	f_y_t[11,ϕ] = -f_y_t[11,F]
	f_x_t[11,A] = -f_y_t[11,F]

	#Job creation condition

	f_x_t[12,A] = -γ/((1-η)*ϕ_ss*m*θ_ss^(-σ))
	f_y_t[12,ϕ] = -γ/((1-η)*ϕ_ss*m*θ_ss^(-σ))
	f_y_t[12,θ] = σ*γ/((1-η)*ϕ_ss*m*θ_ss^(-σ))
	f_y_t[12,zs] = (1-ρ)*(1-CDF(zs_ss))*zs_ss
	f_y_t[12,zf] = ρ*(1-CDF(zf_ss))*zf_ss

	#Definition of zp

	f_x_t[13,A] = A_ss*ϕ_ss*zp_ss
	f_y_t[13,zp] = A_ss*ϕ_ss*zp_ss
	f_y_t[13,ϕ] = A_ss*ϕ_ss*zp_ss
	f_y_t[13,c] = β*(1-s)*(A_ss*ϕ_ss*int(zp_ss) - F_ss)
	f_y_tp1[13,c] = -f_y_t[13,c]
	f_x_tp1[13,A] = β*(1-s)*A_ss*ϕ_ss*int(zp_ss)
	f_y_tp1[13,ϕ] = f_x_tp1[13,A]
	f_y_tp1[13,zp] = -β*(1-s)*(1-CDF(zp_ss))*A_ss*ϕ_ss*zp_ss
	f_y_t[13,F] = F_ss
	f_y_tp1[13,F] = -β*(1-s)*F_ss
	f_y_t[13,b] = -b_ss
	f_y_t[13,θ] = -η*γ*θ_ss/(1-η)

	#Definition of zf

	f_x_t[14,A] = ρ*A_ss*ϕ_ss*zf_ss
	f_y_t[14,ϕ] = ρ*A_ss*ϕ_ss*zf_ss
	f_y_t[14,zf] = ρ*A_ss*ϕ_ss*zf_ss
	f_y_tp1[14,ϕ] = ρ*β*(1-δ)*A_ss*ϕ_ss*(Ez - zf_ss)
	f_x_tp1[14,A] = f_y_tp1[14,ϕ]
	f_y_tp1[14,c] = - f_y_tp1[14,ϕ]
	f_y_t[14,c] = f_y_tp1[14,ϕ]
	f_y_tp1[14,zf] = -ρ*β*(1-δ)*A_ss*ϕ_ss*zf_ss
	f_y_t[14,b] = -b_ss
	f_y_t[14,θ] = -η*γ*θ_ss/(1-η)

	#Definition of wp

	f_y_t[15,wp] = wp_ss
	f_x_t[15,A] = -η*((1-s)*ϕ_ss*I(zp_ss) + ϕ_ss*m*θ_ss^(-σ)*v_ss*I(zs_ss)/np_ss)
	f_y_t[15,ϕ] = f_x_tp1[15,A]
	f_x_t[15,np] = -η*((1-s)*ϕ_ss*I(zp_ss) + F_ss*(1-s)*(1-CDF(zp_ss)))
	f_y_t[15,zp] = η*(1-s)*ϕ_ss*zp_ss^2*PDF(zp_ss) + η*F_ss*(1-s)*zp_ss*PDF(zp_ss)
	f_y_t[15,θ] = η*ϕ_ss*m*θ_ss^(-σ)*v_ss*I(zs_ss)*σ/np_ss - η*γ*θ_ss
	f_y_t[15,v] = -η*ϕ_ss*m*θ_ss^(-σ)*v_ss*I(zs_ss)/np_ss
	f_y_t[15,zs] = η*ϕ_ss*m*θ_ss^(-σ)*v_ss*zs_ss^2*PDF(zs_ss)/np_ss
	f_y_t[15,F] = -η*F_ss*(1-s)*(1-CDF(zp_ss))
	f_y_t[15,c] = η*β*(1-s)*F_ss
	f_y_tp1[15,c] = -f_y_t[15,c]
	f_y_tp1[15,F] = η*β*(1-s)*F_ss
	f_y_t[15,b] = -(1-η)*b_ss
	f_x_tp1[15,np] = η*((1-s)*ϕ_ss*I(zp_ss) + (1-s)*(1-CDF(zp_ss))*F_ss + ϕ_ss*m*θ_ss^(-σ)*v_ss*I(zs_ss)/np_ss)

	#Definition of wf

	f_y_t[16,wf] = wf_ss
	f_y_t[16,ϕ] = -η*ρ*ϕ_ss*((1-δ)*Ez + m*θ_ss^(-σ)*v_ss*(I(zf_ss)-I(zs_ss))/nf_ss)
	f_x_t[16,A] = f_y_tp1[16,ϕ]
	f_x_tp1[16,nf] = -f_y_tp1[16,ϕ]
	f_x_t[16,nf] = -η*ρ*ϕ_ss*(1-δ)*Ez
	f_y_t[16,θ] = -η*γ*θ_ss + η*ρ*ϕ_ss*m*θ_ss^(-σ)*v_ss*(I(zf_ss)-I(zs_ss))*σ/nf_ss
	f_y_t[16,v] = -η*ρ*ϕ_ss*m*θ_ss^(-σ)*v_ss*(I(zf_ss)-I(zs_ss))/nf_ss
	f_y_t[16,zs] = -η*ρ*ϕ_ss*m*θ_ss^(-σ)*v_ss*(zs_ss)^2*PDF(zs_ss)/nf_ss
	f_y_t[16,zf] = +η*ρ*ϕ_ss*m*θ_ss^(-σ)*v_ss*(zf_ss)^2*PDF(zf_ss)/nf_ss
	f_y_t[16,b] = -(1-η)*b_ss

	#Definition of b

	f_y_t[17,b] = b_ss*(np_ss+nf_ss)
	f_x_t[17,np] = np_ss*(b_ss - (ρᵇ*wp_ss+h))
	f_x_t[17,nf] = nf_ss*(b_ss - (ρᵇ*wf_ss+h))
	f_y_t[17,wp] = -ρᵇ*np_ss*wp_ss
	f_y_t[17,wf] = -ρᵇ*nf_ss*wf_ss

	#Definition of F

	f_y_t[18,F] = F_ss*(1-ρᶠ*η)
	f_y_t[18,b] = -ρᶠ*(1-η)*b_ss
	f_y_t[18,θ] = -ρᶠ*η*γ*θ_ss
	f_y_tp1[18,F] = ρᶠ*η*β*(1-s)*F_ss
	f_y_t[18,c] = f_y_tp1[18,F]
	f_y_tp1[18,c] = -f_y_tp1[18,F]
	f_y_t[18,ϕ] = -ρᶠ*η*ϕ_ss*I(zp_ss)/(1-CDF(zp_ss))
	f_x_t[18,A] = f_y_t[18,ϕ]
	f_y_t[18,zp] = -ρᶠ*η*ϕ_ss*int(zp_ss)*zp_ss*PDF(zp_ss)/(1-CDF(zp_ss))^2

	#Relationship between F_y in controls and y in controls

	f_y_tp1[19,π] = -1.
	f_y_t[19,F_π] = 1.


end

"Computes the symbolic system of equilibrium equations"
function sym_mod(param, cal, ss_x, ss_y)

	n_x = length(ss_x)
	n_y = length(ss_y)

	Eq = Vector{SymEngine.Basic}(undef,n_x+n_y)

	@vars np_tm1_hat nf_tm1_hat R_tm1_hat g_t_hat A_t_hat
	@vars np_t_hat nf_t_hat R_t_hat g_tp1_hat A_tp1_hat
	@vars c_t_hat zp_t_hat zf_t_hat θ_t_hat ϕ_t_hat F_t_hat b_t_hat wp_t_hat wf_t_hat v_t_hat Y_t_hat zs_t_hat π_t_hat F_π_t_hat
	@vars c_tp1_hat zp_tp1_hat zf_tp1_hat θ_tp1_hat ϕ_tp1_hat F_tp1_hat b_tp1_hat wp_tp1_hat wf_tp1_hat v_tp1_hat Y_tp1_hat zs_tp1_hat π_tp1_hat F_π_tp1_hat

	@load cal σᶻ β σ η ϵ ρᶠ h ρᵇ s δ ρ m γ gsY
	@load_vec param ρ_A ρ_g ρ_R ρ_π ρ_y ψ σ_A σ_μ σ_g σ_m
	@load_cat ss ss_x np nf R g A
	@load_cat ss ss_y c zp zf θ ϕ F b wp wf v Y zs π F_π

	np_tm1 = np_ss*exp(np_tm1_hat)
	nf_tm1 = nf_ss*exp(nf_tm1_hat)
	R_tm1 = R_ss*exp(R_tm1_hat)
	g_t = g_ss*exp(g_t_hat)
	A_t = A_ss*exp(A_t_hat)
	np_t = np_ss*exp(np_t_hat)
	nf_t = nf_ss*exp(nf_t_hat)
	R_t = R_ss*exp(R_t_hat)
	g_tp1 = g_ss*exp(g_tp1_hat)
	A_tp1 = A_ss*exp(A_tp1_hat)
	c_t = c_ss*exp(c_t_hat)
	zp_t = zp_ss*exp(zp_t_hat)
	zf_t = zf_ss*exp(zf_t_hat)
	θ_t = θ_ss*exp(θ_t_hat)
	ϕ_t = ϕ_ss*exp(ϕ_t_hat)
	F_t = F_ss*exp(F_t_hat)
	b_t = b_ss*exp(b_t_hat)
	wp_t = wp_ss*exp(wp_t_hat)
	wf_t = wf_ss*exp(wf_t_hat)
	v_t = v_ss*exp(v_t_hat)
	Y_t = Y_ss*exp(Y_t_hat)
	zs_t = zs_ss*exp(zs_t_hat)
	π_t = π_ss*exp(π_t_hat)
	F_π_t = F_π_ss*exp(F_π_t_hat)
	c_tp1 = c_ss*exp(c_tp1_hat)
	zp_tp1 = zp_ss*exp(zp_tp1_hat)
	zf_tp1 = zf_ss*exp(zf_tp1_hat)
	θ_tp1 = θ_ss*exp(θ_tp1_hat)
	ϕ_tp1 = ϕ_ss*exp(ϕ_tp1_hat)
	F_tp1 = F_ss*exp(F_tp1_hat)
	b_tp1 = b_ss*exp(b_tp1_hat)
	wp_tp1 = wp_ss*exp(wp_tp1_hat)
	wf_tp1 = wf_ss*exp(wf_tp1_hat)
	v_tp1 = v_ss*exp(v_tp1_hat)
	Y_tp1 = Y_ss*exp(Y_tp1_hat)
	zs_tp1 = zs_ss*exp(zs_tp1_hat)
	π_tp1 = π_ss*exp(π_tp1_hat)
	F_π_tp1 = F_π_ss*exp(F_π_tp1_hat)

	@spec_funs

	ξ_t = s + (1-s)*CDF(zp_t)
	q_t = m*θ_t^(-σ)
	β_t_tp1 = β*c_t/c_tp1

	Eq[1] = -c_ss*(1/c_t - β*R_t/(π_tp1*c_tp1))

	Eq[2] = π_tp1_hat - β*F_π_tp1_hat - (1-β*ψ)*(1-ψ)*ϕ_tp1_hat/ψ

	Eq[3] = log(A_tp1) - ρ_A*log(A_t)

	Eq[4] = log(g_tp1/g_ss) - ρ_g*log(g_t/g_ss)

	Eq[5] = log(R_tp1/R_ss) - ρ_R*log(R_t/R_ss) - (1-ρ_R)*(ρ_π*log(F_π_tp1) + ρ_y*log(Y_tp1/Y_ss))

	Eq[6] = (Y_t - c_t - g_t - γ*v_t)/Y_ss

	Eq[7] = (v_t - θ_t*(1-(1-ξ_t)*np_tm1-(1-δ)*nf_tm1))/v_ss

	Eq[8] = (np_t - (1-ξ_t)*np_tm1 - v_t*q_t*(1-CDF(zs_t)))/np_ss

	Eq[9] = (nf_t - (1-δ)*nf_tm1 - v_t*q_t*(CDF(zs_t)-CDF(zf_t)))/nf_ss

	Eq[10] = (Y_t - ( (1-s)*A_t*I(zp_t)*np_tm1 + ρ*(1-δ)*Ez*A_t*nf_tm1 + A_t*v_t*q_t*I(zs_t) + ρ*q_t*v_t*A_t*(I(zf_t)-I(zs_t)) ))/Y_ss

	Eq[11] = (1-ρ)*zs_t - zp_t - F_t/(A_t*ϕ_t) + ρ*zf_t

	Eq[12] =  γ/((1-η)*A_t*ϕ_t*q_t) - ((1-ρ)*int(zs_t) + ρ*int(zf_t))

	Eq[13] = A_t*ϕ_t*zp_t - b_t + F_t - β_t_tp1*(1-s)*F_tp1 + β_t_tp1*A_tp1*ϕ_tp1*(1-s)*int(zp_tp1) - η*γ*θ_t/(1-η)

	Eq[14] = ρ*A_t*ϕ_t*zf_t - b_t + ρ*β_t_tp1*(1-δ)*A_tp1*ϕ_tp1*(Ez-zf_tp1) - η*γ*θ_t/(1-η)

	Eq[15] = wp_t - η*( (A_t*(1-s)*ϕ_t*I(zp_t) + (1-ξ_t)*F_t)*np_tm1 + A_t*ϕ_t*q_t*v_t*I(zs_t) )/np_t - η*( -β_t_tp1*(1-s)*F_tp1 + γ*θ_t ) - (1-η)*b_t

	Eq[16] = wf_t - η*(ϕ_t*ρ*A_t*((1-δ)*Ez*nf_tm1 + q_t*v_t*(I(zf_t) - I(zs_t)))/nf_t + γ*θ_t) - (1-η)*b_t

	Eq[17] = b_t*(np_t+nf_t) - ρᵇ*(np_t*wp_t + nf_t*wf_t) - h*(np_t+nf_t)

	Eq[18] = F_t - ρᶠ*(η*(A_t*ϕ_t*I(zp_t)/(1-CDF(zp_t)) + F_t - β_t_tp1*(1-s)*F_tp1 + γ*θ_t) + (1-η)*b_t)

	Eq[19] = F_π_t - π_tp1

	xp = [np_tp1_hat, nf_tp1_hat, R_tp1_hat, g_tp1_hat, A_tp1_hat]
	x = [np_t_hat, nf_t_hat, R_t_hat, g_t_hat, A_t_hat]
	yp = [c_tp1_hat, zp_tp1_hat, zf_tp1_hat, θ_tp1_hat, ϕ_tp1_hat, F_tp1_hat, b_tp1_hat, wp_tp1_hat, wf_tp1_hat, v_tp1_hat, Y_tp1_hat, zs_tp1_hat, π_tp1_hat, F_c_tp1_hat, F_F_tp1_hat, F_π_tp1_hat]
	y = [c_t_hat, zp_t_hat, zf_t_hat, θ_t_hat, ϕ_t_hat, F_t_hat, b_t_hat, wp_t_hat, wf_t_hat, v_t_hat, Y_t_hat, zs_t_hat, π_t_hat, F_c_t_hat, F_F_t_hat, F_π_t_hat]

	return (xp, x, yp, y, Eq)

end

"Defines the linear model with a Dual Labor Market"
function lin_dual()
	cal = calibrate()
	ss_x, ss_y = steady_state(cal)
	return LinMod{eltype(values(cal))}(cal, ss_x, ss_y, 4, (f_y_tp1, f_y_t, f_x_tp1, f_x_t, param)->set_lin_mod!(f_y_tp1, f_y_t, f_x_tp1, f_x_t, param, cal, ss_x, ss_y))
end

"Defines the symbolic model with a Dual Labor Market"
function sym_dual()
	cal = calibrate()
	ss_x, ss_y = steady_state(cal)
	return SymMod{eltype(values(cal))}(cal, ss_x, ss_y, param -> sym_mod(param, cal, ss_x, ss_y))
end

end
