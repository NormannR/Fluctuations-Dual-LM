using NLsolve
using Distributions
using JLD2, FileIO
# %%
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
q = 0.7
ϵ = 6
μ = ϵ/(ϵ-1)
ϕ = 1/μ
n = 0.74
# %%
u = 1-n
nf = nfsn*n
np = n-nf
s = ssξ*ξ
δ = ξ*(1-nfsn)*μfsμ/(nfsn*(1-μfsμ))
e = u+δ*nf+ξ*np
μᵖ = ξ*np/e
μᶠ = δ*nf/e
# %%
Ez = exp(σᶻ^2/2)
Φ(x) = cdf(Normal(0,1),x)
G(x) = Φ(log(max(x,0))/σᶻ)
I(x) = exp(σᶻ^2/2)*Φ((σᶻ^2-log(max(x,0)))/σᶻ)
int(x) = I(x) - x*(1-G(x))
# %%
function SS!(R,x)
	zp = x[1]
	F = x[2]
	zs = x[3]
	zf = x[4]

	zc = zp+F/ϕ
	Up = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp)
	wp = F/ρᶠ
	p = μᵖ/(1-G(zs))
	θ = p/q

	ρ = (zs-zc)/(zs-zf)
	Uf = ρ*ϕ*((1-β*(1-δ))*zf + β*(1-δ)*Ez)
	γ = (1-η)*ϕ*q*(int(zf) + ρ*(int(zf)-int(zs)))
	bp = Up - β*(1-ξ)*η*γ*θ/(1-η)
	bf = Uf - β*(1-δ)*η*γ*θ/(1-η)

	R[1] = ξ-s-(1-s)*G(zp)
	R[2] = η*ϕ*((1-ξ)*I(zp)/(1-G(zp)) + ξ*I(zs)/(1-G(zs))) - η*β*(1-s)*F + (1-ξ)*η*F + (1-η)*Up - wp
	R[3] = (G(zs)-G(zf)) - μfsμ*(1-G(zf))
 	R[4] = (bp-bf)

end
# %%
R = zeros(4)
sol = nlsolve(SS!, [0.6,0.38, 1.28, 1.07], show_trace = true)
# %%
zp = sol.zero[1]
F = sol.zero[2]
zs = sol.zero[3]
zf = sol.zero[4]

zc = zp+F/ϕ
Up = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp)
wp = F/ρᶠ
p = μᵖ/(1-G(zs))
θ = p/q

ρ = (zs-zc)/(zs-zf)
Uf = ρ*ϕ*((1-β*(1-δ))*zf + β*(1-δ)*Ez)
γ = (1-η)*ϕ*q*(int(zf) + ρ*(int(zf)-int(zs)))
b = Up - β*(1-ξ)*η*γ*θ/(1-η)
wf = η*ϕ*((1-δ)*Ez + δ*(I(zf)-I(zs))/(G(zs)-G(zf))) + (1-η)*Uf
m = q*θ^σ
(wp-wf)/wp

# %%
@save "./code/v1.2/input/parameters.jld2" σᶻ β σ η ϵ F b s δ ρ m γ gsY
# %%
#When F is indexed on A and ϕ
function SS!(R,x)
	zp = x[1]
	F = x[2]
	zs = x[3]
	zf = x[4]

	zc = zp+F
	Up = ϕ*zp + (1-β*(1-s))*ϕ*F + β*(1-s)*ϕ*int(zp)
	wp = F/ρᶠ
	p = μᵖ/(1-G(zs))
	θ = p/q

	ρ = (zs-zc)/(zs-zf)
	Uf = ρ*ϕ*((1-β*(1-δ))*zf + β*(1-δ)*Ez)
	γ = (1-η)*ϕ*q*(int(zf) + ρ*(int(zf)-int(zs)))
	bp = Up - β*(1-ξ)*η*γ*θ/(1-η)
	bf = Uf - β*(1-δ)*η*γ*θ/(1-η)

	R[1] = ξ-s-(1-s)*G(zp)
	R[2] = η*ϕ*((1-ξ)*I(zp)/(1-G(zp)) + ξ*I(zs)/(1-G(zs))) - η*β*(1-s)*ϕ*F + (1-ξ)*η*ϕ*F + (1-η)*Up - wp
	R[3] = (G(zs)-G(zf)) - μfsμ*(1-G(zf))
 	R[4] = (bp-bf)

end
# %%
R = zeros(4)
sol = nlsolve(SS!, [0.6,0.38, 1.28, 1.07], show_trace = true)
# %%
zp = sol.zero[1]
F = sol.zero[2]
zs = sol.zero[3]
zf = sol.zero[4]

zc = zp+F
Up = ϕ*zp + (1-β*(1-s))*ϕ*F + β*(1-s)*ϕ*int(zp)
wp = F/ρᶠ
p = μᵖ/(1-G(zs))
θ = p/q

ρ = (zs-zc)/(zs-zf)
Uf = ρ*ϕ*((1-β*(1-δ))*zf + β*(1-δ)*Ez)
γ = (1-η)*ϕ*q*(int(zf) + ρ*(int(zf)-int(zs)))
b = Up - β*(1-ξ)*η*γ*θ/(1-η)
wf = η*ϕ*((1-δ)*Ez + δ*(I(zf)-I(zs))/(G(zs)-G(zf))) + (1-η)*Uf
m = q*θ^σ
(wp-wf)/wp

# %%
@save "./code/v1.2/parallel/input/parameters_idx.jld2" σᶻ β σ η ϵ F b s δ ρ m γ gsY
