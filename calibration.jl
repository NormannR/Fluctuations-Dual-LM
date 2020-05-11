import Pkg; Pkg.activate(".")
using NLsolve
using Distributions
using SpecialFunctions
# %%
σᶻ = 0.2
η = 0.6
σ = 0.6
β = 0.99
nfsn = 0.15
ssξ = 0.6
ξ = 0.03
gsY = 0.2
μfsμ = 0.675
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

Ez = exp(σᶻ^2/2)
Φ(x) = 0.5*(1+erf(x/sqrt(2)))
CDF(x) = Φ(log(max(x,0.))/σᶻ)
PDF(x) = exp(-0.5*log(max(x,0.))^2/σᶻ^2)/(x*σᶻ*sqrt(2*pi))
I(x) = exp(σᶻ^2/2)*Φ((σᶻ^2-log(max(x,0.)))/σᶻ)
int(x) = I(x) - x*(1-CDF(x))

R = ones(1)
function solve_zp!(R, x)
    R[1] = ξ - s - (1-s)*CDF(x[1])
end

sol = nlsolve( solve_zp!, [0.6376100147942086], show_trace=true, ftol=1e-12)
zp = sol.zero[1]
F = ρᶠ*ϕ*(η*I(zp)/(1-CDF(zp)) + (1-η)*(zp + β*(1-s)*int(zp)))/(1-ρᶠ*(1-β*(1-s)))

rU = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp)
zc = zp + F/ϕ
# %%
using Plots
pyplot()
# %%
function f(ρ)

    zf = (rU - ρ*ϕ*(1-δ)*β*Ez)/(ρ*ϕ*(1-β*(1-δ)))
    zs = (zc-ρ*zf)/(1-ρ)

    return (CDF(zs)-CDF(zf))/(1-CDF(zf))
end
# %%
ρ = range(0.987, stop=0.999, length=100)
plot(ρ, f.(ρ))
# %%
function solve_ρ!(R,x)

    ρ = x[1]

    zf = (rU - ρ*ϕ*(1-δ)*β*Ez)/(ρ*ϕ*(1-β*(1-δ)))
    zs = (zc-ρ*zf)/(1-ρ)

    R[1] = μfsμ - (CDF(zs)-CDF(zf))/(1-CDF(zf))

end
# %%
R = ones(1)
sol = nlsolve( solve_ρ!, [0.986], show_trace=true, ftol=1e-12, iterations=20)
ρ = sol.zero[1]

zf = (rU - ρ*ϕ*(1-δ)*β*Ez)/(ρ*ϕ*(1-β*(1-δ)))
zs = (zc-ρ*zf)/(1-ρ)
p = μᵖ/(1-CDF(zs))
θ = p/q

m = q*θ^σ
γ = (1-η)*ϕ*q*(int(zs) + ρ*(int(zf)-int(zs)))
b = rU - η*γ*θ/(1-η)

p = θ*q
# %%
using Classic
# %%
σᶻ = 0.2
η = 0.6
σ = 0.6
β = 0.99
μᵖ = 0.25
ssξ = 0.5
gsY = 0.2
ρᶠ = 4/9
q = 0.7
ρᵇ = 0.4
ϵ = 6
μ = ϵ/(ϵ-1)
ϕ = 1/μ
n = 0.9

u = 1-n
e = u/(1-μᵖ)
ξ = μᵖ*e/n
s = ssξ*ξ
# %%
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
