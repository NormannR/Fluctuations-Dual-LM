import Pkg; Pkg.activate("../../../")
path = pwd() * "/functions/"
push!(LOAD_PATH, path)
using Revise
using Dual
using Macros
using SpecialFunctions
# %%
cal = calibrate()
# %%
Macros.@load cal σᶻ β σ η ϵ ρᶠ h ρᵇ s δ ρ m γ gsY
# %%
Ez = exp(σᶻ^2/2)
Φ(x) = 0.5*(1+erf(x/sqrt(2)))
CDF(x) = Φ(log(max(x,0.))/σᶻ)
PDF(x) = exp(-0.5*log(max(x,0.))^2/σᶻ^2)/(x*σᶻ*sqrt(2*pi))
I(x) = exp(σᶻ^2/2)*Φ((σᶻ^2-log(max(x,0.)))/σᶻ)
int(x) = I(x) - x*(1-CDF(x))
# %%
function f(zp)

    ϕ = (ϵ-1)/ϵ
    F = ρᶠ*ϕ*(η*I(zp)/(1-CDF(zp)) + (1-η)*(zp + β*(1-s)*int(zp)))/(1-ρᶠ*(1-β*(1-s)))
    rU = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp)
    ξ = s + (1-s)*CDF(zp)

    zc = zp + F/ϕ
    zf = (rU - ρ*ϕ*(1-δ)*β*Ez)/(ρ*ϕ*(1-β*(1-δ)))
    zs = (zc-ρ*zf)/(1-ρ)

    q = γ/((1-η)*ϕ*(int(max(zs,zc)) + ρ*(int(zf)-int(max(zs,zf)))))
    θ = (q/m)^(-1/σ)
    p = θ*q

    b = rU - η*γ*θ/(1-η)

    μᵖ = p*(1-CDF(max(zs,zc)))
    μᶠ = p*(CDF(max(zs,zf))-CDF(zf))
    np = δ*μᵖ/(ξ*μᶠ*(1-δ) + δ*μᵖ*(1-ξ) + ξ*δ)
    nf = ξ*μᶠ/(ξ*μᶠ*(1-δ) + δ*μᵖ*(1-ξ) + ξ*δ)
    u = 1-np-nf
    e = u + ξ*np + δ*nf

    if nf > 0. && np > 0.

        wf = η*ϕ*ρ*((1-δ)*Ez + e*p*(I(zf)-I(max(zs,zf)))/nf) + (1-η)*rU
        wp = η*ϕ*((1-ξ)*I(zp)/(1-CDF(zp)) + e*p*I(max(zs,zc))/np) - η*β*(1-s)*F + (1-ξ)*η*F + (1-η)*rU
        w = (np*wp + nf*wf)/(np + nf)

    elseif nf == 0. && np > 0.

        wp = η*ϕ*((1-ξ)*I(zp)/(1-CDF(zp)) + e*p*I(max(zs,zc))/np) - η*β*(1-s)*F + (1-ξ)*η*F + (1-η)*rU
        w = wp

    elseif nf > 0. && np == 0.

        wf = η*ϕ*ρ*((1-δ)*Ez + e*p*(I(zf)-I(max(zs,zf)))/nf) + (1-η)*rU
        w = wf

    end

    return [np, nf, b, ρᵇ*w+h]

end
# %%
N = 50
zp = range(0.55, stop=0.75, length=N)
Y = zeros(4,N)
for n in 1:N
    Y[:,n] .= f(zp[n])
end
# %%
using Plots
pyplot()
# %%
p1 = plot(zp, Y[1,:], label="np")
plot!(p1, zp, Y[2,:], label="nf")
p2 = plot(zp, Y[3,:], label="b")
plot!(p2, zp, Y[4,:], label="ρᵇ w+h")
plot(p1,p2,layout=(2,1))
# %%
plot(p1,p2,layout=(2,1))
# %%
cal = calibrate()
# %%
Macros.@load cal σᶻ β σ η ϵ ρᶠ h ρᵇ s δ ρ m γ gsY
# %%
ρᵇ -= 0.01
# %%
Ez = exp(σᶻ^2/2)
Φ(x) = 0.5*(1+erf(x/sqrt(2)))
CDF(x) = Φ(log(max(x,0.))/σᶻ)
PDF(x) = exp(-0.5*log(max(x,0.))^2/σᶻ^2)/(x*σᶻ*sqrt(2*pi))
I(x) = exp(σᶻ^2/2)*Φ((σᶻ^2-log(max(x,0.)))/σᶻ)
int(x) = I(x) - x*(1-CDF(x))
# %%
function f(zp)

    ϕ = (ϵ-1)/ϵ
    F = ρᶠ*ϕ*(η*I(zp)/(1-CDF(zp)) + (1-η)*(zp + β*(1-s)*int(zp)))/(1-ρᶠ*(1-β*(1-s)))
    rU = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp)
    ξ = s + (1-s)*CDF(zp)

    zc = zp + F/ϕ
    zf = (rU - ρ*ϕ*(1-δ)*β*Ez)/(ρ*ϕ*(1-β*(1-δ)))
    zs = (zc-ρ*zf)/(1-ρ)

    q = γ/((1-η)*ϕ*(int(max(zs,zc)) + ρ*(int(zf)-int(max(zs,zf)))))
    θ = (q/m)^(-1/σ)
    p = θ*q

    b = rU - η*γ*θ/(1-η)

    μᵖ = p*(1-CDF(max(zs,zc)))
    μᶠ = p*(CDF(max(zs,zf))-CDF(zf))
    np = δ*μᵖ/(ξ*μᶠ*(1-δ) + δ*μᵖ*(1-ξ) + ξ*δ)
    nf = ξ*μᶠ/(ξ*μᶠ*(1-δ) + δ*μᵖ*(1-ξ) + ξ*δ)
    u = 1-np-nf
    e = u + ξ*np + δ*nf

    if nf > 0. && np > 0.

        wf = η*ϕ*ρ*((1-δ)*Ez + e*p*(I(zf)-I(max(zs,zf)))/nf) + (1-η)*rU
        wp = η*ϕ*((1-ξ)*I(zp)/(1-CDF(zp)) + e*p*I(max(zs,zc))/np) - η*β*(1-s)*F + (1-ξ)*η*F + (1-η)*rU
        w = (np*wp + nf*wf)/(np + nf)

    elseif nf == 0. && np > 0.

        wp = η*ϕ*((1-ξ)*I(zp)/(1-CDF(zp)) + e*p*I(max(zs,zc))/np) - η*β*(1-s)*F + (1-ξ)*η*F + (1-η)*rU
        w = wp

    elseif nf > 0. && np == 0.

        wf = η*ϕ*ρ*((1-δ)*Ez + e*p*(I(zf)-I(max(zs,zf)))/nf) + (1-η)*rU
        w = wf

    end

    return [np, nf, b, ρᵇ*w+h]

end
# %%
N = 50
zp = range(0.5, stop=1., length=N)
Y = zeros(4,N)
for n in 1:N
    Y[:,n] .= f(zp[n])
end
# %%
using Plots
pyplot()
# %%
p1 = plot(zp, Y[1,:], label="np")
plot!(p1, zp, Y[2,:], label="nf")
p2 = plot(zp, Y[3,:], label="b")
plot!(p2, zp, Y[4,:], label="ρᵇ w+h")
plot(p1,p2,layout=(2,1))
# %%
plot(p1,p2,layout=(2,1))
# %%
cal = calibrate_wo_idx()
# %%
Macros.@load cal σᶻ β σ η ϵ F h ρᵇw s δ ρ m γ gsY
# %%
Ez = exp(σᶻ^2/2)
Φ(x) = 0.5*(1+erf(x/sqrt(2)))
CDF(x) = Φ(log(max(x,0.))/σᶻ)
PDF(x) = exp(-0.5*log(max(x,0.))^2/σᶻ^2)/(x*σᶻ*sqrt(2*pi))
I(x) = exp(σᶻ^2/2)*Φ((σᶻ^2-log(max(x,0.)))/σᶻ)
int(x) = I(x) - x*(1-CDF(x))
# %%
ρᵇw = 0.98*ρᵇw
# %%
function f!(R,x)

    zp = x[1]

    ϕ = (ϵ-1)/ϵ
    rU = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp)
    θ1 = (rU - ρᵇw - h)/(η*γ/(1-η))
    ξ = s + (1-s)*CDF(zp)

    zc = zp + F/ϕ
    zf = (rU - ρ*ϕ*(1-δ)*β*Ez)/(ρ*ϕ*(1-β*(1-δ)))
    zs = (zc-ρ*zf)/(1-ρ)

    q = γ/((1-η)*ϕ*(int(max(zs,zc)) + ρ*(int(zf)-int(max(zs,zf)))))
    θ2 = (q/m)^(-1/σ)

    R[1] = θ2 - θ1

end
# %%
using NLsolve
R = ones(1)
sol = nlsolve( f!, [0.8], show_trace=true, ftol=1e-12, iterations=20)
# %%
zp = sol.zero[1]
# %%
ϕ = (ϵ-1)/ϵ
rU = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp)
θ = (rU - ρᵇw - h)/(η*γ/(1-η))
ξ = s + (1-s)*CDF(zp)
zc = zp + F/ϕ
zf = (rU - ρ*ϕ*(1-δ)*β*Ez)/(ρ*ϕ*(1-β*(1-δ)))
zs = (zc-ρ*zf)/(1-ρ)
q = γ/((1-η)*ϕ*(int(max(zs,zc)) + ρ*(int(zf)-int(max(zs,zf)))))
p = θ*q

b = rU - η*γ*θ/(1-η)

μᵖ = p*(1-CDF(max(zs,zc)))
μᶠ = p*(CDF(max(zs,zf))-CDF(zf))
np = δ*μᵖ/(ξ*μᶠ*(1-δ) + δ*μᵖ*(1-ξ) + ξ*δ)
nf = ξ*μᶠ/(ξ*μᶠ*(1-δ) + δ*μᵖ*(1-ξ) + ξ*δ)
u = 1-np-nf
e = u + ξ*np + δ*nf

if nf > 0. && np > 0.

    wf = η*ϕ*ρ*((1-δ)*Ez + e*p*(I(zf)-I(max(zs,zf)))/nf) + (1-η)*rU
    wp = η*ϕ*((1-ξ)*I(zp)/(1-CDF(zp)) + e*p*I(max(zs,zc))/np) - η*β*(1-s)*F + (1-ξ)*η*F + (1-η)*rU
    w = (np*wp + nf*wf)/(np + nf)

elseif nf == 0. && np > 0.

    wp = η*ϕ*((1-ξ)*I(zp)/(1-CDF(zp)) + e*p*I(max(zs,zc))/np) - η*β*(1-s)*F + (1-ξ)*η*F + (1-η)*rU
    w = wp

elseif nf > 0. && np == 0.

    wf = η*ϕ*ρ*((1-δ)*Ez + e*p*(I(zf)-I(max(zs,zf)))/nf) + (1-η)*rU
    w = wf

end
# %%
using Plots
pyplot()
# %%
p1 = plot(zp, Y[1,:], label="θ1")
plot!(p1, zp, Y[2,:], label="θ2")
