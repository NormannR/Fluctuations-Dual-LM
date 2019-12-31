function SS!(R,x)
	zp = x[1]
	θ = x[2]

	ϕ = (ϵ-1)/ϵ
	q = m*θ^(-σ)

	ξ = s+(1-s)*CDF(zp)

	zf = (b + (1-δ)*β*(η*γ*θ/(1-η) - ρ*ϕ*Ez))/(ρ*ϕ*(1-β*(1-δ)))
	zc = zp + F/ϕ
	zs = (zc - ρ*zf)/(1-ρ)

    R[1] = ϕ*zp + (1-β*(1-s))*F + β*(1-s)*ϕ*int(zp) - b - β*(1-ξ)*η*γ*θ/(1-η)
	R[2] = γ/((1-η)*ϕ*q)  - (int(zf) + ρ*(int(zf)-int(zs)))
end

function steady_state()
	x0 = [0.638, 0.83]
	R = zeros(2)
	sol = nlsolve(SS!,x0, show_trace=true)

	zp = sol.zero[1]
	θ = sol.zero[2]

	ϕ = (ϵ-1)/ϵ
	μ = 1/ϕ
	q = m*θ^(-σ)
	R = 1/β
	p = θ*q

	ξ = s+(1-s)*CDF(zp)

	zf = (b + (1-δ)*β*(η*γ*θ/(1-η) - ρ*ϕ*Ez))/(ρ*ϕ*(1-β*(1-δ)))
	zc = zp + F/ϕ
	zs = (zc - ρ*zf)/(1-ρ)

	μᵖ = p*(1-CDF(zs))
	μᶠ = p*(CDF(zs)-CDF(zf))
	np = δ*μᵖ/(ξ*μᶠ*(1-δ) + δ*μᵖ*(1-ξ) + ξ*δ)
    nf = ξ*μᶠ/(ξ*μᶠ*(1-δ) + δ*μᵖ*(1-ξ) + ξ*δ)
    u = 1-np-nf

    e = u+ξ*np+δ*nf
    v = θ*e

    Y = np*((1-ξ)*I(zp)/(1-CDF(zp)) + ξ*I(zs)/(1-CDF(zs))) + nf*ρ*((1-δ)*Ez +
        δ*(I(zf)-I(zs))/(CDF(zs)-CDF(zf)))

    g = gsY*Y
    c = Y-g-γ*v

    A = 1
    π = 1

    return [R, c, np, nf, zp, zf, zs, θ, ϕ, π, Y, v, A, μ, g]
end

@everywhere function set_lin_mod!(Γ₀,Γ₁,Ψ,Π,SDX,param)

    Γ₀ .= zeros(n_y,n_y)
    Γ₁ .= zeros(n_y,n_y)
    Ψ .= zeros(n_y,n_x)
    Π .= zeros(n_y,n_η)

    ρ_A = param[1]
    ρ_μ = param[2]
    ρ_g = param[3]
    ρ_R = param[4]
    ρ_π = param[5]
    ρ_y = param[6]
    ψ = param[7]

    σ_A = param[8]
    σ_μ = param[9]
    σ_g = param[10]
    σ_m = param[11]

    #Endogenous variables

    R = 1
    c = 2
    np = 3
    nf = 4
    zp = 5
    zf = 6
    zs = 7
    θ = 8
    ϕ = 9
    π = 10
    Y = 11
    v = 12

    A = 13
    μ = 14
    g = 15

    E_c = 16
    E_zp = 17
    E_zf = 18
    E_θ = 19
    E_ϕ = 20
    E_π = 21

    #Perturbations

    e_A = 1
    e_μ = 2
    e_g = 3
    e_m = 4

    #Expectation errors

    η_c = 1
    η_zp = 2
    η_zf = 3
    η_θ = 4
    η_ϕ = 5
    η_π = 6

    #Euler equation

    Γ₀[1,c] = 1
    Γ₀[1,E_c] = -1
    Γ₀[1,R] = 1
    Γ₀[1,E_π] = -1

    #Definition of zp

    Γ₀[2,A] = ss[A]*ss[ϕ]*(ss[zp] + β*(1-s)*ρ_A*int(ss[zp]))
    Γ₀[2,ϕ] = ss[A]*ss[ϕ]*ss[zp]
    Γ₀[2,zp] = Γ₀[2,ϕ]
    Γ₀[2,c] = b - ss[A]*ss[ϕ]*ss[zp] - F
    Γ₀[2,E_c] = -Γ₀[2,c]
    Γ₀[2,E_ϕ] = β*(1-s)*ss[A]*ss[ϕ]*int(ss[zp])
    Γ₀[2,E_θ] = -β*(1-s)*(1-CDF(ss[zp]))*η*γ*ss[θ]/(1-η)
    Γ₀[2,E_zp] = β*(1-s)*ss[zp]*(-ss[A]*ss[ϕ]*(1-CDF(ss[zp])) + η*γ*ss[θ]*PDF(ss[zp])/(1-η))

    #Definition of zf

    Γ₀[3,A] = ρ*ss[A]*ss[ϕ]*(ss[zf] + β*(1-δ)*(Ez - ss[zf])*ρ_A)
    Γ₀[3,ϕ] = ρ*ss[A]*ss[ϕ]*ss[zf]
    Γ₀[3,zf] = Γ₀[3,ϕ]
    Γ₀[3,E_ϕ] = ρ*β*ss[A]*ss[ϕ]*(1-δ)*(Ez-ss[zf])
    Γ₀[3,c] = b - ρ*ss[A]*ss[ϕ]*ss[zf]
    Γ₀[3,E_c] = -Γ₀[3,c]
    Γ₀[3,E_zf] = -ρ*ss[A]*ss[ϕ]*β*(1-δ)*ss[zf]
    Γ₀[3,E_θ] = -β*(1-δ)*η*γ*ss[θ]/(1-η)

    #Definition of zs

    Γ₀[4,zs] = (1-ρ)*ss[zs]
    Γ₀[4,zp] = -ss[zp]
    Γ₀[4,zf] = ρ*ss[zf]
	Γ₀[4,A] = F/(ss[A]*ss[ϕ])
	Γ₀[4,ϕ] = Γ₀[4,A]

    #Job creation condition

    Γ₀[5,A] = -γ
    Γ₀[5,ϕ] = -γ
    Γ₀[5,θ] = σ*γ
    Γ₀[5,zs] = (1-ρ)*(1-CDF(ss[zs]))*ss[zs]*((1-η)*ss[ϕ]*m*ss[θ]^(-σ))
    Γ₀[5,zf] = ρ*(1-CDF(ss[zf]))*ss[zf]*((1-η)*ss[ϕ]*m*ss[θ]^(-σ))

    #Permanent employment

    Γ₀[6,np] = 1
    Γ₁[6,np] = (1-s)*(1-CDF(ss[zp]))
    Γ₀[6,zp] = (1-s)*ss[zp]*PDF(ss[zp])
    Γ₀[6,v] = -(1-CDF(ss[zs]))*m*ss[θ]^(-σ)*ss[v]/ss[np]
    Γ₀[6,θ] = σ*(1-CDF(ss[zs]))*m*ss[θ]^(-σ)*ss[v]/ss[np]
    Γ₀[6,zs] = ss[zs]*PDF(ss[zs])*m*ss[θ]^(-σ)*ss[v]/ss[np]

    #Temporary employment

    Γ₀[7,nf] = 1
    Γ₁[7,nf] = (1-δ)
    Γ₀[7,v] = -(CDF(ss[zs])-CDF(ss[zf]))*m*ss[θ]^(-σ)*ss[v]/ss[nf]
    Γ₀[7,θ] = σ*(CDF(ss[zs])-CDF(ss[zf]))*m*ss[θ]^(-σ)*ss[v]/ss[nf]
    Γ₀[7,zs] = -ss[zs]*PDF(ss[zs])*m*ss[θ]^(-σ)*ss[v]/ss[nf]
    Γ₀[7,zf] = ss[zf]*PDF(ss[zf])*m*ss[θ]^(-σ)*ss[v]/ss[nf]

    #Production function

    Γ₀[8,Y] = 1.
    Γ₀[8,A] = -1.
    Γ₁[8,np] = (1-s)*I(ss[zp])*ss[np]/ss[Y]
    Γ₁[8,nf] = ρ*(1-δ)*Ez*ss[nf]/ss[Y]
    Γ₀[8,zp] = (1-s)*ss[np]*ss[zp]^2*PDF(ss[zp])/ss[Y]
    Γ₀[8,zs] = (1-ρ)*ss[zs]^2*PDF(ss[zs])*ss[v]*m*ss[θ]^(-σ)/ss[Y]
    Γ₀[8,zf] = ρ*ss[zf]^2*PDF(ss[zf])*ss[v]*m*ss[θ]^(-σ)/ss[Y]
    Γ₀[8,v] = -(I(ss[zs]) + ρ*(I(ss[zf]) - I(ss[zs])))*ss[v]*m*ss[θ]^(-σ)/ss[Y]
    Γ₀[8,θ] = σ*(I(ss[zs]) + ρ*(I(ss[zf]) - I(ss[zs])))*ss[v]*m*ss[θ]^(-σ)/ss[Y]

    #NK PC

    Γ₀[9,π] = 1
    Γ₀[9,E_π] = - β
    Γ₀[9,μ] = - 1
    Γ₀[9,ϕ] = - (1-β*ψ)*(1-ψ)/ψ

    #Taylor rule

    Γ₀[10,R] = 1
    Γ₀[10,E_π] = -(1-ρ_R)*ρ_π
    Γ₀[10,Y] = -(1-ρ_R)*ρ_y
    Γ₁[10,R] = ρ_R
    Ψ[10,e_m] = 1

    #Definition of v

    Γ₀[11,v] = 1.
    Γ₀[11,θ] = -1.
    Γ₁[11,np] = -(1-s)*(1-CDF(ss[zp]))*ss[θ]*ss[np]/ss[v]
    Γ₁[11,nf] = -(1-δ)*ss[θ]*ss[nf]/ss[v]
    Γ₀[11,zp] = -(1-s)*ss[θ]*PDF(ss[zp])*ss[zp]*ss[np]/ss[v]

    #Ressource constraint

    Γ₀[12,Y] = 1
    Γ₀[12,c] = -ss[c]/ss[Y]
    Γ₀[12,g] = -ss[g]/ss[Y]
    Γ₀[12,v] = -γ*ss[v]/ss[Y]

    #Shock processes

    Γ₀[13,A] = 1
    Γ₁[13,A] = ρ_A
    Ψ[13,e_A] = 1

    Γ₀[14,g] = 1
    Γ₁[14,g] = ρ_g
    Ψ[14,e_g] = 1

    Γ₀[15,μ] = 1
    Γ₁[15,μ] = ρ_μ
    Ψ[15,e_μ] = 1

    #Expectation errors

    Γ₀[16,c] = 1
    Γ₁[16,E_c] = 1
    Π[16,η_c] = 1

    Γ₀[17,zp] = 1
    Γ₁[17,E_zp] = 1
    Π[17,η_zp] = 1

    Γ₀[18,zf] = 1
    Γ₁[18,E_zf] = 1
    Π[18,η_zf] = 1

    Γ₀[19,θ] = 1
    Γ₁[19,E_θ] = 1
    Π[19,η_θ] = 1

    Γ₀[20,ϕ] = 1
    Γ₁[20,E_ϕ] = 1
    Π[20,η_ϕ] = 1

    Γ₀[21,π] = 1
    Γ₁[21,E_π] = 1
    Π[21,η_π] = 1

    SDX .= diagm([σ_A,σ_μ,σ_g,σ_m])

end

@everywhere function set_obs_eq!(Z)

    R = 1
    c = 2
    np = 3
    nf = 4
    zp = 5
    zf = 6
    zs = 7
    θ = 8
    ϕ = 9
    π = 10
    Y = 11
    v = 12

    A = 13
    μ = 14
    g = 15

    Z[1,Y] = 1
    Z[2,π] = 1
    Z[3,R] = 1
    Z[4,np] = +ss[np]/(ss[np]+ss[nf])
    Z[4,nf] = +ss[nf]/(ss[np]+ss[nf])

end
