path = "input/rawdata.csv"
vars = [:y,:pi,:R,:n]
detrend = [:y,:n]
demean = [:pi,:R]
t1 = "1997Q1"
t2 = "2007Q4"
Y = linear_detrend(path,vars,detrend,demean,t1,t2)
# %%
"Sets the prior distributions"
function prior()
	prior = Vector{Distribution}(undef, 10)
	α0_ρ = 1
	α1_ρ = 1
	prior[1:3] .= Beta(α0_ρ,α1_ρ)
	prior[4] = truncated(Normal(1.5,0.75),1,Inf)
	prior[5] = truncated(Normal(0.12,0.15),0,Inf)
	μ = 0.7
	σ = 0.05
	α0_ψ = μ^2*(1-μ)/(σ)^2 - μ
	α1_ψ = α0_ψ*(1-μ)/μ
	prior[6] = Beta(α0_ψ,α1_ψ)
	μ = 0.5
	σ = 4.
	α0_σ = 2. + (μ/σ)^2
	α1_σ = μ*(α0_σ-1)
	prior[7:10] .= InverseGamma(α0_σ,α1_σ)
	return prior
end
# %%
function prior_flat()
	prior = Vector{Distribution}(undef, 10)
	α0_ρ = 1
	α1_ρ = 1
	prior[1:3] .= Beta(α0_ρ,α1_ρ)
	prior[4] = truncated(Normal(1.5,0.75),1,Inf)
	prior[5] = truncated(Normal(0.12,0.15),0,Inf)
	prior[6] = Beta(α0_ρ,α1_ρ)
	μ = 0.5
	σ = 4.
	α0_σ = 2. + (μ/σ)^2
	α1_σ = μ*(α0_σ-1)
	prior[7:10] .= InverseGamma(α0_σ,α1_σ)
	return prior
end
# %%
function prior_wo_g()
	prior = Vector{Distribution}(undef, 10)
	α0_ρ = 1
	α1_ρ = 1
	prior[1:2] .= Beta(α0_ρ,α1_ρ)
	prior[3] = truncated(Normal(1.5,0.75),1,Inf)
	prior[4] = truncated(Normal(0.12,0.15),0,Inf)
	prior[5] = Beta(α0_ρ,α1_ρ)
	μ = 0.5
	σ = 4.
	α0_σ = 2. + (μ/σ)^2
	α1_σ = μ*(α0_σ-1)
	prior[6:8] .= InverseGamma(α0_σ,α1_σ)
	return prior
end
# %%
bound = [
0. 1.;
0. 1.;
0. 1.;
0. Inf;
0. Inf;
0. 1.;
0. Inf;
0. Inf;
0. Inf;
0. Inf ]

bound_wo_g = [
0. 1.;
0. 1.;
0. Inf;
0. Inf;
0. 1.;
0. Inf;
0. Inf;
0. Inf ]
