using SpecialFunctions
using StatsBase
using Macros

"Simulates draws for a given model."
function draw(EstM::EstLinMod ; N=10000, T=500)

	n_x = length(keys(EstM.model.ss_x))
	n_y = length(keys(EstM.model.ss_y))
	n_ϵ = EstM.model.n_ϵ
	e = SharedArray{Float64}((N, T, n_ϵ))
	e .= randn(N, T, n_ϵ)
    sim_x = SharedArray{Float64}((N, T, n_x))
	sim_y = SharedArray{Float64}((N, T, n_y))
    sim_x[:,1,:] .= 0.

    @sync @distributed for n in 1:N
        for t in 1:T-1
            sim_x[n,t+1,:] .= EstM.G*sim_x[n,t,:] + EstM.R*e[n,t,:]
			sim_y[n,t+1,:] .= EstM.g_x*sim_x[n,t+1,:]
        end
    end

    return (convert(SharedArray,sim_x[:,3:end,:]), convert(SharedArray,sim_y[:,2:end-1,:]))

end

# "Variance and covariances of main components of inflation"
# function π_var(M::EstLinMod, sim_x::SharedArray, sim_y::SharedArray; Tburn=100)
#
# 	println("Initialization...")
#
#     N, T , _ = size(sim_x)
# 	n_data = 5
#     sim = SharedArray{Float64}(N, T, n_data)
#
# 	ss_x = M.model.ss_x
# 	ss_y = M.model.ss_y
# 	cal = M.model.calibrated
# 	param = M.param
#
# 	Macros.@load cal σᶻ β σ η ϵ F b s δ ρ m γ gsY
# 	Dual_pi.@spec_funs
# 	Macros.@load_vec param ρ_A ρ_g ρ_R ρ_π ρ_y ψ σ_A σ_μ σ_g σ_m
# 	Macros.@index np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m
# 	Macros.@index c zp zf θ ϕ v Y zs π F_A zc hc subsp Πᶠ
# 	Macros.@load_cat ss ss_x np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m
# 	Macros.@load_cat ss ss_y c zp zf θ ϕ v Y zs π F_A zc
#
# 	μᵖzᶜ = m*θ_ss^(-σ)*(1-CDF(zs_ss))*zc_ss
# 	μᶠzᶠ = m*θ_ss^(-σ)*(CDF(zs_ss)-CDF(zf_ss))*zf_ss
# 	κ = (1-β*ψ)*(1-ψ)/ψ
#
# 	println("Deriving obervables")
#
# 	@views @. begin
# 		sim[:,:,1] = sim_y[:,:,hc]
# 		sim[:,:,2] = sim_y[:,:,subsp]
# 		sim[:,:,3] = sim_y[:,:,Πᶠ]
# 		sim[:,:,4] = -κ*sim_x[:,:,A]/(1-β*ρ_A)
# 		sim[:,2:end,5] = σ_μ*sim_x[:,1:end-1,ϵ_μ]
# 	end
#
# 	println("Computing covariances for each draw...")
#
#     cov_sim = SharedArray{Float64}((N, N, n_data))
#     @sync @distributed for n in 1:N
#         for i in 1:n_data
#             for j in 1:n_data
#                 cov_sim[n,i,j] = StatsBase.cov(sim[n,Tburn:end,i], sim[n,Tburn:end,j])
#             end
#         end
#     end
#
#     println("Computing covariances: means...")
#
#     cov_sim_mean = zeros(n_data, n_data)
#     for i in 1:n_data
#         for j in 1:n_data
#             cov_sim_mean[i,j] = mean(cov_sim[:,i,j])
#         end
#     end
#
# 	return (μᵖzᶜ, μᶠzᶠ, cov_sim_mean, sim)
#
# end


"Variance and covariances of main components of inflation"
function π_var(M::EstLinMod, sim_x::SharedArray, sim_y::SharedArray; Tburn=100)

	println("Initialization...")

    N, T , _ = size(sim_x)
	n_data = 5
    sim = SharedArray{Float64}(N, T, n_data)

	ss_x = M.model.ss_x
	ss_y = M.model.ss_y
	cal = M.model.calibrated
	param = M.param

	Macros.@load cal σᶻ β σ η ϵ F b s δ ρ m γ gsY
	Dual_pi.@spec_funs
	Macros.@load_vec param ρ_A ρ_g ρ_R ρ_π ρ_y ψ σ_A σ_μ σ_g σ_m
	Macros.@index np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	Macros.@index c zp zf θ ϕ v Y zs π F_A zc hc subsp Πᶠ
	Macros.@load_cat ss ss_x np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	Macros.@load_cat ss ss_y c zp zf θ ϕ v Y zs π F_A zc

	# μᵖzᶜ = m*θ_ss^(-σ)*(1-CDF(zs_ss))*zc_ss
	# μᶠzᶠ = m*θ_ss^(-σ)*(CDF(zs_ss)-CDF(zf_ss))*zf_ss

	μᵖzs = m*θ_ss^(-σ)*(1-CDF(zs_ss))*zs_ss
	μzf = m*θ_ss^(-σ)*(1-CDF(zf_ss))*zf_ss
	κ = (1-β*ψ)*(1-ψ)/ψ

	println("Deriving obervables")

	@views @. begin
		sim[:,:,1] = sim_y[:,:,hc]
		sim[:,:,2] = sim_y[:,:,subsp]
		sim[:,:,3] = sim_y[:,:,Πᶠ]
		sim[:,:,4] = -κ*sim_x[:,:,A]/(1-β*ρ_A)
		sim[:,2:end,5] = σ_μ*sim_x[:,1:end-1,ϵ_μ]
	end

	println("Computing covariances for each draw...")

    cov_sim = SharedArray{Float64}((N, N, n_data))
    @sync @distributed for n in 1:N
        for i in 1:n_data
            for j in 1:n_data
                cov_sim[n,i,j] = StatsBase.cov(sim[n,Tburn:end,i], sim[n,Tburn:end,j])
            end
        end
    end

    println("Computing covariances: means...")

    cov_sim_mean = zeros(n_data, n_data)
    for i in 1:n_data
        for j in 1:n_data
            cov_sim_mean[i,j] = mean(cov_sim[:,i,j])
        end
    end

	# return (μᵖzᶜ, μᶠzᶠ, cov_sim_mean, sim)
	return (μᵖzs, μzf, cov_sim_mean, sim)

end
