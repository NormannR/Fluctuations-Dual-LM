using Macros
using SpecialFunctions
using LaTeXStrings
using StatsBase

"Returns the table of parameters estimates."
function table_estimates(prior, μ, σ, draws, weight, param_labels)

    top = "\\begin{table}[H]\n\\begin{center}\n\\caption{Prior and posterior distributions of structural parameters.}\n\\begin{tabular}{cccccccc}\n\\toprule\nParameter & \\multicolumn{3}{c}{Prior distribution} & \\multicolumn{4}{c}{Posterior distribution} \\\\ \\cmidrule{2-4} \\cmidrule{5-8} & Distr. & Para (1) & Para(2) & Mean  & Std. Dev. & 5\\%   & 95\\% \\\\ \\midrule \n"
    body = ""
    for i in 1:length(μ)
        x = vcat( param_labels[i] , prior[i,:] , μ[i] , σ[i] , quantile(draws[:,i], Weights(weight), [0.05,0.95]) )
        n = size(x,1)
        for j in 1:n-1
            if typeof(x[j]) == Float64
                body = string(body, "$(round(x[j],digits=2)) & ")
            else
                body = string(body, "$(x[j]) & ")
            end
        end
        body = string(body, "$(round(x[n],digits=2)) \\\\\n")
    end
    bottom = "\\bottomrule\n\\end{tabular}\n\\end{center}\n\\label{estimates}\n\\begin{flushleft}
\\footnotesize{Para(1) and Para(2) correspond to mean and standard deviation of the prior distribution if the latter is Normal or Inverse Gamma. Para(1) and Para(2) correspond to lower and upper bound of the prior distribution when the latter is uniform}\n\\end{flushleft}\n\\end{table}"

    return println(string(top,body,bottom))

end


"Plots the impulse response functions of a dual model."
function irf(μ::Vector, shock, M::EstLinMod ; n_periods = 24)

	M.param .= μ
	solve!(M)

	ss_x = M.model.ss_x
	ss_y = M.model.ss_y

	n_x = length(keys(ss_x))
	n_y = length(keys(ss_y))

    irf_x = zeros(n_x, n_periods)
	irf_y = zeros(n_y, n_periods)

	cal = M.model.calibrated
	param = M.param

	Macros.@load cal σᶻ β σ η ϵ F b s δ ρ m γ gsY
	Dual.@spec_funs
	Macros.@load_vec param ρ_A ρ_g ρ_R ρ_π ρ_y ψ σ_A σ_μ σ_g σ_m
	Macros.@index np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	Macros.@index c zp zf θ ϕ v Y zs π F_A
	Macros.@load_cat ss ss_x np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	Macros.@load_cat ss ss_y c zp zf θ ϕ v Y zs π F_A

    impulse = zeros(n_x)
	impulse[shock] = 1

    for t in 1:n_periods
		irf_y[:,t] = M.g_x*impulse
        impulse = M.h_x*impulse
        irf_x[:,t] = impulse
    end

    t = 0:n_periods-1

    # R, π, Y
    p1 = plot(t, irf_y[Y,:], label= L"$Y$", color= "black", linestyle = :solid, linewidth=2)
    plot!(t, irf_x[R,:], label= L"$R$", color= "black", linestyle = :dash, linewidth=2)
    plot!(t, irf_y[π,:], label= L"$\pi$", color= "black", linestyle = :dot, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)

    # θ, share of TC in JC
    c_zs = PDF(zs_ss)*zs_ss/(CDF(zs_ss)-CDF(zf_ss))
    c_zf = -(1-CDF(zs_ss))*PDF(zf_ss)*zf_ss/(1-CDF(zf_ss))/(CDF(zs_ss)-CDF(zf_ss))
    p2 = plot(t, irf_y[θ,:], label=L"$\theta$", color= "black", linestyle = :solid, linewidth=2)
    plot!(t, c_zs*irf_y[zs,:]+c_zf*irf_y[zf,:], label = L"$jc^f / jc$", color= "black", linestyle = :dash, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)

    # Employments
    p3 = plot(t, irf_x[np,:], label=L"$n^p$", color= "black", linestyle = :solid, linewidth=2)
    plot!(t, irf_x[nf,:], label=L"$n^f$", color= "black", linestyle = :dash, linewidth=2)
    plot!(t, -(np_ss*irf_x[np,:]+nf_ss*irf_x[nf,:])/(1-np_ss-nf_ss), label=L"$u$", color= "black", linestyle = :dot, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
    ylabel!("Deviation (%)")

    # Thresholds
    p4 = plot(t, irf_y[zp,:], label=L"$z^p$", color= "black", linestyle = :solid, linewidth=2)
    plot!(t, irf_y[zf,:], label=L"$z^f$", color= "black", linestyle = :dash, linewidth=2)
    plot!(t, irf_y[zs,:], label=L"$z^*$", color= "black", linestyle = :dot, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)

    # Job creation
    p5 = plot(t, irf_y[v,:]-σ*irf_y[θ,:]-zs_ss*PDF(zs_ss)*irf_y[zs,:]/(1-CDF(zs_ss)), label=L"$jc^p$", color= "black", linestyle = :solid, linewidth=2)
    plot!(t, irf_y[v,:]-σ*irf_y[θ,:]+(zs_ss*PDF(zs_ss)*irf_y[zs,:] - zf_ss*PDF(zf_ss)*irf_y[zf,:])/(CDF(zs_ss)-CDF(zf_ss)),label=L"$jc^f$", linestyle = :dash, linewidth=2, color = "black", legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
    xlabel!("Quarters")

    # Job destruction
    p6 = plot(t, vcat(0,irf_x[np,1:n_periods-1]) + (1-s)*PDF(zp_ss)*zp_ss*irf_y[zp,:]/(s + (1-s)*CDF(zp_ss)), label=L"$jd^p$", color= "black", linestyle = :solid, linewidth=2)
    plot!(t, vcat(0,irf_x[nf,1:n_periods-1]), label=L"$jd^f$", color= "black", linestyle = :dash, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
    xlabel!("Quarters")

    graphe = plot(p1,p2,p3,p4,p5,p6,layout=(3,2))

    return graphe

end


"Plots the impulse response functions of a classic model."
function irf_classic(μ::Vector, shock, M::EstLinMod ; n_periods = 24)

	M.param .= μ
	solve!(M)

	ss_x = M.model.ss_x
	ss_y = M.model.ss_y

	n_x = length(keys(ss_x))
	n_y = length(keys(ss_y))

    irf_x = zeros(n_x, n_periods)
	irf_y = zeros(n_y, n_periods)

	cal = M.model.calibrated
	param = M.param

	Macros.@load cal σᶻ β σ η ϵ F b s m γ gsY
	Dual.@spec_funs
	Macros.@load_vec param ρ_A ρ_g ρ_R ρ_π ρ_y ψ σ_A σ_μ σ_g σ_m
	Macros.@index np R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	Macros.@index c zp θ ϕ v Y zc π F_A
	Macros.@load_cat ss ss_x np R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	Macros.@load_cat ss ss_y c zp θ ϕ v Y zc π F_A

    impulse = zeros(n_x)
	impulse[shock] = 1

    for t in 1:n_periods
		irf_y[:,t] = M.g_x*impulse
        impulse = M.h_x*impulse
        irf_x[:,t] = impulse
    end

    t = 0:n_periods-1

    # R, π, Y
    p1 = plot(t, irf_y[Y,:], label= L"$Y$", color= "black", linestyle = :solid, linewidth=2)
    plot!(t, irf_x[R,:], label= L"$R$", color= "black", linestyle = :dash, linewidth=2)
    plot!(t, irf_y[π,:], label= L"$\pi$", color= "black", linestyle = :dot, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)

    # θ
    p2 = plot(t, irf_y[θ,:], label=L"$\theta$", color= "black", linestyle = :solid, linewidth=2)
	plot!(t, -np_ss*irf_x[np,:]/(1-np_ss), label=L"$u$", color= "black", linestyle = :dash, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)

    # Thresholds
    p3 = plot(t, irf_y[zp,:], label=L"$z^p$", color= "black", linestyle = :solid, linewidth=2)
    plot!(t, irf_y[zc,:], label=L"$z^c$", color= "black", linestyle = :dash, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
	ylabel!("Deviation (%)")
	xlabel!("Quarters")

    # Job creation
    p4 = plot(t, irf_y[v,:]-σ*irf_y[θ,:]-zc_ss*PDF(zc_ss)*irf_y[zc,:]/(1-CDF(zc_ss)), label=L"$jc$", color= "black", linestyle = :solid, linewidth=2)
	plot!(t, vcat(0,irf_x[np,1:n_periods-1]) + (1-s)*PDF(zp_ss)*zp_ss*irf_y[zp,:]/(s + (1-s)*CDF(zp_ss)), label=L"$jd$", color= "black", linestyle = :dash, linewidth=2)
    xgrid!(:off)
    ygrid!(:off)
    xlabel!("Quarters")

    graphe = plot(p1,p2,p3,p4,layout=(2,2))

    return graphe

end

"Plots prior and posterior distributions of estimated parameters"
function prior_post_plot(S::EstSMC, param_latex ; n = 100)

    n_param = size(S.draws,2)
    graphe = Vector{Any}(undef, n_param)

    for k in 1:n_param

        graphe[k] = histogram(S.draws[:,k], label = "", color = "white", bins = :auto, weights = S.weights, normalize = :pdf)
        x = range(minimum(S.draws[:,k]), stop = maximum(S.draws[:,k]), length = n)
        p = S.EstM.prior[k]
        f(x) = pdf(p, x)
        plot!(x, f.(x), label = "", color = "black", linestyle = :solid, linewidth=2, legend = :outertopright)
        xgrid!(:off)
        ygrid!(:off)
        title!(param_latex[k])

    end

    k = n_param
    graphe[k] = histogram(S.draws[:, k], label = "Posterior", color = "white", bins = :auto, weights = S.weights, normalize = :pdf)
    x = range(minimum(S.draws[:,k]), stop = maximum(S.draws[:,k]), length = n)
	p = S.EstM.prior[k]
    f(x) = pdf(p, x)
    plot!(x, f.(x), label = "Prior", color = "black", linestyle = :solid, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
    title!(param_latex[k])

    return plot(graphe...)
end

"Returns the table of estimated parameters for different filters."
function table_filter_robust(μ_h, μ_lt, μ_fd, μ_hp, σ_h, σ_lt, σ_fd, σ_hp, param_labels)
	n_param = length(μ_h)
    Head = "\\begin{table}
\\begin{center}
\\begin{tabular}{ccccccccc}
\\toprule
Parameters & \\multicolumn{2}{c}{Hamilton} & \\multicolumn{2}{c}{Linear trend} & \\multicolumn{2}{c}{First difference} & \\multicolumn{2}{c}{Hodrick-Prescott}\\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} & Mean & Std. Dev. &  Mean & Std. Dev. &  Mean & Std. Dev. &  Mean & Std. Dev.\\\\ \\midrule \n"
    Body = ""
    for i in 1:n_param
        Body = string(Body, "$(param_labels[i]) & $(round(μ_h[i],digits=2)) & $(round(σ_h[i],digits=2)) & $(round(μ_lt[i],digits=2)) & $(round(σ_lt[i],digits=2)) & $(round(μ_fd[i],digits=2)) & $(round(σ_fd[i],digits=2)) & $(round(μ_hp[i],digits=2)) & $(round(σ_hp[i],digits=2))\\\\ \n")
    end
    Foot = "\\bottomrule \n\\end{tabular}\n\\end{center}\n\\caption{Estimations with Hamilton, linear-trend, first-difference and Hodrick-Prescott filters}\n\\end{table}"
    println(string(Head,Body,Foot))
end

"Plots the labor market moments across time."
function plot_lm_series(data,b,e)
    p0 = plot(b[:y]:e[:y],data[b[:y]:e[:y],:y], label=L"$Y$", linewidth = 2, color = "black", linestyle = :solid)
    plot!(b[:n]:e[:n],data[b[:n]:e[:n],:n], label=L"$n$", linewidth = 2, color = "black", linestyle = :dash, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
    xticks!(b[:y]:8:e[:y], data[b[:y]:8:e[:y],:q], rotation=45)
    # %%
    q1 = minimum([b[:jcp],b[:jcf]])
    q2 = maximum([e[:jcp],e[:jcf]])
    p1 = plot(b[:jcp]:e[:jcp],data[b[:jcp]:e[:jcp],:jcp], label=L"$jc^p$", linewidth = 2, color = "black", linestyle = :solid)
    plot!(b[:jcf]:e[:jcf],data[b[:jcf]:e[:jcf],:jcf], label=L"$jc^f$", linewidth = 2, color = "black", linestyle = :dash)
    plot!(q1:q2,data[q1:q2,:y], label=L"$Y$", linewidth = 2, color = "black", linestyle = :dot, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
    xticks!(q1:8:q2, data[q1:8:q2,:q], rotation=45)
    # %%
    q1 = minimum([b[:jdp],b[:jdf]])
    q2 = maximum([e[:jdp],e[:jdf]])
    p2 = plot(b[:jdp]:e[:jdp],data[b[:jdp]:e[:jdp],:jdp], label=L"$jd^p$", linewidth = 2, color = "black", linestyle = :solid)
    plot!(b[:jdf]:e[:jdf],data[b[:jdf]:e[:jdf],:jdf], label=L"$jd^f$", linewidth = 2, color = "black", linestyle = :dash)
    plot!(q1:q2,data[q1:q2,:y], label=L"$Y$", linewidth = 2, color = "black", linestyle = :dot, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
    xticks!(q1:8:q2, data[q1:8:q2,:q], rotation=45)
    # %%
    q1 = minimum([b[:nf],b[:mufsmu]])
    q2 = maximum([e[:nf],e[:mufsmu]])
    p3 = plot(b[:nf]:e[:nf],data[b[:nf]:e[:nf],:nf], label=L"$n^f$", linewidth = 2, color = "black", linestyle = :solid)
    plot!(b[:mufsmu]:e[:mufsmu],data[b[:mufsmu]:e[:mufsmu],:mufsmu], label=L"$\mu^f / \left( \mu^p + \mu^f \right)$", linewidth = 2, color = "black", linestyle = :dash)
    plot!(q1:q2,data[q1:q2,:y], label=L"$Y$", linewidth = 2, color = "black", linestyle = :dot, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
    xticks!(q1:8:q2, data[q1:8:q2,:q], rotation=45)
    # %%
    graphe = plot(p0,p1,p2,p3,layout = (2,2))
    return graphe
end


"Plots estimation data."
function plot_estimation_data(data,b,e)

    graphe = plot(b[:y]:e[:y],data[b[:y]:e[:y],:y], label=L"$Y$", linewidth = 1, color = "black", linestyle = :solid)
    plot!(b[:n]:e[:n],data[b[:n]:e[:n],:n], label=L"$n$", linewidth = 1, color = "black", linestyle = :dash)
    plot!(b[:R]:e[:R],data[b[:R]:e[:R],:R], label=L"$R$", linewidth = 1, color = "black", linestyle = :dot)
    plot!(b[:pi]:e[:pi],data[b[:pi]:e[:pi],:pi], label=L"$\pi$", linewidth = 1, color = "black", linestyle = :dashdot, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
    xticks!(b[:R]:4:e[:R], data[b[:R]:4:e[:R],:q], rotation=45)

    return graphe

end

"Computes covariances and correlations for the data."
function cov_cor_obs(data,b,e;h=5)

    n_var = size(names(data),1)-1
    std_obs = zeros(n_var)
    cov_obs = zeros(n_var,n_var,h+1)
    cor_obs = zeros(n_var,n_var,h+1)

    for i in 1:n_var
        v = names(data)[i+1]
        std_obs[i] = std(data[b[v]:e[v],v])
        for j in 1:n_var
            w = names(data)[j+1]
            for k in 0:h
                q1 = max(b[v],b[w])
                q2 = min(e[v],e[w])
                cov_obs[i,j,k+1] = cov(data[q1:q2-k,v],data[q1+k:q2,w])
                cor_obs[i,j,k+1] = cor(data[q1:q2-k,v],data[q1+k:q2,w])
            end
        end
    end

    return (std_obs, cov_obs, cor_obs)

end


"Simulates draws for a given model."
function simul_draws(EstM::EstLinMod, draws::SharedArray, n_part::Int ; T=500)

    println("Initialization...")

	n_x = length(keys(EstM.model.ss_x))
	n_y = length(keys(EstM.model.ss_y))
	n_ϵ = EstM.model.n_ϵ
	e = SharedArray{Float64}((n_part, T, n_ϵ))
	e .= randn(n_part, T, n_ϵ)
    sim_x = SharedArray{Float64}((n_part, T, n_x))
	sim_y = SharedArray{Float64}((n_part, T, n_y))
    sim_x[:,1,:] .= 0.

    println("Simulation...")

    @sync @distributed for n in 1:n_part
		EstM.param .= draws[n,:]
		solve!(EstM)
		state_space!(EstM)
        for t in 1:T-1
            sim_x[n,t+1,:] .= EstM.G*sim_x[n,t,:] + EstM.R*e[n,t,:]
			sim_y[n,t+1,:] .= EstM.g_x*sim_x[n,t+1,:]
        end
    end
	#Removing the lags between series
    return (convert(SharedArray,sim_x[:,3:end,:]), convert(SharedArray,sim_y[:,2:end-1,:]))

end

"Computes covariances and correlations from simulations."
function cov_cor_draws(M::EstLinMod, sim_x::SharedArray, sim_y::SharedArray, w ; Tburn=100, lags=5)

    println("Initialization...")
    n_part, T , _ = size(sim_x)

	n_data = 13
    sim_obs = SharedArray{Float64}(n_part, T, n_data)

	ss_x = M.model.ss_x
	ss_y = M.model.ss_y
	cal = M.model.calibrated
	param = M.param

	Macros.@load cal σᶻ β σ η ϵ F b s δ ρ m γ gsY
	Dual.@spec_funs
	Macros.@load_vec param ρ_A ρ_g ρ_R ρ_π ρ_y ψ σ_A σ_μ σ_g σ_m
	Macros.@index np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	Macros.@index c zp zf θ ϕ v Y zs π F_A
	Macros.@load_cat ss ss_x np nf R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	Macros.@load_cat ss ss_y c zp zf θ ϕ v Y zs π F_A

    println("Deriving the counterpart of observables...")

	q_ss = m*θ_ss^(-σ)
	jcp_ss = q_ss*(1-CDF(zs_ss))*v_ss
	jcf_ss = q_ss*(CDF(zs_ss)-CDF(zf_ss))*v_ss
	jc_ss = jcp_ss + jcf_ss

	jdp_ss = (s+(1-s)*CDF(zp_ss))*np_ss
	jdf_ss = δ*nf_ss
	jd_ss = jdp_ss + jdf_ss
	n_ss = np_ss + nf_ss

    @views @. begin

		sim_obs[:,:,1] = sim_y[:,:,Y] #output
		sim_obs[:,:,2] = sim_y[:,:,π] #inflation
		sim_obs[:,:,3] = sim_x[:,:,R] #interest rates
		sim_obs[:,:,4] = (np_ss*sim_x[:,:,np] + nf_ss*sim_x[:,:,nf])/n_ss #employment
		sim_obs[:,:,5] = sim_x[:,:,nf]
		sim_obs[:,:,6] = sim_y[:,:,v] - σ*sim_y[:,:,θ] - zs_ss*PDF(zs_ss)*sim_y[:,:,zs] / (1-CDF(zs_ss))  #permanent job creation
		sim_obs[:,:,7] = sim_y[:,:,v] - σ*sim_y[:,:,θ] + zs_ss*PDF(zs_ss)*sim_y[:,:,zs] / (CDF(zs_ss)-CDF(zf_ss)) - zf_ss*PDF(zf_ss)*sim_y[:,:,zf] / (CDF(zs_ss)-CDF(zf_ss))
		sim_obs[:,:,8] = (sim_obs[:,:,6]*jcp_ss + sim_obs[:,:,7]*jcf_ss)/(jcp_ss + jcf_ss) #job creation
		sim_obs[:,:,9] = PDF(zs_ss)*zs_ss*sim_y[:,:,zs]/(CDF(zs_ss)-CDF(zf_ss)) + (1/(1-CDF(zf_ss)) - 1/(CDF(zs_ss)-CDF(zf_ss)))*PDF(zf_ss)*zf_ss*sim_y[:,:,zf]
		sim_obs[:,2:end,10] = sim_x[:,1:end-1,np] + PDF(zp_ss)*zp_ss*sim_y[:,2:end,zp]/CDF(zp_ss) #jdp
		sim_obs[:,2:end,11] = sim_x[:,1:end-1,nf] #jdf
		sim_obs[:,2:end,12] = (jdp_ss*sim_x[:,1:end-1,np] + (1-s)*PDF(zp_ss)*zp_ss*np_ss*sim_y[:,2:end,zp] + jdf_ss*sim_obs[:,2:end,11])/jd_ss #jd
		sim_obs[:,:,13] = sim_y[:,:,v]

	end

    println("Computing standard deviations...")

    s = reshape(std(sim_obs[:,Tburn:end,:], dims = 2), n_part, n_data)
    std_sim_mean = reshape(mean(s, Weights(w), dims = 1), n_data)
    std_sim_q = [ quantile(s[:,j], Weights(w), [0.025,0.975]) for j in 1:n_data ]

    println("Computing covariances and correlations for each draw...")
    cor_sim = SharedArray{Float64}((n_part, n_data, n_data, lags+1))
    cov_sim = SharedArray{Float64}((n_part, n_data, n_data, lags+1))
    @sync @distributed for n in 1:n_part
        for i in 1:n_data
            for j in 1:n_data
                for k in 0:lags
                    cor_sim[n,i,j,k+1] = StatsBase.cor(sim_obs[n,Tburn:T-k,i], sim_obs[n,Tburn+k:T,j])
                    cov_sim[n,i,j,k+1] = StatsBase.cov(sim_obs[n,Tburn:T-k,i], sim_obs[n,Tburn+k:T,j])
                end
            end
        end
    end

    println("Computing covariances and correlations: means and quantiles...")

    cor_sim_mean = zeros(n_data, n_data,lags+1)
    cor_sim_q = zeros(n_data, n_data,lags+1, 2)
    cov_sim_mean = zeros(n_data, n_data,lags+1)
    cov_sim_q = zeros(n_data, n_data,lags+1, 2)

    for i in 1:n_data
        for j in 1:n_data
            for k in 0:lags
                cor_sim_mean[i,j,k+1] = mean(cor_sim[:,i,j,k+1], Weights(w))
                cor_sim_q[i,j,k+1,:] .= quantile(cor_sim[:,i,j,k+1], Weights(w), [0.025, 0.975])
                cov_sim_mean[i,j,k+1] = mean(cov_sim[:,i,j,k+1], Weights(w))
                cov_sim_q[i,j,k+1,:] .= quantile(cov_sim[:,i,j,k+1], Weights(w), [0.025, 0.975])
            end
        end
    end

    return (std_sim_mean, std_sim_q, cor_sim_mean, cor_sim_q, cov_sim_mean, cov_sim_q)

end


"Computes covariances and correlations from simulations (classic model)."
function cov_cor_classic_draws(M::EstLinMod, sim_x::SharedArray, sim_y::SharedArray, w ; Tburn=100, lags=5)

    println("Initialization...")
    n_part, T , _ = size(sim_x)

	n_data = 7
    sim_obs = SharedArray{Float64}(n_part, T, n_data)

	ss_x = M.model.ss_x
	ss_y = M.model.ss_y
	cal = M.model.calibrated
	param = M.param

	Macros.@load cal σᶻ β σ η ϵ F b s m γ gsY
	Dual.@spec_funs
	Macros.@load_vec param ρ_A ρ_g ρ_R ρ_π ρ_y ψ σ_A σ_μ σ_g σ_m
	Macros.@index np R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	Macros.@index c zp θ ϕ v Y zc π F_A
	Macros.@load_cat ss ss_x np R g A ϵ_A ϵ_μ ϵ_g ϵ_m
	Macros.@load_cat ss ss_y c zp θ ϕ v Y zc π F_A

    println("Deriving the counterpart of observables...")

    @views @. begin

		sim_obs[:,:,1] = sim_y[:,:,Y] #output
		sim_obs[:,:,2] = sim_y[:,:,π] #inflation
		sim_obs[:,:,3] = sim_x[:,:,R] #interest rates
		sim_obs[:,:,4] = sim_x[:,:,np] #employment
		sim_obs[:,:,5] = sim_y[:,:,v] - σ*sim_y[:,:,θ] - zc_ss*PDF(zc_ss)*sim_y[:,:,zc] / (1-CDF(zc_ss))  #permanent job creation
		sim_obs[:,2:end,6] = sim_x[:,1:end-1,np] + PDF(zp_ss)*zp_ss*sim_y[:,2:end,zp]/CDF(zp_ss) #jdp
		sim_obs[:,:,7] = sim_y[:,:,v] #vacancies

	end

    println("Computing standard deviations...")

    s = reshape(std(sim_obs[:,Tburn:end,:], dims = 2), n_part, n_data)
    std_sim_mean = reshape(mean(s, Weights(w), dims = 1), n_data)
    std_sim_q = [ quantile(s[:,j], Weights(w), [0.025,0.975]) for j in 1:n_data ]

    println("Computing covariances and correlations for each draw...")
    cor_sim = SharedArray{Float64}((n_part, n_data, n_data, lags+1))
    cov_sim = SharedArray{Float64}((n_part, n_data, n_data, lags+1))
    @sync @distributed for n in 1:n_part
        for i in 1:n_data
            for j in 1:n_data
                for k in 0:lags
                    cor_sim[n,i,j,k+1] = StatsBase.cor(sim_obs[n,Tburn:T-k,i], sim_obs[n,Tburn+k:T,j])
                    cov_sim[n,i,j,k+1] = StatsBase.cov(sim_obs[n,Tburn:T-k,i], sim_obs[n,Tburn+k:T,j])
                end
            end
        end
    end

    println("Computing covariances and correlations: means and quantiles...")

    cor_sim_mean = zeros(n_data, n_data,lags+1)
    cor_sim_q = zeros(n_data, n_data,lags+1, 2)
    cov_sim_mean = zeros(n_data, n_data,lags+1)
    cov_sim_q = zeros(n_data, n_data,lags+1, 2)

    for i in 1:n_data
        for j in 1:n_data
            for k in 0:lags
                cor_sim_mean[i,j,k+1] = mean(cor_sim[:,i,j,k+1], Weights(w))
                cor_sim_q[i,j,k+1,:] .= quantile(cor_sim[:,i,j,k+1], Weights(w), [0.025, 0.975])
                cov_sim_mean[i,j,k+1] = mean(cov_sim[:,i,j,k+1], Weights(w))
                cov_sim_q[i,j,k+1,:] .= quantile(cov_sim[:,i,j,k+1], Weights(w), [0.025, 0.975])
            end
        end
    end

    return (std_sim_mean, std_sim_q, cor_sim_mean, cor_sim_q, cov_sim_mean, cov_sim_q)

end

"Computes cross covariances and correlations for data."
function cov_cor_obs(data,b,e;h=5)

    n_var = size(names(data),1)-1
    std_obs = zeros(n_var)
    cov_obs = zeros(n_var,n_var,h+1)
    cor_obs = zeros(n_var,n_var,h+1)

    for i in 1:n_var
        v = names(data)[i+1]
        std_obs[i] = std(data[b[v]:e[v],v])
        for j in 1:n_var
            w = names(data)[j+1]
            for k in 0:h
                q1 = max(b[v],b[w])
                q2 = min(e[v],e[w])
                cov_obs[i,j,k+1] = cov(data[q1:q2-k,v],data[q1+k:q2,w])
                cor_obs[i,j,k+1] = cor(data[q1:q2-k,v],data[q1+k:q2,w])
            end
        end
    end

    return (std_obs, cov_obs, cor_obs)

end

"Returns the table with labor market moments."
function table_lm_moments(std_obs,std_sim,cor_obs,cor_sim,lm_latex)
    top = "\\begin{table}[H]\n\\begin{center}\n\\begin{tabular}{ccccc}\n\\toprule\n Variables & \\multicolumn{2}{c}{Data} & \\multicolumn{2}{c}{Model} \\\\ \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} & Std. Dev. & \$Cor\\left( Y, . \\right)\$ & Std. Dev. & \$Cor\\left( Y, . \\right)\$ \\\\ \\midrule \n"
    bottom = "\\bottomrule\n\\end{tabular}\n\\end{center}\n\\end{table}"
    body = ""
    for i in 1:size(lm_latex,1)
        body = string(body, "$(lm_latex[i]) & $(round(std_obs[i], digits=2)) & $(round(cor_obs[1,i,1], digits=2)) & $(round(std_sim[i], digits=2)) & $(round(cor_sim[1,i,1], digits=2)) \\\\ \n")
    end
    table = string(top,body,bottom)
    return println(table)
end

"Returns the table with labor market moments with quantiles."
function table_lm_moments_withq(std_obs, std_sim_mean, std_sim_q, cor_obs, cor_sim_mean, cor_sim_q, lm_latex)
    top = "\\begin{table}[H]\n\\begin{center}\n\\begin{tabular}{ccccccccc}\n\\toprule\nVariables & \\multicolumn{2}{c}{Data} & \\multicolumn{6}{c}{Model} \\\\\n\\cmidrule(lr){2-3} \\cmidrule(lr){4-9} & Std. Dev. & \$Cor\\left( Y, . \\right)\$ & \\multicolumn{3}{c}{Std. Dev.} &  \\multicolumn{3}{c}{\$Cor\\left( Y, . \\right)\$} \\\\\n\\cmidrule(lr){4-6} \\cmidrule(lr){7-9} & & & Mean & 2.5 \\% & 97.5 \\% & Mean & 2.5 \\% & 97.5 \\% \\\\ \\midrule\n"
    bottom = "\\bottomrule\n\\end{tabular}\n\\end{center}\n\\end{table}"
    body = ""
    for i in 1:size(lm_latex,1)
        body = string(body, "$(lm_latex[i]) & $(round(std_obs[i], digits=2)) & $(round(cor_obs[1,i,1], digits=2)) & $(round(std_sim_mean[i], digits=2)) & $(round(std_sim_q[i][1], digits=2)) & $(round(std_sim_q[i][2], digits=2)) & $(round(cor_sim_mean[1,i,1], digits=2)) & $(round(cor_sim_q[1,i,1,1], digits=2)) & $(round(cor_sim_q[1,i,1,2], digits=2)) \\\\ \n")
    end
    table = string(top,body,bottom)
    return println(table)
end

"Returns the table with labor market moments."
function table_lm_moments_comp(std_obs, cor_obs, std_sim, cor_sim, std_sim2, cor_sim2, lm_latex)
    top = "\\begin{table}[H]\n\\begin{center}\n\\begin{tabular}{ccccccc}\n\\toprule\n Variables & \\multicolumn{2}{c}{Data} & \\multicolumn{2}{c}{Baseline}  & \\multicolumn{2}{c}{Reduced firing costs} \\ \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} &  Std. Dev. & \$Cor\\left( Y, . \\right)\$ & Std. Dev. & \$Cor\\left( Y, . \\right)\$ & Std. Dev. & \$Cor\\left( Y, . \\right)\$\\ \\midrule \n"
    bottom = "\\bottomrule\n\\end{tabular}\n\\end{center}\n\\end{table}"
    body = ""
    for i in 1:size(lm_latex,1)
        body = string(body, "$(lm_latex[i]) & $(round(std_obs[i], digits=2)) & $(round(cor_obs[1,i,1], digits=2)) & $(round(std_sim[i], digits=2)) & $(round(cor_sim[1,i,1], digits=2)) & $(round(std_sim2[i], digits=2)) & $(round(cor_sim2[1,i,1], digits=2)) \\\\ \n")
    end
    table = string(top,body,bottom)
    return println(table)
end

"Plots the cross correlations of labor market moments"
function plot_cor(cor_obs, cor_sim_mean, cor_sim_q, lm_names)
    n_data = size(cor_sim_mean,1)
    h = size(cor_sim_mean,3)
    mat_graphe = []
    for i in 1:n_data
        for j in 1:n_data
            p = plot(0:h-1,cor_sim_mean[i,j,:], label = "", color="black", linewidth = 2, linestyle = :solid)
            plot!(0:h-1,cor_sim_q[i,j,:,:], label = "", color="black", linewidth = 2, linestyle = :dash)
            plot!(0:h-1,cor_obs[i,j,:], label = "", color="black", linewidth = 2, linestyle = :dot)
            xgrid!(:off)
            ygrid!(:off)
            title!(string("\$",lm_names[i],"_{t}\$ \$",lm_names[j],"_{t+h}\$"))
            append!(mat_graphe,[p])
        end
    end
    return mat_graphe
end
