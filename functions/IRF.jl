#Stochastic steady-state
function sto_ss(O3::Order3; T_burn = 1000, T_ss=1000)

    T = T_burn + T_ss

    n_x = O3.O2.O1.M.n_x
    n_y = O3.O2.O1.M.n_y
    h_x = O3.O2.O1.h_x
    h_xx = O3.O2.h_xx
    h_xxx = O3.h_xxx

    x = zeros(n_x, T)
    x[end,1] = 100.
    for t in 2:T
        x[:,t] .= h_x * x[:, t-1] + 0.5*h_xx*kron_pow(x[:, t-1], 2) + h_xxx*kron_pow(x[:, t-1], 3)/6
    end

    return x

end

"Computes stochastic steady-state with pruning using simulations"
function sto_ss_pruning(O3::Order3; T = 1000)

    n_x = O3.O2.O1.M.n_x
    n_y = O3.O2.O1.M.n_y
    h_x = O3.O2.O1.h_x
    h_xx = O3.O2.h_xx
    h_xxx = O3.h_xxx

    g_x = O3.O2.O1.g_x
    g_xx = O3.O2.g_xx
    g_xxx = O3.g_xxx

    x_f = spzeros(n_x, T)
    x_s = spzeros(n_x, T)
    x_rd = spzeros(n_x, T)

    y_rd = spzeros(n_y, T)

    x_f[end,1] = 1.

    for t in 2:T
        x_f[:, t] .= h_x*x_f[:, t-1]
        x_s[:, t] .= h_x*x_s[:, t-1] + 0.5*h_xx*kron_pow(x_f[:, t-1], 2)
        x_rd[:, t] .= h_x*x_rd[:, t-1] + h_xx*kron(x_f[:, t-1], x_s[:, t-1]) + h_xxx*kron_pow(x_f[:, t-1], 3)/6

        y_rd[:, t] .= g_x*(x_f[:, t] + x_s[:, t] + x_rd[:, t]) + 0.5*g_xx*(kron_pow(x_f[:, t], 2) + 2*kron(x_f[:, t], x_s[:, t]) ) + g_xxx*kron_pow(x_f[:, t], 3)/6

    end

    return (x_f[:, end], x_s[:, end], x_rd[:, end], y_rd[:,end])

end

"Computes IRFs with the stochastic steady-state as starting point."
function irf(O3::Order3, shock::Int, m::Tuple{Vararg{SparseVector, 4}} ; T = 24)

    η = O3.O2.O1.M.η
    n_x = O3.O2.O1.M.n_x
    n_y = O3.O2.O1.M.n_y
    n_ϵ = size(η,2)

    g_x = O3.O2.O1.g_x
    h_x = O3.O2.O1.h_x
    g_xx = O3.O2.g_xx
    h_xx = O3.O2.h_xx
    g_xxx = O3.g_xxx
    h_xxx = O3.h_xxx

    ϵ = spzeros(n_ϵ, T)
    ϵ[shock, 1] = 1.

    x_f = spzeros(n_x, T)
    x_s = spzeros(n_x, T)
    x_rd = spzeros(n_x, T)
    y_rd = spzeros(n_y, T)

    x_f[:,1], x_s[:,1], x_rd[:,1], y_rd[:,1] = m

    for t in 2:T

        x_f[:, t] .= h_x * x_f[:, t-1] + η * ϵ[:, t-1]
        x_s[:, t] .= h_x * x_s[:, t-1] + 0.5 * h_xx * kron_pow(x_f[:, t-1], 2)
        x_rd[:, t] .= h_x * x_rd[:, t-1] + h_xx * kron(x_f[:, t-1], x_s[:, t-1]) + h_xxx*kron_pow(x_f[:, t-1], 3)/6

        y_rd[:, t] .= g_x*(x_f[:, t] + x_s[:, t] + x_rd[:, t]) + 0.5*g_xx*(kron_pow(x_f[:, t], 2) + 2*kron(x_f[:, t], x_s[:, t]) ) + g_xxx*kron_pow(x_f[:, t], 3)/6

    end

    for x in (x_f, x_s, x_rd, y_rd)
        x .= x - repeat(x[:, 1], outer=(1,T))
    end

    return ((x_f + x_s + x_rd)[:, 3:end], y_rd[:, 2:end])

end

"Plots the IRF using the simulations x and y."
function plot_irf(x, y, cal, ss_x, ss_y)

    Macros.@load cal σᶻ β σ η ϵ F b s δ ρ m γ gsY
    SymDual.@spec_funs
    Macros.@load_cat ss ss_x np nf R g A
    Macros.@load_cat ss ss_y c zp zf θ ϕ v Y zs
    Macros.@index np nf R g A Δ
    Macros.@index c zp zf θ ϕ v Y zs π π_s F_A P_n P_d
    n_periods = size(x, 2)

    # R, π, Y
    p1 = plot(y[Y,:], label= L"$Y$", color= "black", linestyle = :solid, linewidth=2)
    plot!(x[R,:], label= L"$R$", color= "black", linestyle = :dash, linewidth=2)
    plot!(y[π,:], label= L"$\pi$", color= "black", linestyle = :dot, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)

    # θ, share of TC in JC
    c_zs = PDF(zs_ss)*zs_ss/(CDF(zs_ss)-CDF(zf_ss))
    c_zf = -(1-CDF(zs_ss))*PDF(zf_ss)*zf_ss/(1-CDF(zf_ss))/(CDF(zs_ss)-CDF(zf_ss))
    p2 = plot(y[θ,:], label=L"$\theta$", color= "black", linestyle = :solid, linewidth=2)
    plot!(c_zs*y[zs,:]+c_zf*y[zf,:], label = L"$jc^f / jc$", color= "black", linestyle = :dash, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)

    # Employments
    p3 = plot(x[np,:], label=L"$n^p$", color= "black", linestyle = :solid, linewidth=2)
    plot!(x[nf,:], label=L"$n^f$", color= "black", linestyle = :dash, linewidth=2)
    plot!(-(np_ss*x[np,:]+nf_ss*x[nf,:])/(1-np_ss-nf_ss), label=L"$u$", color= "black", linestyle = :dot, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
    ylabel!("Deviation (%)")

    # Thresholds
    p4 = plot(y[zp,:], label=L"$z^p$", color= "black", linestyle = :solid, linewidth=2)
    plot!(y[zf,:], label=L"$z^f$", color= "black", linestyle = :dash, linewidth=2)
    plot!(y[zs,:], label=L"$z^*$", color= "black", linestyle = :dot, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)

    # Job creation
    p5 = plot(y[v,:]-σ*y[θ,:]-zs_ss*PDF(zs_ss)*y[zs,:]/(1-CDF(zs_ss)), label=L"$jc^p$", color= "black", linestyle = :solid, linewidth=2)
    plot!(y[v,:]-σ*y[θ,:]+(zs_ss*PDF(zs_ss)*y[zs,:] - zf_ss*PDF(zf_ss)*y[zf,:])/(CDF(zs_ss)-CDF(zf_ss)),label=L"$jc^f$", linestyle = :dash, linewidth=2, color = "black", legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
    xlabel!("Quarters")

    # Job destruction
    p6 = plot(vcat(0, x[np,1:n_periods-1]) + (1-s)*PDF(zp_ss)*zp_ss*y[zp,1:n_periods]/(s + (1-s)*CDF(zp_ss)), label=L"$jd^p$", color= "black", linestyle = :solid, linewidth=2)
    plot!(vcat(0, x[nf,1:n_periods-1]), label=L"$jd^f$", color= "black", linestyle = :dash, linewidth=2, legend = :outertopright)
    xgrid!(:off)
    ygrid!(:off)
    xlabel!("Quarters")

    graphe = plot(p1,p2,p3,p4,p5,p6,layout=(3,2))

    return graphe

end


# "Computes the irf starting from the ergodic unconditional mean."
# function irf(O3::Order3, shock::Int; N = 10000, T_burn = 100, T_irf = 24)
#
#     η = O3.O2.O1.M.η
#     n_x = O3.O2.O1.M.n_x
#     n_y = O3.O2.O1.M.n_y
#     n_ϵ = size(η,2)
#
#     g_x = O3.O2.O1.g_x
#     h_x = O3.O2.O1.h_x
#     g_xx = O3.O2.g_xx
#     h_xx = O3.O2.h_xx
#     g_xxx = O3.g_xxx
#     h_xxx = O3.h_xxx
#
#     T = T_burn + T_irf
#
#     ϵ = randn(n_ϵ, N, T)
#     impulse = zeros(n_ϵ)
#     impulse[shock] = 1.
#
#     @everywhere begin
#
#         T_burn = $T_burn
#
#         η = $η
#
#         g_x = $g_x
#         h_x = $h_x
#         g_xx = $g_xx
#         h_xx = $h_xx
#         g_xxx = $g_xxx
#         h_xxx = $h_xxx
#
#         ϵ = $ϵ
#         impulse = $impulse
#
#     end
#
#     x_f = SharedArray(zeros(n_x, N, T, 2))
#     x_s = SharedArray(zeros(n_x, N, T, 2))
#     x_rd = SharedArray(zeros(n_x, N, T, 2))
#     y_rd = SharedArray(zeros(n_y, N, T, 2))
#     x_f[end,:,1,:] .= 1
#
#
#     @sync @distributed for n in 1:N
#     # for n in 1:N
#         for t in 2:T
#
#             x_f[:, n, t, 1] .= h_x * x_f[:, n, t-1, 1] + η * ϵ[:, n, t-1]
#             x_s[:, n, t, 1] .= h_x * x_s[:, n, t-1, 1] + 0.5 * h_xx * kron_pow(x_f[:, n, t-1, 1], 2)
#             x_rd[:, n, t, 1] .= h_x * x_rd[:, n, t-1, 1] + h_xx * kron(x_f[:, n, t-1, 1], x_s[:, n, t-1, 1]) + h_xxx*kron_pow(x_f[:, n, t-1, 1], 3)/6
#
#             y_rd[:, n, t, 1] .= g_x*(x_f[:, n, t, 1] + x_s[:, n, t, 1] + x_rd[:, n, t, 1]) + 0.5*g_xx*(kron_pow(x_f[:, n, t, 1], 2) + 2*kron(x_f[:, n, t, 1], x_s[:, n, t, 1]) ) + g_xxx*kron_pow(x_f[:, n, t, 1], 3)/6
#
#
#             x_f[:, n, t, 2] .= h_x * x_f[:, n, t-1, 2] + η * (ϵ[:, n, t-1] + (t==T_burn+2)*impulse)
#             x_s[:, n, t, 2] .= h_x * x_s[:, n, t-1, 2] + 0.5 * h_xx * kron_pow(x_f[:, n, t-1, 2], 2)
#             x_rd[:, n, t, 2] .= h_x * x_rd[:, n, t-1, 2] + h_xx * kron(x_f[:, n, t-1, 2], x_s[:, n, t-1, 2]) + h_xxx*kron_pow(x_f[:, n, t-1, 2], 3)/6
#
#             y_rd[:, n, t, 2] .= g_x*(x_f[:, n, t, 2] + x_s[:, n, t, 2] + x_rd[:, n, t, 2]) + 0.5*g_xx*(kron_pow(x_f[:, n, t, 2], 2) + 2*kron(x_f[:, n, t, 2], x_s[:, n, t, 2]) ) + g_xxx*kron_pow(x_f[:, n, t, 2], 3)/6
#
#         end
#     end
#
#     # println(x_f[10, 1, T_burn+2, 1])
#
#     return (x_f, x_s, x_rd, y_rd)
#
#     # x_tot = (x_f + x_s + x_rd)[:,:,T_burn+3:end,:]
#     # return (x_tot[:,:,:, 2] - x_tot[:,:,:, 1], y_rd[:,:,T_burn+2:end, 2] - y_rd[:,:,T_burn+2:end, 1])
#
# end


function expectation(O3::Order3)

    n_x_sgu = O3.O2.O1.M.n_x_sgu
    n_y = O3.O2.O1.M.n_y
    η_sgu = O3.O2.O1.M.η_sgu
    n_ϵ = size(η,2)

    h_x_sgu = O3.O2.O1.h_x_sgu
    h_xx_sgu = O3.O2.h_xx_sgu
    h_σσ_sgu = O3.O2.h_σσ_sgu
    h_xxx_sgu = O3.h_xxx_sgu

    g_x_sgu = O3.O2.O1.g_x_sgu
    g_xx_sgu = O3.O2.g_xx_sgu
    g_σσ_sgu = O3.O2.g_σσ_sgu

    E_f_2 = (I(n_x_sgu^2) - kron_pow(h_x_sgu, 2)) \ Array(kron_pow(η_sgu,2)*spI(n_ϵ)[:])
    E_x_s = 0.5*(I(n_x_sgu) - h_x_sgu)\ Array(h_xx_sgu*E_f_2 + h_σσ_sgu)
    E_y_s = g_x_sgu*E_x_s + 0.5*g_xx_sgu*E_f_2 + 0.5*g_σσ_sgu

    return E_x_s

end
