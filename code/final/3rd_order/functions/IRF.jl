#Stochastic steady-state
function sto_ss(O3::Order3; T_burn = 1000, T_ss=1000)

    T = T_burn + T_ss

    n_x = O3.O2.O1.M.n_x
    n_y = O3.O2.O1.M.n_y
    h_x = O3.O2.O1.h_x
    h_xx = O3.O2.h_xx
    h_xxx = O3.h_xxx

    x = zeros(n_x, T)
    x[end,1] = 1
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

    x_f[end,1] = 1

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

"Computes the irf starting from the ergodic unconditional mean."
function irf(O3::Order3, shock::Int; N = 10000, T_burn = 100, T_irf = 24)

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

    T = T_burn + T_irf

    ϵ = randn(n_ϵ, N, T)
    impulse = zeros(n_ϵ)
    impulse[shock] = 1.

    @everywhere begin

        T_burn = $T_burn

        η = $η

        g_x = $g_x
        h_x = $h_x
        g_xx = $g_xx
        h_xx = $h_xx
        g_xxx = $g_xxx
        h_xxx = $h_xxx

        ϵ = $ϵ
        impulse = $impulse

    end

    x_f = SharedArray(zeros(n_x, N, T, 2))
    x_s = SharedArray(zeros(n_x, N, T, 2))
    x_rd = SharedArray(zeros(n_x, N, T, 2))
    y_rd = SharedArray(zeros(n_y, N, T, 2))
    x_f[end,:,1,:] .= 1


    @sync @distributed for n in 1:N
    # for n in 1:N
        for t in 2:T

            x_f[:, n, t, 1] .= h_x * x_f[:, n, t-1, 1] + η * ϵ[:, n, t-1]
            x_s[:, n, t, 1] .= h_x * x_s[:, n, t-1, 1] + 0.5 * h_xx * kron_pow(x_f[:, n, t-1, 1], 2)
            x_rd[:, n, t, 1] .= h_x * x_rd[:, n, t-1, 1] + h_xx * kron(x_f[:, n, t-1, 1], x_s[:, n, t-1, 1]) + h_xxx*kron_pow(x_f[:, n, t-1, 1], 3)/6

            y_rd[:, n, t, 1] .= g_x*(x_f[:, n, t, 1] + x_s[:, n, t, 1] + x_rd[:, n, t, 1]) + 0.5*g_xx*(kron_pow(x_f[:, n, t, 1], 2) + 2*kron(x_f[:, n, t, 1], x_s[:, n, t, 1]) ) + g_xxx*kron_pow(x_f[:, n, t, 1], 3)/6


            x_f[:, n, t, 2] .= h_x * x_f[:, n, t-1, 2] + η * (ϵ[:, n, t-1] + (t==T_burn+2)*impulse)
            x_s[:, n, t, 2] .= h_x * x_s[:, n, t-1, 2] + 0.5 * h_xx * kron_pow(x_f[:, n, t-1, 2], 2)
            x_rd[:, n, t, 2] .= h_x * x_rd[:, n, t-1, 2] + h_xx * kron(x_f[:, n, t-1, 2], x_s[:, n, t-1, 2]) + h_xxx*kron_pow(x_f[:, n, t-1, 2], 3)/6

            y_rd[:, n, t, 2] .= g_x*(x_f[:, n, t, 2] + x_s[:, n, t, 2] + x_rd[:, n, t, 2]) + 0.5*g_xx*(kron_pow(x_f[:, n, t, 2], 2) + 2*kron(x_f[:, n, t, 2], x_s[:, n, t, 2]) ) + g_xxx*kron_pow(x_f[:, n, t, 2], 3)/6

        end
    end

    # println(x_f[10, 1, T_burn+2, 1])

    return (x_f, x_s, x_rd, y_rd)

    # x_tot = (x_f + x_s + x_rd)[:,:,T_burn+3:end,:]
    # return (x_tot[:,:,:, 2] - x_tot[:,:,:, 1], y_rd[:,:,T_burn+2:end, 2] - y_rd[:,:,T_burn+2:end, 1])

end


function expectation(O3::Order3)

    n_x = O3.O2.O1.M.n_x
    n_y = O3.O2.O1.M.n_y
    η = O3.O2.O1.M.η
    n_ϵ = size(η,2)

    h_x = O3.O2.O1.h_x
    h_xx = O3.O2.h_xx
    h_xxx = O3.h_xxx

    E_f_2 = (I(n_x^2) - kron_pow(h_x, 2)) \ kron_pow(η,2)*spI(n_ϵ)[:]
    E_x_s = 0.5*(I(n_x) - h_x)\(h_xx*E_f_2)

    return E_x_s

end
