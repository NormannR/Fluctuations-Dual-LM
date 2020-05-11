module FilterData

    using CSV
    using DataFrames
    using Statistics
    using LinearAlgebra

    export linear_detrend, hamilton_detrend, first_difference, hp_detrend, linear_df, hamilton_detrend_df, firstdifference_df, hodrickprescott_df, non_missing, time_bounds, hamilton_filter!, demean!

    #One half of the code is to enforce filtering using common bounds for non -missing data, while the other enforces filters on non-missing data with bounds determined for each time series.

    "Computes the common beginning and ending periods without missing values of a dataframe of time series."
    function non_missing_index(d)
        n,p = size(d)
        lines = ones(Int64,2,p)
        lines[2,:] .= n
        for j = 1:p
            if ismissing(d[1,j])
                lines[1,j] = findfirst(x->!(ismissing(x)),d[:,j])
            end
            if ismissing(d[n,j])
                lines[2,j] = findlast(x->!(ismissing(x)),d[:,j])
            end
        end
        return (maximum(lines[1,:]),minimum(lines[2,:]))
    end

    "Computes the beginning and ending periods without missing values of each time series in `d`"
    function non_missing_index_list(d)
        n,p = size(d)
        b = Dict()
        e = Dict()
        for j in names(d)
            if ismissing(d[1,j])
                b[j] = findfirst(x->!(ismissing(x)),d[:,j])
            else
                b[j] = 1
            end
            if ismissing(d[n,j])
                e[j] = findlast(x->!(ismissing(x)),d[:,j])
            else
                e[j] = n
            end
        end
        return (b,e)
    end

    "Returns values of variables `v` in the dataframe `d` over the biggest common time interval, where the associated values are non-missing."
    function non_missing(d,v)
        b,e = non_missing_index(d[:,v])
        return d[b:e,:]
    end

    "Returns the line indexes associated with quarters `b` and `e` in `data`."
    function time_bounds(data,b,e)
        q1 = findfirst(x->x==b, data[:,:q])
        q2 = findfirst(x->x==e, data[:,:q])
        return (q1,q2)
    end

    "Applies the log, linear detrending and factor 100 to `v` in `data` between lines `q1` and `q2`."
    function log_lt!(data,q1,q2,v)
        p = size(v,1)
        t = q1:q2
        for j = 1:p
            data[q1:q2,v[j]] .= log.(data[q1:q2,v[j]])
            β = cov(data[q1:q2,v[j]],t)/cov(t)
            α = mean(data[q1:q2,v[j]]) - mean(t)*β
            data[q1:q2,v[j]] .= data[q1:q2,v[j]] .- (α .+ β.*t)
        end
        data[q1:q2,v] .= 100 .* data[q1:q2,v]
    end

    "Demeans `v` between lines `q1` and `q2` in `data`."
    function demean!(data,q1,q2,v)
        p = size(v,1)
        for j = 1:p
            m = mean(data[q1:q2,v[j]])
            data[q1:q2,v[j]] .= data[q1:q2,v[j]] .- m
        end
    end

    "Loads data in `path`, demeans variables `demean` and linearly detrends variables `detrend` between `t1` and `t2`."
    function linear_detrend(path,vars,detrend,demean,t1,t2)
        rawdata = CSV.File(path; delim = ",") |> DataFrame!
        data = non_missing(rawdata,vars)
        q1,q2 = time_bounds(data,t1,t2)
        if isnothing(q1)
            q1 = 1
        end
        if isnothing(q2)
            q2 = size(data, 1)
        end
        log_lt!(data,q1,q2,detrend)
        demean!(data,q1,q2,demean)
        return convert(Array{Float64,2},data[q1:q2,vars])
    end

    "Applies the log, first-difference detrending and factor 100 to `v` in `data` between lines `q1` and `q2`."
    function log_fd!(data,q1,q2,v)
        p = size(v,1)
        t = q1:q2
        for j = 1:p
            data[q1-1:q2,v[j]] .= log.(data[q1-1:q2,v[j]])
            data[q1:q2,v[j]] .= data[q1:q2,v[j]] .- data[q1-1:q2-1,v[j]]
        end
        data[q1:q2,v] .= 100 .* data[q1:q2,v]
    end

    "Loads data in `path`, demeans variables `demean` and applies first-difference detrending `detrend` between `t1` and `t2`."
    function first_difference(path,vars,detrend,t1,t2)
        rawdata = CSV.File(path; delim = ",") |> DataFrame!
        data = non_missing(rawdata,vars)
        q1,q2 = time_bounds(data,t1,t2)
        if q1 == 1
            q1 = 2
            println("Warning: Starting period changed from $(t1) to $(data[q1,:q]) !")
        end
        log_fd!(data,q1,q2,detrend)
        demean!(data,q1,q2,vars)
        return convert(Array{Float64,2},data[q1:q2,vars])
    end

    "Applies Hodrick-Prescott filter with filtering parameter `λ` on the time series `y`."
    function hp(y,λ)

        n = size(y,1)

        d2 = λ*ones(n-2)

        d1 = zeros(n-1)
        d1[1] = -2*λ
        d1[2:n-2] .= -4*λ
        d1[n-1] = -2*λ

        d0 = zeros(n)
        d0[1] = 1+λ
        d0[2] = 1+5*λ
        d0[3:n-2] .= 1+6*λ
        d0[n-1] = 1+5*λ
        d0[n] = 1+λ

        D = diagm(0=>d0, 1=>d1, -1=>d1, 2=>d2, -2=>d2)

        return y - D\y

    end

    "Applies the log, hodrick-prescott filter with parameter `λ` and factor 100 to `v` in `data` between lines `q1` and `q2`"
    function hp_filter!(data,q1,q2,v,λ)
        p = size(v,1)
        t = q1:q2
        for j = 1:p
            data[q1:q2,v[j]] .= log.(data[q1:q2,v[j]])
            data[q1:q2,v[j]] .= hp(data[q1:q2,v[j]],λ)
        end
        data[q1:q2,v] .= 100 .* data[q1:q2,v]
    end

    "Loads data in `path`, demeans variables `demean` and applies Hodrick-Prescott filter to variables `detrend` between `t1` and `t2`."
    function hp_detrend(path,vars,detrend,demean,t1,t2;λ=1600)
        rawdata = CSV.File(path; delim = ",") |> DataFrame!
        data = non_missing(rawdata,vars)
        q1,q2 = time_bounds(data,t1,t2)
        hp_filter!(data,q1,q2,detrend,λ)
        demean!(data,q1,q2,demean)
        return convert(Array{Float64,2},data[q1:q2,vars])
    end

    "Applies Hamilton filter with filtering parameters `h` and `p` on the time series `y`."
    function hamilton(y,h,p)
        T = size(y,1)
        x = ones(T-p-h+1,p+1)
        for j = 0:p-1
            x[:,j+2] .= y[p-j:T-h-j]
        end
        β_hat = (x'*x)\(x'*y[p+h:end])
        return y[p+h:end] - x*β_hat
    end

    "Applies the log, Hamilton filter with parameters `h` and `p` and factor 100 to `v` in `data` between lines `q1` and `q2`"
    function hamilton_filter!(data, q1, q2, v, h, p)
        n = size(v,1)
        t = q1:q2
        for j = 1:n
            data[q1:q2,v[j]] .= 100 .* log.(data[q1:q2,v[j]])
            data[q1+p+h-1:q2,v[j]] .= hamilton(data[q1:q2,v[j]],h,p)
        end
    end

    "Loads data in `path`, demeans variables `demean` and applies Hodrick-Prescott filter to variables `detrend` between `t1` and `t2`."
    function hamilton_detrend(path, vars, detrend, demean, t1, t2 ; h=8, p=4)
        rawdata = CSV.File(path; delim = ",") |> DataFrame!
        data = non_missing(rawdata, vars)
        q1,q2 = time_bounds(data, t1, t2)
        if isnothing(q1)
            q1 = 1
        end
        if isnothing(q2)
            q2 = size(data, 1)
        end
        demean!(data,q1,q2,demean)
        hamilton_filter!(data, q1, q2, detrend, h, p)
        return convert(Array{Float64,2}, data[q1+p+h-1:q2,vars])
    end

    "Applies linear detrending to `detrend` for variables "
    function log_lt_df!(data,b,e,detrend)
        for v in detrend
            t = b[v]:e[v]
            data[t,v] .= log.(data[t,v])
            β = cov(data[b[v]:e[v],v],t)/cov(t)
            α = mean(data[b[v]:e[v],v]) - mean(t)*β
            data[t,v] .= data[t,v] .- (α .+ β.*t)
            data[t,v] .= 100 .* data[t,v]
        end
    end

    function linear_df(path,detrend,demean)
        data = CSV.read(path; delim = ",", copycols=true)
        b,e = non_missing_index_list(data)
        demean_df!(data,demean,b,e)
        log_lt_df!(data,b,e,detrend)
        return (data,b,e)
    end

    function demean_df!(data,v,b,e)
        for j in v
            data[b[j]:e[j],j] .= data[b[j]:e[j],j] .- mean(data[b[j]:e[j],j])
        end
    end

    function fd_df!(data,b,e,detrend)
        for v in detrend
            t = b[v]:e[v]
            data[t,v] .= 100*log.(data[t,v])
            data[b[v]+1:e[v],v] .= data[b[v]+1:e[v],v] - data[b[v]:e[v]-1,v]
            b[v] += 1
        end
    end

    function firstdifference_df(path,detrend,demean)
        data = CSV.read(path; delim = ",", copycols=true)
        b,e = non_missing_index_list(data)
        demean_df!(data,demean,b,e)
        fd_df!(data,b,e,detrend)
        return (data,b,e)
    end

    function hp_df!(data,detrend,b,e,λ)
        for v in detrend
            t = b[v]:e[v]
            data[t,v] .= 100*log.(data[t,v])
            data[t,v] .= hp(data[t,v],λ)
        end
    end

    function hodrickprescott_df(path,detrend,demean;λ=1600)
        data = CSV.read(path; delim = ",", copycols=true)
        b,e = non_missing_index_list(data)
        demean_df!(data,demean,b,e)
        hp_df!(data,detrend,b,e,λ)
        return (data,b,e)
    end


    function hamilton_df!(data,detrend,b,e,h,p)
        for v in detrend
            t = b[v]:e[v]
            data[t,v] .= 100*log.(data[t,v])
            data[b[v]+p+h-1:e[v],v] .= hamilton(data[t,v],h,p)
            b[v] += p+h-1
        end
    end

    function hamilton_detrend_df(path,detrend,demean;h=8,p=4)
        data = CSV.read(path; delim = ",", copycols=true)
        b,e = non_missing_index_list(data)
        demean_df!(data,demean,b,e)
        hamilton_df!(data,detrend,b,e,h,p)
        return (data,b,e)
    end

end
