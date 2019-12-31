function non_missing_index(d)
    n,p = size(d)
    lines = ones(Int8,2,p)
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

function non_missing(d,v)
    b,e = non_missing_index(d[:,v])
    return d[b:e,:]
end

function time_bounds(data,b,e)
    q1 = findfirst(x->x==b,data[:,:q])
    q2 = findfirst(x->x==e,data[:,:q])
    return (q1,q2)
end

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

function demean!(data,q1,q2,v)
    p = size(v,1)
    for j = 1:p
        m = mean(data[q1:q2,v[j]])
        data[q1:q2,v[j]] .= data[q1:q2,v[j]] .- m
    end
end

function linear_detrend(path,vars,detrend,demean,t1,t2)
    rawdata = CSV.File(path; delim = ",") |> DataFrame!
    data = non_missing(rawdata,vars)
    q1,q2 = time_bounds(data,t1,t2)
    log_lt!(data,q1,q2,detrend)
    demean!(data,q1,q2,demean)
    return convert(Array{Float64,2},data[q1:q2,vars])
end
