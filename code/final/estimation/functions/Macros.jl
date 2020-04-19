module Macros

export @def, @save, @load, @index, @load_cat, @load_vec, @load_hat

"""
	@def name definition

Defines the macro `name`, which writes the code in `definition`
"""
macro def(name, definition)
  return quote
      macro $(esc(name))()
          esc($(Expr(:quote, definition)))
      end
  end
end

"""
	@save v1 ...

Saves local variables `v1`, ... and their associated values in a dictionary
"""
macro save(vars...)
	expr = Expr(:call, :(Dict{Symbol, Float64}))
	for c in vars
		push!(expr.args, :($(QuoteNode(c)) => $(esc(c))))
	end
	return expr
end

"""
	@load d v1 ...

Loads variables `v1`, ... from the dictionary `d`
"""
macro load(d, vars...)
	expr = Expr(:block)
	for v in vars
		push!(expr.args, :($v = $d[$(QuoteNode(v))]))
	end
	return esc(expr)
end

"""
	@index v1 v2 ...

Assign values 1, 2 ... to `v1`, `v2`, ...
"""
macro index(vars...)
	expr = Expr(:block)
	for (k,v) in enumerate(vars)
		push!(expr.args, :($v = $k))
	end
	return esc(expr)
end

"""
	@load_cat name d v1 v2 ...

Load variables `v1`, `v2`, ... from dictionary `d` and rename them as suffix `v1_name`, `v2_name`, ...
"""
macro load_cat(name, d, vars...)
	expr = Expr(:block)
	for v in vars
		push!(expr.args, :($(Symbol("$(v)_$(name)")) = $d[$(QuoteNode(v))]))
	end
	return esc(expr)
end

"""
	@load_vec vec v1 ...

Assign values `vec[1]`, `vec[2]`, ... to  variables `v1`, `v2`, ...
"""
macro load_vec(vec, vars...)
	expr = Expr(:block)
	for (k,v) in enumerate(vars)
		push!(expr.args, :($v = $vec[$k]))
	end
	return esc(expr)
end

end
