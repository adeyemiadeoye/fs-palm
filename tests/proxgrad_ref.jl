# Reference proximal-gradient solutions from ProximalAlgorithms.jl.
#
# Driven as a subprocess rather than through juliacall: the bridge is not
# installed in this environment, and a subprocess keeps the reference
# implementation entirely independent of the Python side -- which is the point
# of a cross-check. Problem data comes in as JSON, solutions go out as JSON.
#
# Three problems, chosen to exercise different parts of the linesearch:
#   lasso-easy   well conditioned, the linesearch should barely engage
#   lasso-hard   ill conditioned, forces repeated backtracking
#   boxqp        a different prox (indicator of a box) rather than l1
using ProximalAlgorithms, ProximalOperators, LinearAlgebra, JSON

# The smooth term is supplied with an explicit value_and_gradient rather than
# through ProximalOperators' LeastSquares. With adaptive = true the solver's
# linesearch calls value_and_gradient, and LeastSquares does not provide that
# method in this package pairing. Writing the least-squares gradient out by
# hand also removes any doubt about WHICH function the reference is minimising,
# which is the whole point of an independent check.
struct LS
    A::Matrix{Float64}
    b::Vector{Float64}
end
(f::LS)(x) = (r = f.A * x - f.b; 0.5 * dot(r, r))
function ProximalAlgorithms.value_and_gradient(f::LS, x)
    r = f.A * x - f.b
    return 0.5 * dot(r, r), f.A' * r
end

spec = JSON.parsefile(ARGS[1])
out = Dict{String,Any}()

for (name, p) in spec
    A = Matrix{Float64}(reduce(hcat, p["A"])')
    b = Vector{Float64}(p["b"])
    x0 = Vector{Float64}(p["x0"])
    f = LS(A, b)
    g = p["prox"] == "l1" ? NormL1(Float64(p["lam"])) :
        IndBox(Float64(p["lo"]), Float64(p["hi"]))
    # ForwardBackward with its own backtracking linesearch (adaptive = true),
    # i.e. the same class of method as ours, tolerance well below what we
    # compare at so the reference is not the limiting factor.
    solver = ProximalAlgorithms.ForwardBackward(tol = 1e-12, maxit = 200000,
                                                adaptive = true)
    x, it = solver(x0 = copy(x0), f = f, g = g)
    out[name] = Dict("x" => collect(x), "iters" => it,
                     "obj" => f(x) + g(x))
end

open(ARGS[2], "w") do io
    JSON.print(io, out)
end
println("reference solutions written")
