##
using Random
using Distributions
using Plots

Random.seed!(123)

pd_g = Gumbel()
pd_l = Laplace()
##
x = 0:0.1:10
y = pdf.(pd_g, x)
Plots.plot(x, y)

##