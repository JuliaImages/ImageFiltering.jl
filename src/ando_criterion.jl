using Cubature, OffsetArrays, JuMP

## 2d

ψ(u,v,m,n) = u.*cos(π*n*u).*sin(π*m*v) - v.*sin(π*m.*u)*cos(π*n.*v)
Ruv(u,v,k,l,m,n) = (u==v==0 ? zero(u*v) : ψ(u,v,k,l).*ψ(u,v,m,n)./(u.^2+v.^2))
Ruv(x,k,l,m,n) = Ruv(x[1],x[2],k,l,m,n)
R(k,l,m,n) = hcubature(x->Ruv(x,k,l,m,n), (-0.5,-0.5), (0.5, 0.5))[1]

# 3x3
R3 = OffsetArray([R(k,l,m,n) for k=-1:1, l=-1:1, m=-1:1, n=-1:1], -1:1, -1:1, -1:1, -1:1)

ando3start = OffsetArray([-1 -2 -1; 0 0 0; 1 2 1]/16, -1:1, -1:1)

mod = Model()
@variable(mod, a[i=-1:1,j=-1:1], start=ando3start[i,j])
@constraint(mod, sum{i*a[i,j], i=-1:1, j=-1:1} == 0.5)
@objective(mod, Min, sum{a[i,j]*a[k,l]*R3[i,j,k,l], i=-1:1, j=-1:1, k=-1:1, l=-1:1})

status = solve(mod)
println(getvalue(a))
