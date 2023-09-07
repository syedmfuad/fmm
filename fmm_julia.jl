using Pandas
using DataFrames
using GLM
using StatsBase
using Rmath
using Optim
using Distributions

#read in data

df= read_csv("HHdata.csv")

Y = Array(df["Price"])

#house attributes

varsIndp = df[["SquareFoot","Lot","HouseAge","Garage","ExpBird"]]
varsIndp = Array(varsIndp)
X=[ones(size(varsIndp,1)) varsIndp]

#demographic variables

Z = [ones(size(varsIndp,1)) Array(df["Educ"]) Array(df["Inc"])/10000 Array(df["Age"]) Array(df["HHSize"])]

#aggregate linear model

model = lm(X,Y)

#getting residuals and starting values for mixing algorithm

res = StatsBase.residuals(model)

coeffs=coeftable(model).cols[1]

ncolX = size(X,2)

beta_start = [coeffs; coeffs]

ncolZ = size(Z,2)

gamma_start = [0.01 for i=1:ncolZ]

sigma_start = [sqrt(mean(res.^2)) sqrt(mean(res.^2))]

#

val_start = vcat(beta_start,gamma_start,sigma_start')

vals = val_start
types = 2
Iter_conv = 0.0001

j = types
niv = ncolX
gvs = ncolZ
n = size(X,1)
conv_cg = 5000
conv_cb = 5000
#Define some functions for mixing algorithm
function FnOne(par,x,y)
    return map((y,multi)->pdf(Normal(multi,par[1]),y),y,x*par[2:end])
end
#FnTwo max prob densities over type probabilities
function FnTwo(par,d,x,y)
    f = zeros(n,j)
    b = par[1:(niv*j)]
    s = par[(niv*j+1):((niv+1)*j)]
    for h=1:j
       f[:,h] = FnOne(vcat(s[h],b[((h-1)*niv+1):(h*niv)]),X,Y)
    end
    return sum(d.*map(log,f))*-1.0
end
#FnThree logit for gamma estimates
function FnThree(g,z)
    return map(exp,z*g)
end
#FnFour max gamma estimates, type probabilities
function FnFour(par,d,z,y)
    V = zeros(n,j)
    V[:,1]=ones(size(V[:,1]))
    for m=1:j-1
        V[:,m+1]= FnThree(par[((m-1)*gvs+1):(m*gvs)],z)
    end
    V2 = (sum(V,dims=(2)))
    Pi = V ./ hcat(V2,V2)
    return sum(d.*map(log,Pi))*-1.0
end

function FMM(par,X,z,y)
    V = zeros(n,j)
    f = copy(V)
    d = copy(V)
    b = copy(par[1:(j*niv)])
    g = copy(par[(j*niv+1):((j*(niv+gvs)-gvs))])
    s = copy(par[(j*(niv+gvs)-gvs)+1:end])
    b = reshape(b,niv,j)
    iter = 0
    
    conv_cg=5000.0
    conv_cb=5000.0
    parms =0
    dvector=0
    
    while (abs(conv_cg)+abs(conv_cb) > Iter_conv)
        #store parameter estimates of preceding iteration of mix through loop

        beta_old = copy(b)
        gamma_old = copy(g)
        iter = iter+1
        for h=1:j
            f[:,h]=FnOne([s[h] b[:,h]'],X,Y)
        end
        
        for h=1:j-1
            V[:,1]=0*V[:,1]
            V[:,h+1] = z*g[((h-1)*gvs+1):(h*gvs)]
        end
        V2 = (sum(map(exp,V[:,(1:j)]),dims=(2)))+ones(size(sum(V[:,(1:j)],dims=(2))))
        
        #estimate Pi (P) and individual probabilities of belonging to a certain type (d):

        P=map(exp,V)./hcat(V2,V2)
        for i =1:n
            multi = P[i,:].*f[i,:]
            summation = sum(P[i,:].*f[i,:])
            d[i,:] = [multi[j]/summation for j=1:size(d[i,:],1)]
        end

        #use individual probs (d) to estimate beta (b), gamma (g)      
        b1 = reshape(b,niv*j,1)
        par1 = vcat(b1,s)
        beta_opt = optimize(par1->FnTwo(par1,d,X,Y),par1,Optim.Options(iterations = 100000))
     
        b = reshape(beta_opt.minimizer[1:j*niv],niv,j)
        s = beta_opt.minimizer[j*niv+1:(j*(niv+1))]
           
        gamma_opt = optimize(g->FnFour(g,d,Z,Y),g,Optim.Options(iterations = 100000))
       
        g = gamma_opt.minimizer
        
        
        #convergence check 
        conv_cg = sum(abs.(g-gold))
        conv_cb = sum(abs.(b-bold))
        
        
        #recollecting parameter estimates to impute log likelihood

        par2 = reshape(b,(niv*j),1)
        par2 = vcat(par2,s)
        LL = FnTwo(par2,d,X,Y) + FnFour(g,d,Z,Y)
        print("\n\nFnFour: ",FnFour(g,d,Z,Y),"\n")
         print(b)
        print(g)
        print(iter)
        print(conv_cg)
        print(conv_cb)
        
        bvector = reshape(b,j*niv,1)
        vals_fin = vcat(bvector,g,s)
        dvector = d
    end
    return vals_fin
end

results = FMM(val_start,X,Z,Y)

print("\nhere\n")

#final updating and repeating computation in FMM to extract standard errors
V = zeros(n,j)
f = V
d = V
b = results[1:(j*niv)]
g = results[(j*niv+1):((j*(niv+gvs)-gvs))]
s = results[(j*(niv+gvs)-gvs)+1:end]
b = reshape(b,niv,j)

for h=1:j
    f[:,h]=FnOne(vcat(s[h],b[:,h]),X,Y)
end

for h=1:j-1
    V[:,1]=zeros(size(V,1),1)
    print("\nhere loop ")
    V[:,h+1] = Z*g[((h-1)*gvs+1):(h*gvs)]
end

V2 = (sum(map(exp,V[:,(1:j)]),dims=(2)))+ones(size(sum(V[:,(1:j)],dims=(2))))
P = V ./ hcat(V2,V2)

for i=1:n
    multi = P[i,:].*f[i,:]
    summation = sum(P[i,:].*f[i,:])
    d[i,:] = [multi[j]/summation for j=1:size(d[i,:],1)]
end

b1 = reshape(b,niv*j,1)
par3 = vcat(b1,s)
