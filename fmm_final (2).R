
path <- "C:/Users/Michael/Desktop/Farmer/"

data <- read.table(paste0(path,"HHdata.csv"), header=TRUE, sep=",")

Y <- I(data$Price)/100000

#house attributes

X <- cbind(1,I(data$SquareFoot)/1000, I(data$Lot)/1000, data$HouseAge, data$Garage, data$ExpBird)

#demographic variables/mixing variables

Z <- cbind(1,data$Educ, I(data$Inc)/10000, data$Age, data$HHSize)

ols_agg <- lm(Y~X-1);

#starting values for the hedonic estimates/betas for each type i.e. for mixing algorithm

beta_start <- matrix(ols_agg$coef,(2*ncol(X)),1);  

#starting values for the gamma estimates for the demographic variables

gamma_start <- matrix(0.01,(1*ncol(Z)),1);

#starting values for sigma
sigma_start <- matrix(sqrt(mean(ols_agg$residuals^2)),2,1)     

#collecting initializing values

val_start <- c(beta_start,gamma_start,sigma_start);


vals <- val_start;
types <- 2;

#convergence criteria comparing new and old estimates:

Iter_conv <- 0.0001;
j <- types; 

#number of independent variables or beta estimates we need to keep track of - so to use when indexing

niv <- ncol(X); 

#number of demographic variables to use when indexing

gvs <- ncol(Z); 

#row dim of aggregate

n <- nrow(X);
conv_cg = 5000; 
conv_cb = 5000; 

#FnOne prob density  of observing prices given mean of cross product of house attributes and current 
#iteration of hedonic estimates and sigma

FnOne <- function(par,x,y) 
{
      dnorm(y, mean=x%*%par[-1], sd = par[1], log=FALSE)   
}

#FnTwo max prob densities over type probabilities

FnTwo <- function(par,d,x,y)   
{
      pdy <- matrix(0,n,j) 
      b <- par[1:(niv*j)] 
      s <- par[(niv*j+1):((niv+1)*j)]
      for (i in 1:j)
       {	
        pdy[,i] <- FnOne(c(s[i],b[((i-1)*niv+1):(i*niv)]),X,Y)        
       }
      sum(d*log(pdy))
}

#FnThree logit for gamma estimates 

FnThree <- function(g,z)  
{ 
      L <- exp(z%*%g)  	
}

#FnFour max gamma estimates, type probabilities

FnFour <- function(par,d,z,y)   
{
	L <- matrix(0,n,j) 
      L[,1] <- 1
	for (m in 1:(j-1))   
         { 
          L[,(m+1)] <- FnThree(par[((m-1)*gvs+1):(m*gvs)],z)     
         }
	Pi <- L / apply(L,1,sum) 
      sum(apply(d*log(Pi),1,sum))  
}

#mixing algorithm

FMM <- function(par,X,Z,y) 
{
b <- par[1:(j*niv)]; 
g <- par[(j*niv+1):((j*(niv+gvs)-gvs))]; 
s <- par[-(1:(j*(niv+gvs)-gvs))];
L <- matrix(0,n,j); 
f <- L; 
d <- L;
b <- matrix(b,niv,j); 
iter <- 0

while (abs(conv_cg) + abs(conv_cb) > Iter_conv)   {   
	
      #store parameter estimates of preceding iteration of mix through loop
      beta_old <- b; 
      gamma_old <- g; 
      
      #counter for while loop
      iter <- iter+1     
	
      for (i in 1:j) 
               { 
                f[,i] <- FnOne(c(s[i],b[,i]),X,Y)		
                }
	for (i in 1:(j-1)) 
                { 
                 L[,1] <- 0
                 L[,(i+1)] <- Z%*%g[((i-1)*gvs+1):(i*gvs)] 
                }

	#estimate Pi (P) and individual probabilities of belonging to a certain type (d):
      
      P <- exp(L)/(1+apply(exp(L[,(1:j)]),1,sum))    
	
      for (i in 1:n) 
              {	
              d[i,] <- P[i,]*f[i,]/sum(P[i,]*f[i,]) 
              }
      
      #use individual probs (d) to estimate beta (b), gamma (g)
	
      b1 <- matrix(b,(niv*j),1); par1 <- c(b1,s);
	beta_m <- optim(par1,FnTwo,d=d,x=X,y=Y,control=list(fnscale=-1,maxit=100000))
	b <- matrix(beta_m$par[1:(j*niv)],niv,j) 
      
      s <- beta_m$par[(j*niv+1):(j*(niv+1))]
	
      gam_m <- optim(g,FnFour,d,z=Z,Y,control=list(fnscale=-1,maxit=100000))
	g <- gam_m$par
	
      #setting up convergence check

      conv_cg <- sum(abs(g-gamma_old)) 
      conv_cb <- sum(abs(b-beta_old))  
	
      #collecting parameter estimates to use to impute LL
      
      par2 <- matrix(b,(niv*j),1)
      par2 <- c(par2,s)
	LL <- FnTwo(par2,d=d,x=X,y=Y) + FnFour(g,d=d,z=Z,y=Y);

      #storing 
	
      bvector <- matrix(b,j*niv,1)
	vals_fin <- c(bvector,g,s)	
	dvector <- d
}
#collecting parameters for output
	
out_pars <- list("vals_fin" = vals_fin, "i_type" = d)
print(b)
print(g)
print(iter)

#return list of estimates - index for subsetting in final updating 
return(out_pars)
}

#calling:

mix <- FMM(val_start,X=X,Z=Z,y=Y)

#final updating:

d <- mix$i_type

b <- mix$vals_fin[1:(j*niv)]; 

g <- mix$vals_fin[(j*niv+1):((j*(niv+gvs)-gvs))]; 

s <- mix$vals_fin[-(1:(j*(niv+gvs)-gvs))];

b <- matrix(b,niv,j);

b1 <- matrix(b,(niv*j),1); 
par3 <- c(b1,s);

#standard errors

beta_opt <- optim(par3,FnTwo,d=d,x=X,y=Y,control=list(fnscale=-1,maxit=10000),hessian=TRUE, method = c("Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN", "Brent"))
b <- matrix(beta_opt$par[1:(j*niv)],niv,j); 
bse1 <- sqrt(-diag(solve(beta_opt$hessian[1:niv,1:niv])))
bse2 <- sqrt(-diag(solve(beta_opt$hessian[(niv+1):(2*niv),(niv+1):(2*niv)])))

s <- beta_opt$par[(j*niv+1):(j*(niv+1))]

gamma_opt <- optim(g,FnFour,d=d,z=Z,y=Y,control=list(fnscale=-1,maxit=100000),hessian=TRUE)
g <- gamma_opt$par
gse1 <- sqrt(-diag(solve(gamma_opt$hessian[1:gvs,1:gvs])))

par2 <- matrix(b,(niv*j),1);  
par2 <- c(par2,s)

LL <- FnTwo(par2,d=d,x=X,y=Y) + FnFour(g,d=d,z=Z,y=Y);

Ds=d;
beta=b;
bse=cbind(bse1,bse2);
gamma=cbind(g[1:gvs],g[(gvs+1):(2*gvs)]);
gse=cbind(gse1);

#check which is which

if(sum(d[,1]>d[,2]) > sum(d[,2]>d[,1])){
col_nombre <- c("Type 2","Type 1")
}else {
col_nombre <- c("Type 1","Type 2")}

row_nombre <- c("Intercept","Square Foot","Lot Size", "House Age", "Garage", "Bird")

write.table(b, file= paste0(path,"Beta.csv"),quote = FALSE, row.names= row_nombre, col.names=col_nombre, sep=",")
write.table(bse, file= paste0(path,"Bse.csv"),quote = FALSE, row.names= row_nombre, col.names=col_nombre, sep=",")
write.table(LL, file= paste0(path,"LL.csv"),quote = FALSE, row.names= TRUE, col.names=TRUE, sep=",")
write.table(s, file= paste0(path,"S.csv"),quote = FALSE, row.names= TRUE, col.names=TRUE, sep=",")
write.table(gse1, file= paste0(path,"Gse.csv"),quote = FALSE, row.names= TRUE, col.names=TRUE, sep=",")
write.table(gamma, file= paste0(path,"Gamma.csv"),quote = FALSE, row.names= TRUE, col.names=TRUE, sep=",")
write.table(d, file= paste0(path,"Dhats.csv"),quote = FALSE, row.names= TRUE, col.names=col_nombre, sep=",")



