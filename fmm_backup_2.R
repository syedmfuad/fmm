rm(list=ls())

setwd("C:/Users/syedm/Desktop/Distance paper")

#Test on Lubbock data 

data <- read.table("HHdata.csv", header=TRUE, sep=",")

Y <- I(data$Price)/100000

#house attributes

X <- cbind(1,I(data$SquareFoot)/1000, I(data$Lot)/1000, data$HouseAge, data$Garage, data$ExpBird)

#demographic variables/mixing variables

Z <- cbind(1,data$Educ, I(data$Inc)/10000, data$Age, data$HHSize)

ols_agg <- lm(Y~X-1);
summary(ols_agg)

#starting values for the hedonic estimates/betas for each type i.e. for mixing algorithm

types <- 2;

beta_start <- matrix(ols_agg$coef,(types*ncol(X)),1);  

#starting values for the gamma estimates for the demographic variables

#gamma_start <- matrix(0.01,(1*ncol(Z)),1);
gamma_start <- matrix(abs(rnorm(types*ncol(Z),mean=0.01,sd=0.01)),(types*ncol(Z)),1);

#starting values for sigma
sigma_start <- matrix(sqrt(mean(ols_agg$residuals^2)),types,1)     

#collecting initializing values

val_start <- c(beta_start,gamma_start,sigma_start);

vals <- val_start;

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

par <- val_start

#FnOne prob density  of observing prices given mean of cross product of house attributes and current iteration of hedonic estimates and sigma (sd) 
#Mean is calculated as the product of house attributes (x) and the hedonic price estimates (par[-1], excluding the first element which is the sd).

FnOne <- function(par,x,y) 
{
  dnorm(y, mean=x%*%par[-1], sd = par[1], log=FALSE)   
}

#FnTwo max prob densities over type probabilities
#Function loops over different types (j) and computes the probability density for each type using FnOne 
#Then sums the logarithms of these densities, weighted by the type probabilities (d), which is a standard approach in maximum likelihood estimation 
#Function is used to update the hedonic price estimates (beta) during the optimization process in FMM 

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
#Computes the logistic function of the demographic variable estimates 
#(The exponential (exp) of the product of demographic variables (z) and their coefficients (g))
#This is a part of the logistic regression model to predict the probability of a house belonging to a particular type based on demographic variables 

FnThree <- function(g,z)  
{ 
  L <- exp(z%*%g)  	
}

#FnFour max gamma estimates, type probabilities
#Maximizes the logistic function over different types, essentially refining the estimates of demographic variable effects (gamma) 
#Constructs a matrix L to store the logistic function values for each type 
#Then calculates the probabilities (Pi) of each house belonging to each type based on these values 
#Returns the sum of the logarithms of these probabilities, which is used in the optimization process to update the demographic variable estimates 

FnFour <- function(par, d, z, y, n, j, gvs) {
  n <- nrow(X)
  #j <- types
  gvs <- ncol(Z)
  L <- matrix(0, n, j) 
  L[,1] <- 1
  
  # Set columns to 1 based on the value of types
  if (j >= 3) {
    L[,3] <- 1
  }
  if (j >= 4) {
    L[,4] <- 1
  }
  
  # Continue with the original functionality
  minus <- j - 1
  for (m in 1:(j - minus)) {
    L[, (m + 1)] <- FnThree(par[((m - 1) * gvs + 1):(m * gvs)], z)     
  }
  
  Pi <- L / apply(L, 1, sum) 
  sum(apply(d * log(Pi), 1, sum))  
}

#mixing algorithm

FMM <- function(par,X,Z,y) {
  
  #separating betas, gamma and sigma from par
  b <- par[1:(j*niv)]; 
  
  if (types == 2) {
    #g <- par[(j * niv + 1):(j * (niv + gvs) - gvs)]
    g <- par[(j*niv+1):((j*(niv+gvs)))]
    s <- par[-(1:(length(par) - types))]
  } else {
    g <- par[(j*niv+1):((j*(niv+gvs)))]
    s <- par[-(1:(length(par) - types))]
  }
  
  #empty matrix to store probs 
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
    
    
    #Computes the probability density of observing house prices given the mean of the cross-product of 
    #house attributes and current hedonic estimates, along with a standard deviation parameter 
    
    for (i in 1:j) 
    { 
      f[,i] <- FnOne(c(s[i],b[,i]),X,Y)		
    }
    
    
    #updating a column of L based on the demographic variables (Z) and their respective coefficients (g)
    #this computation is essentially a part of a logistic regression model where Z %*% g gives the linear predictor for the logistic model 
    
    minus <- types - 1
    for (i in 1:(j - minus)) {
      L[,1] <- 0
      if (types > 2) {
        L[, (i + 1):(types - 1)] <- 0  # This sets the range of columns to 0 based on types
      }
      L[,(i + 1)] <- Z %*% g[((i - 1) * gvs + 1):(i * gvs)]
    }
    
    
    #estimate Pi (P) and individual probabilities of belonging to a certain type (d):
    
    P <- exp(L)/(1+apply(exp(L[,(1:j)]),1,sum))    
    
    for (i in 1:n) 
    {	
      d[i,] <- P[i,]*f[i,]/sum(P[i,]*f[i,]) 
    }
    
    #use individual probs (d) to estimate beta (b), gamma (g)
    #maximizes probability densities 
    #function loops over different types (j) and computes the probability density for each type using FnOne 
    #then sums the logarithms of these densities, weighted by the type probabilities (d), which is a standard approach in maximum likelihood estimation 
    #function is used to update the hedonic price estimates (beta) during the optimization process in FMM 
    
    b1 <- matrix(b,(niv*j),1); par1 <- c(b1,s);
    beta_m <- optim(par1,FnTwo,d=d,x=X,y=Y,control=list(fnscale=-1,maxit=100000))
    b <- matrix(beta_m$par[1:(j*niv)],niv,j) 
    s <- beta_m$par[(j*niv+1):(j*(niv+1))]
    
    
    #maximizes the logistic function (FnThree) over different types essentially refining the estimates of demographic variable effects (gamma) 
    types <- j
    gam_m <- optim(g,FnFour,j=types,d,z=Z,Y,control=list(fnscale=-1,maxit=100000))
    g <- gam_m$par
    
    #setting up convergence check
    
    conv_cg <- sum(abs(g-gamma_old)) 
    conv_cb <- sum(abs(b-beta_old))  
    
    #collecting parameter estimates to use to impute LL
    
    par2 <- matrix(b,(niv*j),1)
    par2 <- c(par2,s)
    
    #types <- j
    LL <- FnTwo(par2,d=d,x=X,y=Y) + FnFour(g,d=d,z=Z,y=Y, j=types);
    
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

end.time <- Sys.time()

start.time-end.time

#send_message(message_body=paste("Operation complete", Sys.time(), start.time-end.time))

#final updating:

d <- mix$i_type

b <- mix$vals_fin[1:(j*niv)]; #probabilities 

#g <- mix$vals_fin[(j*niv+1):((j*(niv+gvs)-gvs))];
g <- mix$vals_fin[(j*niv+1):(length(par)-types)];

#s <- mix$vals_fin[-(1:(j*(niv+gvs)-gvs))];
s <- mix$vals_fin[-(1:(length(par)-types))];

b <- matrix(b,niv,j); #betas

b1 <- matrix(b,(niv*j),1); 
par3 <- c(b1,s);

#standard errors #needs works

beta_opt <- optim(par3,FnTwo,d=d,x=X,y=Y,control=list(fnscale=-1,maxit=10000),hessian=TRUE)
b <- matrix(beta_opt$par[1:(j*niv)],niv,j); 
bse1 <- sqrt(-diag(solve(beta_opt$hessian[1:niv,1:niv])))
bse2 <- sqrt(-diag(solve(beta_opt$hessian[(niv+1):(2*niv),(niv+1):(2*niv)])))
bse3 <- sqrt(-diag(solve(beta_opt$hessian[(niv*2+1):(3*niv),(niv*2+1):(3*niv)])))

s <- beta_opt$par[(j*niv+1):(j*(niv+1))] 







#FnThree logit for gamma estimates

FnThree <- function(g,z)  
{ 
  L <- exp(z%*%g)  	
}

#FnFour max gamma estimates, type probabilities 


FnFour <- function(par, d, z, y, n, j, gvs) {
  n <- nrow(X)
  #j <- types
  gvs <- ncol(Z)
  L <- matrix(0, n, j) 
  L[,1] <- 1
  
  # Set columns to 1 based on the value of types
  if (j >= 3) {
    L[,3] <- 1
  }
  if (j >= 4) {
    L[,4] <- 1
  }
  
  # Continue with the original functionality
  minus <- j - 1
  for (m in 1:(j - minus)) {
    L[, (m + 1)] <- FnThree(par[((m - 1) * gvs + 1):(m * gvs)], z)     
  }
  
  Pi <- L / apply(L, 1, sum) 
  sum(apply(d * log(Pi), 1, sum))  
}




types <- j
gamma_opt <- optim(par=g[1:gvs],FnFour,j=types,d=d,z=Z,y=Y,control=list(fnscale=-1,maxit=100000),hessian=TRUE)
g1 <- gamma_opt$par
gse1 <- sqrt(-diag(solve(gamma_opt$hessian[1:gvs,1:gvs])))

#g <- mix$vals_fin[(j*niv+1):((j*(niv+gvs)))];
gamma_opt <- optim(g[(gvs+1):(gvs*2)],FnFour,j=types,d=d,z=Z,y=Y,control=list(fnscale=-1,maxit=100000),hessian=TRUE)
g2 <- gamma_opt$par
gse2 <- sqrt(-diag(solve(gamma_opt$hessian[1:gvs,1:gvs])))

#g <- mix$vals_fin[(j*niv+1):((j*(niv+gvs)))];
gamma_opt <- optim(g[(gvs*2+1):(3*gvs)],FnFour,j=types,d=d,z=Z,y=Y,control=list(fnscale=-1,maxit=100000),hessian=TRUE)
g3 <- gamma_opt$par
gse3 <- sqrt(-diag(solve(gamma_opt$hessian[1:gvs,1:gvs])))


