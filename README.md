# fmm 

R and Julia implementation of the fully endogenized finite mixture modeling. This model employs a finite mixture model to sort households into endogenously determined latent submarkets. The finite mixture model to predict home prices is: 

$`h(P_i | x_i, \beta_j, p_j)=\sum_{j=1} \pi(z_i)f(P_i|x_i, \beta_j)`$ 

The mixing model $`\pi(z_i)`$, is used to assign each observation a percentage chance of belonging to each latent submarket and $`f(.)`$ is a submarket specific conditional hedonic regression. The home price is therefore a weighted average of predicted values across submarkets weighted by the probability of being located in the submarket. 

We also define $`d_i = (d_{i1}, d_{i2}, ..., d_{im})`$ to be binary variables that indicate the inclusion of household $`i`$ into each latent group. These are incorporated into the likelihood function based on a logistic function which are conditional on factors that do not directly influence the price of the house. 

Since the submarket identification ($`d`$) is not directly observable, an expectation maximization (EM) algorithm is used to estimate the likelihood of class identification: $`d_{ij}=\frac{\pi_j f_j (P_i | x_i, \beta_j)}{\sum_{j=1} \pi_j f_j (P_i | x_i, \beta_j)}`$ 

The Expectation step – the E step – involves imputation of the expected value of $`d_i`$ given the mixing covariates, interim estimates of $`\gamma, \beta, \pi`$. The Maximization step – the M step – involves using estimates of $`d_i`$ from the E step to update the component fractions of $`\pi_j`$ and $`\beta`$. The EM algorithm can be summarized as: 

1. Generate starting values for $`\gamma, \beta, \pi`$

2. Initiate iteration counter for the E-step, $`t`$ (initial $`t`$ at 0) 

3. Use $`\beta^t`$ and $`\pi^t`$ from Step 2 to calculate provisional $`d^t`$ from $`d_{ij}=\frac{e^{\gamma_j z_i}}{1+\sum_{C=1}e^{\gamma_j z_i}}`$ 

4. Initiate second iteration counter, $`v`$, for the M-step 

5. Interim estimators of $`d^{t+1}`$ are then used to impute new estimates of $`\beta^{v+1}`$ and $`\pi^{v+1}`$ with $`d_{ij}=\frac{\pi_j f_j (P_i | x_i, \beta_j)}{\sum_{j=1} \pi_j f_j (P_i | x_i, \beta_j)}`$

6. For each prescribed latent class, estimators of $`\beta^{v+1}`$ are imputed, via M-step, as well as $`\pi^{v+1}`$ 

7. Increase $`v`$ counter by 1, and repeat M-step until: $`f(\beta^{v+1}y, x, \pi, d) - f(\beta^vy, x, \pi, d) < \alpha`$ prescribed constant; if yes, then $`\beta^{t+1}=\beta^{v+1}`$ 

8. Increase $`t`$ counter and continue from Step 3 until: $`f(\beta^{t+1}, \pi^{t+1}, d | y) - f(\beta^t, \pi^t, d | y) < \alpha`$ prescribed constant 

$`d_{ij}`$ is estimated simultaneously with the estimation of the hedonic regression parameters, which are conditional on class identification: 

$`LogL = \sum_{i=1} \sum_{j=1} d_{ij} log[f_j (P_i | x_i, \beta_j)] + d_{ij} log[\pi_j]`$ 

This process is repeated until there is no change in the likelihood function above. 

