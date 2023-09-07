# fmm 

Fully endogenized finite mixture modeling

$`h(P_i | x_i, \beta_j, p_j)=\sum_{j=1} \pi(z_i)f(P_i|x_i, \beta_j)`$

Generate starting values for $`\gamma, \beta, \pi`$

Initiate iteration counter for the E-step, $`t`$ (initial $`t`$ at 0) 

Use $`\beta^t`$ and $`\pi^t`$ from Step 2 to calculate provisional $`d^t`$ and $`\gamma^t`$ from $`d_{ij}=\frac{e^{\gamma_j z_i}}{1+\sum_{C=1}e^{\gamma_j z_i}}`$ 

Initiate second iteration counter, $`v`$, for the M-step 

Interim estimators of $`d^{t+1}`$ are then used to impute new estimates of $`\beta^{v+1}`$ and $`\pi^{v+1}`$ 

For each prescribed latent class, estimators of $`\beta^{v+1}`$ are imputed, via M-step, as well as $`\pi^{v+1}`$ 

Increase $`v`$ counter by 1, and repeat M-step until: $`f(\beta^{v+1}y, x, \pi, d) - f(\beta^vy, x, \pi, d) < \alpha`$ prescribed constant; if yes, then $`\beta^{t+1}=\beta^{v+1}`$ 

Increase $`t`$ counter and continue from Step 3 until: $`f(\beta^{t+1}, \pi^{t+1}, d | y) - f(\beta^t, \pi^t, d | y) < \alpha`$ prescribed constant 


