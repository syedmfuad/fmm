# fmm 

Fully endogenized finite mixture modeling

$`h(P_i | x_i, \beta_j, p_j)=\sum_{j=1} \pi(z_i)f(P_i|x_i, \beta_j)`$

Generate starting values for $`\gamma, \beta, \pi`$

Initiate iteration counter for the E-step, $`t`$ (initial $`t`$ at 0) 

Use $`\beta^t`$ and $`\pi^t`$ from Step 2 to calculate provisional $`d^t`$ and $`\gamma^t`$ from $`d_{ij}=\frac{e^{\gamma_j z_i}}{1+\sum_{C=1}e^{\gamma_j z_i}}`$



