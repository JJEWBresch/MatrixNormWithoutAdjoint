This repository allows to:
  - calculate the operator norm of an operator A (real or complex) which allows the evaluation of the map/oracle: v -> A v.
  - detect orthogonal matrices w.r.t. some scaling parameter, i.e. A^* A = sigma * I_d form some sigma >= 0.

  - calculate the operator norm of an operator W = A - V (real or complex) for which only the evaluation of the map/oracle: v -> A v and u -> V^* u are available.
  - detect the case, where A = V.

The methods guarantee almost sure convergence to the maximal singular value, i.e. the operator norm.
The method is a randomized ascend method for the Rayleigh quotient (or a generalization thereof, respectively) where the search direction is randomly sampled perpendicular to the current point and the stepsize is chosen optimally to maximize the Rayleigh quotient. 
