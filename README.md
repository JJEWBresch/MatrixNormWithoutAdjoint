This repository allows to:
  - calculate the operator norm of an operator A (reel or complex) which allows the avaluation of the map/oracel: v -> A v.
  - detect orthogonal matrices w.r.t. some scaling parameter, i.e. A^* A = sigma * I_d form some sigma >= 0.

  - calculate the operator norm of an operator W = A - V (reel or complex) which just allows the avaluation of the map/oracel: v -> A v and u -> V^* u.

In contrast to other programs solving this task, we are alloved to gurantee the convergence to the maximal singular value, i.e. the operator norm.
Furthermore, the PROBABILISTICALLY results come from the fact that the ascend direction is, in a ruffly speacing way, randomly sampled and the stepsize is optimal chosen, 
whereas the the step size is the parameter which becomes rendomly smapled in common programs for this task.

  - ALL RESULTS are PROBABILISTICALLY
