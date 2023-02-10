# Fragility-index

In general, the code is very straightforward. Notice that in the section 5.3 of minimizing the FI and bAUC, we use the commercial optimizer Gurobi, which is not open source. For people who cannot get Gurobi, you can replace Gurobi by any other open source optimizer that can solve our proposed problem.

Except Gurobi, all other requirements for this project is trivial. In practice, the versions of packages is not restrictive. We provide an example requirement in *requirements.txt*. 