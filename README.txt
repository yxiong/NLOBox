================================================================
NLOBox --- A matlab toolbox for nonlinear optimization.
================================================================

Author: Ying Xiong.
Created: Jan 20, 2014 (original name NLLSBox).
Release: May 10, 2014 (v0.3).

================================================================
Quick start.
================================================================
>> addpath('Utils');
>> NLOBoxTest;
>> demoNLOBox;

The main function of the package is 'NonlinearMinimization.m', which has a
similar interface as Matlab's 'fminunc'. But it can also handle some simple
constraints, say l <= x <= u, currently in a naive way.

For nonlinear least squares problem, we have 'NonlinearLeastSquares.m', which
currently uses Levenberg-Marquardt algorithm and has similar interface as
Matlab's 'lsqnonlin'.

================================================================
Notation and convention.
================================================================

For general nonlinear optimization problem, the objective function has the form
  [f, g] = fcn(x),
where 'x' is a vector of dimension N, 'f' is a scalar and 'g' is the gradient of
the function and therefore also a vector of dimension N.

For nonlinear least squares problem, The cost function we will minimize is
  F(x) = \sum_{i=1}^M f_i(x)^2
where 'x' is a vector of dimension N, 'f' is a vector function of dimension M,
and 'F' is a scalar. We also define 'J' as the Jacobian matrix of function 'f',
which is a matrix of dimension MxN. The objective is given in form of
  [f, J] = fcn(x).

All vectors are column vectors unless otherwise stated.

Abbreviations and acronyms:
  NLM:  Nonlinear minimization.
  NLLS: Nonlinear least squares.

================================================================
Features.
================================================================
* Same interface but better than 'fminunc' / 'lsqnonlin'.
* BFGS algorithm for general nonlinear minimization.
* Levenberg-Marquardt algorithm for nonlinear least squares.
* Support bounded constraints.
* Support using finite difference to compute gradient and Jacobian matrix.

See 'NonlinearOptimization.pdf' for a more detailed documentation.
