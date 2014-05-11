function [alpha, x, f, g] = NLM_LineSearch(fcn, x0, h, f0, g0)

% [alpha, x, f, g] = NLM_LineSearch(fcn, x0, h, f0, g0)
%
% Perform line search on function
%   phi(alpha) = fcn(x0 + alpha * h).
%
% Current implementation uses bracketing algorithm. The algorithm will stop in
% finite number of step, and not increase the objective function when return. It
% will do it best to make the 'alpha' satisfy the Wolfe conditions:
%   f(x) <= f(x0) + c1 * alpha * dot(h, f'(x0)),     --- (W1)
%   dot(h, f'(x)) >= c2 * dot(h, f'(x0)).            --- (W2)
%
%   Author: Ying Xiong.
%   Created: Apr 29, 2014.

% Setup parameters.
c1 = 1e-4;
c2 = 0.9;
b_max = 1e2;          % Maximum allowable step size.
max_iter = 10;        % Maximum number of iterations in line search.
% Evaluating phi(alpha) function.
evalPhi = @(alpha)EvalFcn_(fcn, alpha, x0, h);
% Constants for Wolfe conditions.
dphi0 = dot(h, g0);
c1_dphi0 = c1*dphi0;
c2_dphi0 = c2*dphi0;

% Setup initial brackets [a,b] = [0,1].
a = 0;   fa = f0;   dphi_a = dphi0;
b = 1;   [x, fb, g, dphi_b] = evalPhi(b);

% Refine the bracket such that 'b' does not satisfy (W1) or 'b' satisfies (W2).
while ((fb <= f0 + b*c1_dphi0) && (dphi_b < c2_dphi0) && (b < b_max))
  a = b;   b = 2*b;
  fa = fb;   dphi_a = dphi_b;
  [x, fb, g, dphi_b] = evalPhi(b);
end

% Find 'alpha' in bracket [a,b] that satisfies both (W1) and (W2).
iter = 0;
alpha = b;   f = fb;   dphi_alpha = dphi_b;
while ((f > f0 + alpha*c1_dphi0) || (dphi_alpha < c2_dphi0) && ...
       iter < max_iter)
  iter = iter+1;
  c = (fb - fa - (b-a)*dphi_a) / (b-a)^2;
  if (c > 0)
    alpha = min(0.9*a+0.1*b, max(0.1*a+0.9*b, a-dphi_a/2/c));
  else
    alpha = (a+b)/2;
  end
  [x, f, g, dphi_alpha] = evalPhi(alpha);
  if (f <= f0 + alpha*c1_dphi0)
    a = alpha;   fa = f;   dphi_a = dphi_alpha;
  else
    b = alpha;   fb = f;   dphi_b = dphi_alpha;
  end
end

end

function [x, f, g, dphi_alpha] = EvalFcn_(fcn, alpha, x0, h)

x = x0 + alpha * h;
[f, g] = fcn(x);
dphi_alpha = h' * g;

end
