function [x, f, exitflag] = NLM_BFGS(fcn, x0, opts)

% BFGS algorithm for nonlinear minimization.
%
%   Author: Ying Xiong.
%   Created: May 10, 2014.

%% Initialization.
N = length(x0);
x = x0;
[f, g] = fcn(x);
if (max(abs(g)) < opts.TolFun)
  exitflag = 1;
  return;
end
Binv = eye(N);

if (opts.Display >= 5)
  fprintf('  Iter         F(x)    Step size\n');
  fprintf('%6d   %.4e\n', 0, f);
end

%% Main loop.
for iter = 1:opts.MaxIter
  % Save results from previous iteration.
  x_old = x;  f_old = f;  g_old = g;
  % Compute descent direction 'h'.
  h = - Binv * g;
  % Perform line search.
  [alpha, x, f, g] = NLM_LineSearch(fcn, x, h, f, g);
  % Display results.
  if (opts.Display >= 5)
    fprintf('%6d   %.4e   %.4e\n', iter, f, alpha);
  end
  % Check stop criterion.
  exitflag = NLM_StopCriterion(x_old, x, opts.TolX, f_old, f, opts.TolFun, g);
  if (exitflag)   break;   end
  % Update 'Binv' for next iteration.
  s = x - x_old;
  y = g - g_old;
  v = Binv * y;
  Binv = Binv + (s'*y+y'*v) / ((s'*y)^2) * (s*s') - (s*v'+v*s') / (s'*y);
end

end
