function [x, F, f, exitflag] = NLLS_LM(fcn, x0, opts)

% Levenberg-Marquardt algorithm for nonlinear least squares.
%
%   Author: Ying Xiong.
%   Created: May 10, 2014.

% Get options from the struct.
[tau, JJDamp] = GetLMOptions(opts);

%% Initialization.
N = length(x0);
x = x0;
[f, J] = fcn(x);
F = sum(f.^2);
JJ = J' * J;
Jf = J' * f;
mu = tau * max(diag(JJ));
if (mu == 0)
  % J is an zero matrix.
  exitflag = 1;
  return;
end
mu_min = 1e-12;
nu = 2;

% The followings might be needed for the first iteration's stop criternion, in
% case one starts at a local minimum.
x_old = x;
F_old = F;
if (opts.Display >= 3)
  fprintf('  Iter       F(x)\n');
  fprintf('%6d    %.4e\n', 0, F);
end

%% Main loop.
for iter = 1:opts.MaxIter
  % Compute direction 'h'.
  if (JJDamp)    h = -(JJ + mu*diag(diag(JJ))) \ Jf;
  else           h = -(JJ + mu*eye(N)) \ Jf;        end
  % Compute gain ratio 'rho'.
  x_new = x + h;
  [f, J] = fcn(x_new);
  F_new = sum(f.^2);
  if (JJDamp)    rho_denom = h' * (mu*diag(diag(JJ))*h - Jf);
  else           rho_denom = h' * (mu*h - Jf);                   end
  rho = (F - F_new) / rho_denom;
  % Update the variable if step is accepted.
  if (rho > 0)
    % Step accepted.
    x_old = x;   x = x_new;
    F_old = F;   F = F_new;
    JJ = J' * J;
    Jf = J' * f;
    mu = max(mu_min, mu * max(1/3, 1-(2*rho-1)^3));
    nu = 2;
  else
    % Step not accepted.
    mu = mu*nu;
    nu = 2*nu;
  end
  % Display information.
  if (opts.Display >= 3)
    fprintf('%6d    %.4e\n', iter, F);
    if (opts.Display >= 4)
      fprintf('rho=%.3f, mu=%8.4e, nu=%d\n', rho, mu, nu);
    end
  end
  % Check the stop criterion.
  exitflag = StopCriterion(rho_denom, rho, x_old, x, F_old, F, opts);
  if (exitflag)    break;    end
end

end

function s = StopCriterion(rho_denom, rho, x_old, x, F_old, F, opts)

if (rho_denom < eps)
  % When this happens, the step size is usually very small and the calculation of
  % 'rho' is below numerical accuracy. We claim to find a local minimum.
  s = 1;
  return;
end

if (rho > 0)
  % If a step has been made.
  if (max(abs(x_old-x)) < opts.TolX)               s = 2;
  elseif ((F_old-F) < opts.TolFun)                 s = 3;
  else                                             s = 0;
  end
else
  s = 0;
end

end

function [tau, JJDamp] = GetLMOptions(options)

% Get Levenberg-Marquardt specific options.
if (~isfield(options, 'LMtau'))               tau = 1e-3;
else   tau = options.tau;   end

if (~isfield(options, 'LMDampMatx'))          JJDamp = 0;
elseif (strcmp(options.LMDampMatx, 'eye'))    JJDamp = 0;
elseif (strcmp(options.LMDampMatx, 'JJ'))     JJDamp = 1;
else   error('Unknown ''options.LMDampMatx''.');
end

end
