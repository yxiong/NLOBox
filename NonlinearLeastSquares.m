function [x, F, f, exitflag] = NonlinearLeastSquares(fcn, x0, lb, ub, opts)

% x = NonlinearLeastSquares(fcn, x0)
% x = NonlinearLeastSquares(fcn, x0, lb, ub)
% x = NonlinearLeastSquares(fcn, x0, lb, ub, options)
% [x, F, f] = NonlinearLeastSquares(...)
% [x, F, f, exitflag] = NonlinearLeastSquares(...)
%
% Perform nonlinear least squares optimization to minimize the cost function
%   F(x) = sum(fcn(x).^2).
% This function has very similar interface to 'lsqnonlin'.
%
% INPUT:
%   fcn: a vector fuction to be minimized, with the form
%            [f, J] = fcn(x)
%        with 'x' an input Nx1 vector, 'f' an Mx1 vector for function values and
%        'J' a MxN matrix for the Jacobian matrix.
%        One does not have to compute the Jacobian of the function, by simply
%        providing
%            f = fcn(x)
%        and set 'options.Jacobian=off'.
%   x0:  initial guess, Nx1 vector.
%   lb, ub: the lower and upper bound of the variable 'x'. Can be [] or 'Inf'
%           if not bound needs to be enforced.
%   options: a struct with following supported fields.
%     'Algorithm': the algorithm used for optimization, currently only
%                support 'lm' (Levenberg-Marquardt).
%                NOTE: the options are different from 'lsqnonlin'.
%     'DerivativeCheck': compare the user-supplied derivatives with to
%                finite-differencing ones, options {'off'} or 'on'. The gradient
%                will only be checked at 'x0' and therefore not impose a big
%                performance penality.
%     'Display': Level of display, options {'off'/'none'}, 'final',
%                'final-detailed', 'iter', 'iter-detailed'.
%                NOTE: the default is different from 'lsqnonlin'.
%                [TODO] The 'final-detailed' option is not properly supported yet.
%     'Jacobian':Whether to use the Jacobian given by 'fcn' or perform finite
%                difference. Options are {'on'} (use 'fcn') or 'off' (use finite
%                difference).
%                NOTE: the default is different from 'lsqnonlin'.
%     'MaxIter': Maximum number of iterations allowed, default {400}.
%     'TolFun':  Termination tolerance on 'f', default {1e-6}.
%     'TolX':    Termination tolerance on 'x', default {1e-6}.
%     ---- NOTE: The following options are not in 'lsqnonlin'. ----
%     ---- For Levenberg-Marquardt algorithm only. ----
%     'LMtau': the 'tau' parameter, default {1e-3}.
%     'LMDampMatx': the damping matrix, options {'eye'} or 'JJ'. Use the latter
%                   if the problem is poorly scaled.
%
% OUTPUT:
%   x: the output local minimum.
%   F: the cost at 'x', i.e. sum(fcn(x).^2).
%   f: the vector function value at 'x', i.e. fcn(x).
%   exitflag: an integer describing the exit condition, with following values
%     0: number of iterations exceeded 'options.MaxIter'.
%     1: function converges to a solution 'x'.
%     2: change in 'x' less than 'TolX'.
%     3: change in 'f' less than 'TolFun'.
%     4: mangitude of search direction smaller than 'eps'.
%
%   Author: Ying Xiong.
%   Created: Jan 20, 2014.

% Check input and setup parameters.
if (~exist('lb', 'var'))        lb = [];             end
if (~exist('ub', 'var'))        ub = [];             end
if (~exist('opts', 'var'))      opts = struct();     end
opts = ProcessOptions(opts);

% Remove the bounded constraint.
N = length(x0);
if ((~isempty(lb) && any(isfinite(lb))) || (~isempty(ub) && any(isfinite(ub))))
  BoundedConstraint = 1;
  if (isempty(lb))    lb = -Inf(N,1);   end
  if (isempty(ub))    ub =  Inf(N,1);   end
  lb = lb(:);   ub = ub(:);
  assert(all(lb<=x0) && all(x0<=ub));
  fcn = BoundedVecFcnToUnconstrainedVecFcn(fcn, lb, ub);
  mapfcn = MapBoundedToUnconstrained(lb, ub);
  x0 = mapfcn(x0);
else
  BoundedConstraint = 0;
end

% Check gradient.
if (opts.DerivativeCheck)
  f = fcn(x0);
  CheckJacobian(fcn, length(x0), length(f), struct('x0', x0));
end

% Add numerical gradient calculation to 'fcn' if necessary.
if (~opts.Jacobian)
  fcn = @(x)EvalWithNumericalJacobian_(fcn, x);
end

% Dispatch the job to specific optimization functions.
if (opts.Algorithm == 0)
  [x, F, f, exitflag] = NLLS_LM(fcn, x0, opts);
else
  error('Internal error: unknonw algorithm.');
end

% Add back the bounded constraint.
if (BoundedConstraint)
  mapfcn = MapUnconstrainedToBounded(lb, ub);
  x = mapfcn(x);
end

% Display final results.
if (opts.Display >= 1)
  fprintf('Terminate: ');
  if (exitflag == 0)
    fprintf('maximum number of iterations (%d) reached.\n', opts.MaxIter);
  elseif (exitflag == 1)
    fprintf('local minimum reached.\n');
  elseif (exitflag == 2)
    fprintf('change in ''x'' less than ''TolX'' (%g).\n', opts.TolX);
  elseif (exitflag == 3)
    fprintf('change in ''F'' less than ''TolFun'' (%g).\n', opts.TolFun);
  elseif (exitflag == 4);
    fprintf('magnitude of search direction less than ''eps''.\n');
  else
    error('Unknown ''exitflag'' %d.', exitflag);
  end
end

end

function opts = ProcessOptions(opts)

% Process the options: set default values, onvert strings to integers, etc.

if (~isfield(opts, 'Algorithm'))                 opts.Algorithm = 0;
elseif (strcmp(opts.Algorithm, 'lm'))            opts.Algorithm = 0;
else   error('Unknonwn options.Algorithm ''%s''', opts.Algorithm);
end

if (~isfield(opts, 'DerivativeCheck'))           opts.DerivativeCheck = 0;
elseif (strcmp(opts.DerivativeCheck, 'off'))     opts.DerivativeCheck = 0;
elseif (strcmp(opts.DerivativeCheck, 'on'))      opts.DerivativeCheck = 1;
else   error('Unknown options.DerivativeCheck ''%s''', opts.DerivativeCheck);
end

if (~isfield(opts, 'Display'))                   opts.Display = 0;
elseif (strcmp(opts.Display, 'none'))            opts.Display = 0;
elseif (strcmp(opts.Display, 'off'))             opts.Display = 0;
elseif (strcmp(opts.Display, 'final'))           opts.Display = 1;
elseif (strcmp(opts.Display, 'final-detailed'))  opts.Display = 2;
elseif (strcmp(opts.Display, 'iter'))            opts.Display = 3;
elseif (strcmp(opts.Display, 'iter-detailed'))   opts.Display = 4;
else   error('Unknown options.Display ''%s''.', opts.Display);
end

if (~isfield(opts, 'Jacobian'))                  opts.Jacobian = 1;
elseif (strcmp(opts.Jacobian, 'on'))             opts.Jacobian = 1;
elseif (strcmp(opts.Jacobian, 'off'))            opts.Jacobian = 0;
else   error('Unknown options.Jacobian ''%s''.', opts.Jacobian);
end

if (~isfield(opts, 'MaxIter'))                   opts.MaxIter = 400;   end
if (~isfield(opts, 'TolFun'))                    opts.TolFun = 1e-6;   end
if (~isfield(opts, 'TolX'))                      opts.TolX = 1e-6;     end

end


function [f, J] = EvalWithNumericalJacobian_(fcn, x)
[J, f] = NumericalJacobian(fcn, x);
end
