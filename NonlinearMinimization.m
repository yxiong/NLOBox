function [x, f, exitflag] = NonlinearMinimization(fcn, x0, opts)

% x = NonlinearMinimization(fcn, x0)
% x = NonlinearMinimization(fcn, options)
% [x, f] = NonlinearMinimization(...)
% [x, f, exitflag] = NonlinearMinimization(...)
%
% Perform nonlinear minimization to the cost function 'fcn'.
% This function has very similar interface to 'fminunc'.
%
% INPUT:
%   fcn:  the objective function to be minimized, with the form
%             [f, g] = fcn(x)
%         with 'x' an input Nx1 vector, 'f' a scalar output and 'g' an Nx1
%         vector for gradient of 'f' over 'x'.
%   x0:   initial guess, Nx1 vector.
%   options: a struct with following supported fields.
%     'Algorithm': The algorithm used for optimization, currently only
%                support 'bfgs'.
%                NOTE: the options are different from 'fminunc'.
%     'DerivativeCheck': Compare user-supplied derivatives (gradient of
%                objective) to  finite-differencing derivatives. Options are
%                'on' or {'off'}.
%     'Display': Level of display, options {'off'/'none'}, 'notify',
%                'notify-detailed', 'final', 'final-detailed', 'iter',
%                'iter-detailed'.
%     'GradObj': Whether to use the gradient of the objective function 'fcn'
%                or perform finite difference. Options are {'on'} (use 'fcn')
%                or 'off' (use finite difference).
%                NOTE: the default is different from 'fminunc'.
%     'MaxIter': Maximum number of iterations allowed, default {400}.
%     'TolFun':  Terminate tolerance on 'f', default 1e-6.
%     'TolX':    Terminate tolerance on 'x', default 1e-6.
%     ---- NOTE: The following options are not in 'fminunc'. ----
%     'LowerBound': Nx1 vector for lower bound of 'x', entries can be -Inf.
%     'UpperBound': Nx1 vector for upper bound of 'x', entries can be +Inf.
%
% OUTPUT:
%   x: the output local minimum.
%   f: the cost at 'x'.
%   exitflag: an integer describing the exit condition, with following values
%     0: number of iterations exceeded 'options.MaxIter'.
%     1: magnitude of gradient smaller than 'options.TolFun'.
%     2: change in 'x' smaller than 'options.TolX'.
%     3: change in 'f' smaller than 'options.TolFun'.
%
%   Author: Ying Xiong.
%   Created: Apr 28, 2014.

% Process the input options.
if (~exist('opts', 'var'))     opts = struct();     end
opts = ProcessOptions(opts);

% Remove the bounded constraint.
if (opts.HasBoundConstraints_)
  fcn = BoundedFcnToUnconstrainedFcn(fcn, opts.LowerBound, opts.UpperBound);
  mapfcn = MapBoundedToUnconstrained(opts.LowerBound, opts.UpperBound);
  x0 = mapfcn(x0);
end

% Check gradient.
if (opts.DerivativeCheck)
  CheckGradient(fcn, length(x0), struct('x0', x0));
end

% Add numerical gradient calculation to 'fcn' if necessary.
if (~opts.GradObj)
  fcn = @(x)EvalWithNumericalGradient_(fcn, x);
end

% Dispatch the job to specific optimization functions.
if (opts.Algorithm == 0)
  [x,f,exitflag] = NLM_BFGS(fcn, x0, opts);
else
  error('Internal error: unknown algorithm.');
end

% Add back the bounded constraint.
if (opts.HasBoundConstraints_)
  mapfcn = MapUnconstrainedToBounded(opts.LowerBound, opts.UpperBound);
  x = mapfcn(x);
end

end

function opts = ProcessOptions(opts)

% Process the options: set default values, convert strings to integers, etc.

if (~isfield(opts, 'Algorithm'))                 opts.Algorithm = 0;
elseif (strcmp(opts.Algorithm, 'bfgs'))          opts.Algorithm = 0;
else   error('Unknown options.Algorithm ''%s''', opts.Algorithm);
end

if (~isfield(opts, 'DerivativeCheck'))           opts.DerivativeCheck = 0;
elseif (strcmp(opts.DerivativeCheck, 'off'))      opts.DerivativeCheck = 0;
elseif (strcmp(opts.DerivativeCheck, 'on'))     opts.DerivativeCheck = 1;
else   error('Unknown options.DerivativeCheck ''%s''', opts.DerivativeCheck);
end

if (~isfield(opts, 'Display'))                   opts.Display = 0;
elseif (strcmp(opts.Display, 'off'))             opts.Display = 0;
elseif (strcmp(opts.Display, 'none'))            opts.Display = 0;
elseif (strcmp(opts.Display, 'notify'))          opts.Display = 1;
elseif (strcmp(opts.Display, 'notify-detailed')) opts.Display = 2;
elseif (strcmp(opts.Display, 'final'))           opts.Display = 3;
elseif (strcmp(opts.Display, 'final-detailed'))  opts.Display = 4;
elseif (strcmp(opts.Display, 'iter'))            opts.Display = 5;
elseif (strcmp(opts.Display, 'iter-detailed'))   opts.Display = 6;
else   error('Unknown options.Display ''%s''', opts.Display);
end

if (~isfield(opts, 'GradObj'))                   opts.GradObj = 1;
elseif (strcmp(opts.GradObj, 'on'))              opts.GradObj = 1;
elseif (strcmp(opts.GradObj, 'off'))             opts.GradObj = 0;
else   error('Unknown options.GradObj ''%s''', opts.GradObj);
end

if (~isfield(opts, 'MaxIter'))                   opts.MaxIter = 400;   end
if (~isfield(opts, 'TolFun'))                    opts.TolFun = 1e-6;   end
if (~isfield(opts, 'TolX'))                      opts.TolX = 1e-6;     end

opts.HasBoundConstraints_ = 0;
if (isfield(opts, 'LowerBound') && any(isfinite(opts.LowerBound)))
  opts.LowerBound = opts.LowerBound(:);
  opts.HasBoundConstraints_ = 1;
end
if (isfield(opts, 'UpperBound') && any(isfinite(opts.UpperBound)))
  opts.UpperBound = opts.UpperBound(:);
  opts.HasBoundConstraints_ = 1;
end
if (opts.HasBoundConstraints_ && ~isfield(opts, 'LowerBound'))
  opts.LowerBound = -Inf(sizeof(opts.UpperBound));
end
if (opts.HasBoundConstraints_ && ~isfield(opts, 'UpperBound'))
  opts.UpperBound = Inf(sizeof(opts.LowerBound));
end

end

function [f, g] = EvalWithNumericalGradient_(fcn, x)
[g, f] = NumericalGradient(fcn, x);
end
