%   Author: Ying Xiong.
%   Created: Apr 28, 2014.

rng(0);

%% Fitting a 1D curve.
% Set up model parameters.
a_gt = [1; -0.5; 2; 3];
a_gt2 = [a_gt(2); a_gt(1); a_gt(4); a_gt(3)];
nData = 101;
xLims = [-4; 3];
noiseStd = 0.1;

% Generate data.
x = xLims(1) + (xLims(2)-xLims(1)) * rand(nData, 1);
y = NLOTest1DCurve(x, a_gt) + randn(nData, 1) * noiseStd;

% Function to be minimized.
fcn = @(a)NLOCurveToCost(a, @NLOTest1DCurve, x, y);

% 'Regular' mode.
options = struct('Display', 'off', 'DerivativeCheck', 'on');
a0 = randn(4, 1);
a = NonlinearMinimization(fcn, a0, options);
assert(CheckNear(a, a_gt, 0.01) || CheckNear(a, a_gt2, 0.01));

% Optimization with bounded constraints.
a0(2) = -abs(a0(2));
a0(3) = -1 + 4*rand(1);
a0(4) = abs(a0(4)) + 1;
options = struct('Display', 'off', ...
                 'LowerBound', [-Inf, -Inf, -1, 1], ...
                 'UpperBound', [Inf, 0, 3, Inf]);
a = NonlinearMinimization(fcn, a0, options);
assert(CheckNear(a, a_gt, 0.01));

% Use finite difference to compute gradient.
fcn = @(a)NLOCurveToCost(a, @NLOTest1DCurveNoJacobian, x, y);
options = struct('Display', 'off', 'GradObj', 'off');
a = NonlinearMinimization(fcn, a0, options);
assert(CheckNear(a, a_gt, 0.01) || CheckNear(a, a_gt2, 0.01));

fprintf('Passed.\n');
