function [c, dc] = NLOCurveToCost(a, curveFcn, x, y)

% [c, dc] = NLOCurveToCust(a, curveFcn, x, y)
%
% Turn a curve model function to a cost function that can be minimized by
% 'NonlinearMinimization.m' function.
%
%   Author: Ying Xiong.
%   Created: Apr 28, 2014.


if (nargout == 1)
  ya = curveFcn(x, a);
  c = sum((ya-y).^2);
else
  [ya, dya] = curveFcn(x, a);
  ydiff = ya - y;
  c = sum((ya-y).^2);
  dc = 2 * sum(dya .* repmat(ydiff, [1, length(a)]))';
end
