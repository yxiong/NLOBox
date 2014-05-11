function NLOCurveToVecCostTest()

%   Author: Ying Xiong.
%   Created: Jan 20, 2014.

rng(0);

x = randn(100, 1);
y = randn(100, 1);
CheckGradient(@(a)NLOCurveToVecCostTestCost(a, @NLOTest1DCurve, x, y), 4);

fprintf('Passed.\n');

end

function [aggC, aggDc] = NLOCurveToVecCostTestCost(a, curveFcn, x, y)

[c, dc] = NLOCurveToVecCost(a, curveFcn, x, y);
aggC = sum(c(:));
aggDc = sum(dc, 1);

end
