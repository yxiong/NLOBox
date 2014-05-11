function y = NLOTest1DCurveNoJacobian(x, a)

% y = NLOTest1DCurveNoJacobian(x, a)
%
% Same as 'NLOTest1DCurve', but without providing Jacobian.
%
%   Author: Ying Xiong.
%   Created: Feb 05, 2014.

y = a(3) * exp(a(1)*x) + a(4)*exp(a(2)*x);
