function [y, dy] = NLOTest1DCurve(x, a)

% [y, dy] = NLOTest1DCurve(x, a)
%
% A 1D curve function for test.
%   y(x; a) = a(3) * exp(a(1)*x) + a(4)*exp(a(2)*x).
%
% The second output 'dy' is the gradient of 'y' over 'a', which is a Nx4 matrix.
%
%   Author: Ying Xiong.
%   Created: Jan 20, 2014.

MAX = 1e100;
MIN = -1e100;
y1 = a(3) * exp(a(1)*x);   y1(y1>MAX) = MAX;   y1(y1<MIN) = MIN;
y2 = a(4) * exp(a(2)*x);   y2(y2>MAX) = MAX;   y2(y2<MIN) = MIN;
y = y1 + y2;
if (nargout > 1)
  dy = [x.*a(3).*exp(a(1)*x), ...
        x.*a(4).*exp(a(2)*x), ...
        exp(a(1)*x), ...
        exp(a(2)*x)];
end
