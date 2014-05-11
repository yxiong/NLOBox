%   Author: Ying Xiong.
%   Created: Jan 20, 2014.

rng(0);

x = randn();
CheckGradient(@(a)NLOTest1DCurve(x, a), 4);

fprintf('Passed.\n');
