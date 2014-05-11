function exitflag = NLM_StopCriterion(x_old, x, TolX, f_old, f, TolFun, g)

% Stop criterion for nonlinear minimization.
%
%   Author: Ying Xiong.
%   Created: May 10, 2014.

if (max(abs(g)) < TolFun)                    exitflag = 1;
elseif (max(abs(x-x_old)) < TolX)            exitflag = 2;
elseif (f_old-f < TolFun)                    exitflag = 3;
else                                         exitflag = 0;
end
