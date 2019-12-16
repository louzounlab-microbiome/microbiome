% compute data log likelihood under negativ bionomial model
function [L] = NBlogLikErr2(XO,EXX,ev) 
    L = gammaln(XO+ones(size(XO))./ev) -gammaln(ones(size(XO))./ev)+ XO.*log(EXX) -log(ev)./ev -(XO+ones(size(XO))./ev).*log(EXX+ones(size(XO))./ev);
end

