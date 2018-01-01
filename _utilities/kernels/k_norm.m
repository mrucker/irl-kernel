function k = k_norm()
    k = @(x1,x2) x1'*x1 + x2'*x2 - 2*x1'*x2;
end