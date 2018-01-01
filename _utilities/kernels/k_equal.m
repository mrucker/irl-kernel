function k = k_equal(b)    
    k = @(x1,x2) b(x1,x2) == 0;
end
