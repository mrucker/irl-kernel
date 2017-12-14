
%Y_Linearly_Seperable_Y_Centered();
%N_Linearly_Seperable_Y_Centered();
%Y_Linearly_Seperable_N_Centered();
N_Linearly_Seperable_N_Centered();

function Y_Linearly_Seperable_Y_Centered()
    %not linearly separable, centered at 0;
    x = [-1 -1; 1 1; -1 1; 1 -1]';
    y = [1 -1 1 -1]';

    Tests(x,y);
end

function N_Linearly_Seperable_Y_Centered()
    %not linearly separable, centered at 0;
    x = [-1 -1; 1 1; -1 1; 1 -1]';
    y = [1 1 -1 -1]';
    
    Tests(x,y);
end

function Y_Linearly_Seperable_N_Centered()
    %not linearly separable, centered at 0;
    
    y = [1 -1 1 -1]';

    %shifting causes several incorrect classifications because the four points are seperable horizontally
    x = [-1 -1; 1 1; -1 1; 1 -1]';
    x = x + repmat(200,2,4);
    Tests(x,y);
    
    %rotating so the separation is now diagonal allows for correct classification
    x = [-1 -1; 1 1; -1 1; 1 -1]';
    x = [cos(7/4*pi) cos(1/4*pi); sin(7/4*pi) sin(1/4*pi)]*x;
    x = x + repmat(200,2,4);
    Tests(x,y);
    
end

function N_Linearly_Seperable_N_Centered()

    scale = 10;
    count = 200;
    
    x = horzcat(rand(2,count)+[0;.5],rand(2,count)+[.5;0])*scale;
    y = horzcat(ones(1,count),-ones(1,count))';        
    
    [margin(1), right(1), wrong(1), unknown, b, b0] = maxMarginOptimization_4_s(y,x,2, kernel_poly(1));
    [margin(2), right(2), wrong(2), unknown, ~, ~ ] = maxMarginOptimization_4_s(y,x,2, kernel_poly(2));
    [margin(3), right(3), wrong(3), unknown, ~, ~ ] = maxMarginOptimization_4_s(y,x,2, kernel_poly(3));
    [margin(4), right(4), wrong(4), unknown, ~, ~ ] = maxMarginOptimization_4_s(y,x,2, kernel_poly(4));
    [margin(5), right(5), wrong(5), unknown, ~, ~ ] = maxMarginOptimization_4_s(y,x,2, kernel_poly(5));
    [margin(6), right(6), wrong(6), unknown, ~, ~ ] = maxMarginOptimization_4_s(y,x,2, kernel_poly(6));
    [margin(7), right(7), wrong(7), unknown, ~, ~ ] = maxMarginOptimization_4_s(y,x,2, kernel_poly(7));
    [margin(8), right(8), wrong(8), unknown, ~, ~ ] = maxMarginOptimization_4_s(y,x,2, kernel_poly(8));
    [margin(9), right(9), wrong(9), unknown, ~, ~ ] = maxMarginOptimization_4_s(y,x,2, kernel_poly(9));
    
    scatter(x(1,y==-1),x(2,y==-1), 'r')
    hold on   
    scatter(x(1,y==1),x(2,y==1), 'b')            
    line([0 1.5]*scale, [(-b0-b(1)*0*scale)/b(2) (-b0-b(1)*1.5*scale)/b(2)]);
    hold off 
    
    horzcat(right',wrong')
end

function Tests(x,y)
    [margin, right, wrong, unknown] = maxMarginOptimization_1_h(y,x,0);
    [margin, right, wrong, unknown] = maxMarginOptimization_1_s(y,x,0);

    [margin, right, wrong, unknown] = maxMarginOptimization_2_h(y,x,0);
    [margin, right, wrong, unknown] = maxMarginOptimization_2_s(y,x,0);

    [margin, right, wrong, unknown] = maxMarginOptimization_3_s(y,x,0);    
    
    [margin, right, wrong, unknown] = maxMarginOptimization_4_s(y,x,0, kernel_poly(1));
    [margin, right, wrong, unknown] = maxMarginOptimization_4_s(y,x,0, kernel_poly(2));
    [margin, right, wrong, unknown] = maxMarginOptimization_4_s(y,x,0, kernel_poly(3));
    [margin, right, wrong, unknown] = maxMarginOptimization_4_s(y,x,0, kernel_poly(10));
end

%Optimizers
function [margin, right, wrong, unknown] = maxMarginOptimization_1_h(y, x, verbosity)
    f_cnt = size(x,1);
    o_cnt = size(x,2);
        
    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables m w(f_cnt);
        maximize(m);
        subject to
            1 >= norm(w);
            m <= y.*(x'*w);
    cvx_end
    warning('off','all')
    
    d = y.*(x'*w);
    
    margin  = m;
    right   = sum(sign(d) == 1);
    wrong   = sum(sign(d) == -1);
    unknown = sum(sign(d) == 0);
end
function [margin, right, wrong, unknown] = maxMarginOptimization_1_s(y, x, verbosity)
    f_cnt = size(x,1);
    o_cnt = size(x,2);
        
    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables m w(f_cnt) z(o_cnt);
        maximize(m - sum(z)); %in this case z can easily grow much faster than m causing a tug of war
        subject to
            1 >= w'*w;
            m <= y.*(x'*w) + z;
            0 <= z;
    cvx_end
    warning('off','all')
    
    d = y.*(x'*w);
    
    margin  = m;
    right   = sum(sign(d) == 1);
    wrong   = sum(sign(d) == -1);
    unknown = sum(sign(d) == 0);
end
function [margin, right, wrong, unknown] = maxMarginOptimization_2_h(y, x, verbosity)
    f_cnt = size(x,1);
    o_cnt = size(x,2);
    
    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables w(f_cnt);
        minimize(norm(w));
        subject to
            1 <= y.*(x'*w);
    cvx_end
    warning('off','all')

    d = y.*(x'*w);

    margin  = 1/norm(w);
    right   = sum(sign(d) == 1);
    wrong   = sum(sign(d) == -1);
    unknown = sum(sign(d) == 0);
end
function [margin, right, wrong, unknown] = maxMarginOptimization_2_s(y, x, verbosity)
    f_cnt = size(x,1);
    o_cnt = size(x,2);
    
    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables w(f_cnt) z(o_cnt);
        minimize(norm(w) + sum(z));
        subject to
            1 <= y.*(x'*w) + z;
            0 <= z;
    cvx_end
    warning('off','all')
    
    d = y.*(x'*w);
    
    margin  = 1/norm(w);
    right   = sum(sign(d) == 1);
    wrong   = sum(sign(d) == -1);
    unknown = sum(sign(d) == 0);
end
function [margin, right, wrong, unknown] = maxMarginOptimization_3_s(y, x, verbosity)
    f_cnt = size(x,1);
    o_cnt = size(x,2);

    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables w(f_cnt);
        minimize( sum(max(0,1-y.*x'*w)) + norm(w)) %hing-loss
    cvx_end
    warning('off','all')
    
    margin  = 1/norm(w);
    
    d = y.*(x'*w);
    
    right   = sum(sign(d) == 1);
    wrong   = sum(sign(d) == -1);
    unknown = sum(sign(d) == 0);
end
function [margin, right, wrong, unknown, b, b0] = maxMarginOptimization_4_s(y, x,verbosity, k)
    f_cnt = size(x,1);
    o_cnt = size(x,2);    
        
    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables a(o_cnt);
        maximize(sum(a) - 1/2*quad_form(a.*y, k(x,x))) %dual problem
        subject to
            0 == a'*y;
            0 <= a;
    cvx_end
    warning('off','all')            
    
    %Useful to study kernel polynomials of degree 2. This is the new feature space we are working in.
    %h = @(x) [ones(1,size(x,2));sqrt(2)*x(1,:);sqrt(2)*x(2,:);x(1,:).*x(1,:);x(2,:).*x(2,:);sqrt(2)*x(1,:).*x(2,:)];    
    %h_x = h(x);
    
    %When working in higher dimensions I'm not sure this has any meaning. (such as polynomial kernels above 1).
    %When working within the dimensions of x this is the normal to the hyperplane.
    %w = x*(a.*y);
    
    %regarding b0: "we typically use an average of all the solutions for numerical stability" (ESL pg.421)
    b0 = mean(1./y - k(x, x)'*(a.*y));
    
    d = y.*(k(x,x)*(a.*y) + b0);
    
    margin  = 1/sqrt(sum(a));
    right   = sum(sign(d) == 1);
    wrong   = sum(sign(d) == -1);
    unknown = sum(sign(d) == 0);
    b       = x*(a.*y);
end
%Optimizers

%Kernels
function k = kernel_poly(p)
    
    assert( p > 0, 'What are you doing!?');

    if p == 1
        k = @(x1,x2) x1'*x2;
    else
        k = @(x1,x2) power(x1'*x2 + ones(size(x1,2), size(x2,2)), p*ones(size(x1,2), size(x2,2)));
    end
end
%Kernels
