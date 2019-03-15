function [j,reg]=costfunction(y,a,w1,w2)
    j=-y*transpose(log(a))-(1-y)*transpose(log(1-a));
    reg=sum(sum(w1.*w1))+sum(sum(w2.*w2));
end