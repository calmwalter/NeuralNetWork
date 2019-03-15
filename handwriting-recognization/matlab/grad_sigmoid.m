function output = grad_sigmoid(mat)
    output=sigmoid(mat).*(1.0-sigmoid(mat));
end

