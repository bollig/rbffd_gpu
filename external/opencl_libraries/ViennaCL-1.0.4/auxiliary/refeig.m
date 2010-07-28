
function refeig()

dim = 500;

A(dim,dim) = 1.0;

for i=1:dim
  A(i,i) = i;
  A(1,i) = -2.0;
  A(i,1) = -2.0; 
end

eig(A)
endfunction

