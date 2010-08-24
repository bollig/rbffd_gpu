function [] = checkEigenvalues(SparseMatA, filename)

[m n] = size(SparseMatA);

% Compute all Eigenvalues (LAMBDA) and Vectors (V)
[V, LAMBDA] = eig(full(SparseMatA)); 

figure(1); 
hold off; 
spy(SparseMatA); 
label1 = sprintf('Sparsity pattern of %s (Dimensions: %d x %d)', filename, m, n);
title(label1); 
xlabel('column');
ylabel('row'); 

figure(2); 
hold off; % clear plot if anything is already there
plot(diag(LAMBDA), 'ko'); 
label2 = sprintf('Complex Plane Eigenvalue Plot for %s (Num Eigenvalues = %d)', filename, m); 
title(label2); 
xlabel('Real'); 
ylabel('Imaginary'); 


end
