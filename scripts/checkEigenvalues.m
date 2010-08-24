function [] = checkEigenvalues(SparseMatA, filename)

[m n] = size(SparseMatA);

% The first measure of conditioning: 
fprintf(1,'Condition Number (condest(A) [for 1-norm estimate]): %e\n', condest(SparseMatA));

% Force OCTAVE to flush output. Matlab calls this by default
fflush(1);

% Compute all Eigenvalues (LAMBDA) and Vectors (V)
[V, LAMBDA] = eig(full(SparseMatA)); 

figure(5); 
hold off; 
spy(SparseMatA, 1); 
label1 = sprintf('Sparsity pattern of %s (Dimensions: %d x %d)', filename, m, n);
title(label1); 
xlabel('column');
ylabel('row'); 

figure(6); 
hold off; % clear plot if anything is already there
plot(diag(LAMBDA), 'ko'); 
label2 = sprintf('Complex Plane Eigenvalue Plot for %s (Num Eigenvalues = %d)', filename, m); 
title(label2); 
xlabel('Real'); 
ylabel('Imaginary'); 


end
