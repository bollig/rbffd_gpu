function [] = checkEigenvalues(SparseMatA, filename, testCaseName)

[m n] = size(SparseMatA);

% The first measure of conditioning: 
fprintf(1,'Condition Number (condest(A) [for 1-norm estimate]): %e\n', condest(SparseMatA));

% Force OCTAVE to flush output. Matlab calls this by default
flush_io(1);

% Compute all Eigenvalues (LAMBDA) and Vectors (V)
[V, LAMBDA] = eig(full(SparseMatA)); 


figure; 
hold off; % clear plot if anything is already there
eigenvalues = diag(LAMBDA); 
plot(eigenvalues, 'ko'); 
axis('square');

if (nargin < 3)
    titlestr=sprintf('Complex Plane Eigenvalue Plot for %s (Num Eigenvalues = %d)', filename, m);  
    title(titlestr);
else 
    titlestr1=sprintf('[%s] ', testCaseName);
    titlestr2=sprintf('Complex Plane Eigenvalue Plot for %s (Num Eigenvalues = %d)', filename, m); 
    title({titlestr1,titlestr2});
end
xlabel('Real'); 
ylabel('Imaginary'); 
%print('fullMatEigenvalues.png', '-dpng', '-r300');


save('EigenVectors_realpart.mtx', 'V', '-ascii');
save('EigenVectors.mat', 'V', '-mat');

%[err] = mmwrite('Eigenvectors.mtx',V, 'Indicates non-symmetric edges. Row=StencilID, Col=Stencils which do not contain StencilID');

save('EigenValues_realpart.mtx', 'eigenvalues', '-ascii');
save('EigenValues.mat', 'eigenvalues', '-mat');


% Q: how many eigenvalues are 0 or larger
numGreater = length(find(real(eigenvalues) >= 0));
fprintf(1,'Number of EigenValues >= 0: %d\n', numGreater); 

% Q: how many eigenvalues are very close to 0 or larger? 
numGreater = length(find(real(eigenvalues) >= -1e-6));
fprintf(1,'Number of EigenValues >= -1e-6: %d\n', numGreater); 

% Q: how many eigenvalues are greater than -1? 
numGreater = length(find(real(eigenvalues) > -1));
fprintf(1,'Number of EigenValues > -1: %d\n', numGreater); 


% Q: whats the MAX eigenvalue? 
[mx ind] = max(real(eigenvalues)); 
fprintf(1, 'Max EigenValue: %g+%gi\n', real(eigenvalues(ind)), imag(eigenvalues(ind))); 
% Q: whats the MIN eigenvalue? 
[mn ind] = min(real(eigenvalues)); 
fprintf(1, 'Min EigenValue: %g+%gi\n', real(eigenvalues(ind)), imag(eigenvalues(ind))); 

end
