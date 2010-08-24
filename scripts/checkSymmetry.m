function [] = checkSymmetry(SparseMatA)

% Get a matrix with only the non-symmetric 
F = spSymmetricFilter(SparseMatA); 

figure(1); 
hold off; 
spy(F, 1); 
title('Non symmetric stencil edges');
ylabel('Stencil ID (S)'); 
xlabel('IDs of stencils which do not contain S in the stencil set');


% 1 = size of marker
figure(2)
hold off;
spy(SparseMatA, 1);
title('Sparse representation of full system');

figure(3)
hold off; 
image(full(SparseMatA)); 
title('Stencil weight distribution of full system');

spMeasureSymmetry(SparseMatA);


end

