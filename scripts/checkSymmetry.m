function [] = checkSymmetry(SparseMatA, filename)
[m n] = size(SparseMatA);

% Get a matrix with only the non-symmetric 
F = spSymmetricFilter(SparseMatA); 

figure(1); 
hold off; 
spy(F, 1); 
title('Non symmetric stencil edges');
ylabel('Stencil ID (S)'); 
xlabel('IDs of stencils which do not contain S in the stencil set');
print('nonSymEdges.png', '-dpng', '-r300');


% 1 = size of marker
figure(2); 
hold off; 
spy(SparseMatA, 1); 
label1 = sprintf('Sparsity pattern of %s (Dimensions: %d x %d)', filename, m, n);
title(label1); 
xlabel('column');
ylabel('row'); 
print('fullMatSpy.png', '-dpng', '-r300');

figure(3)
hold off; 
image(full(SparseMatA)); 
title('Stencil weight distribution of full system');
print('fullMatWeights.png', '-dpng', '-r300');

spMeasureSymmetry(SparseMatA);


end

