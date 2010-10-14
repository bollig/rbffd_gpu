function [] = checkSymmetry(SparseMatA, filename, testCaseName)
[m n] = size(SparseMatA);

% Get a matrix with only the non-symmetric 
F = spSymmetricFilter(SparseMatA); 

figure; 
hold off; 
spy(F, 5); 
axis('square');
if (nargin < 3)
    titlestr1 = 'Non-symmetric Stencil Edges'; 
else 
    titlestr1 = sprintf('[%s] Non-symmetric Stencil Edges', testCaseName); 
end
title(titlestr1);
ylabel('Stencil ID (S)'); 
xlabel('IDs of stencils which do not contain S in the stencil set');
%print('nonSymEdges.png', '-dpng', '-r300');


% 1 = size of marker
figure; 
hold off; 
spy(SparseMatA, 5); 
if (nargin < 3) 
    label1 = sprintf('Sparsity pattern of %s (Dimensions: %d x %d)', filename, m, n);
else 
    label1 = sprintf('[%s] Sparsity pattern of %s (Dimensions: %d x %d)', testCaseName, filename, m, n);
end
axis('square');
title(label1); 
xlabel('column');
ylabel('row'); 
%print('fullMatSpy.png', '-dpng', '-r300');


figure;
hold off; 
image(full(SparseMatA)); 
axis('square');
if (nargin < 3)
    titlestr2 = 'Stencil weight distribution of full system'; 
else 
    titlestr2 = sprintf('[%s] Stencil weight distribution of full system', testCaseName); 
end
title(titlestr2);
%print('fullMatWeights.png', '-dpng', '-r300');

spMeasureSymmetry(SparseMatA);


end

