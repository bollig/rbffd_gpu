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
    title(titlestr1);
else 
    titlestr1 = sprintf('[%s] ', testCaseName); 
    titlestr2 = sprintf('Non-symmetric Stencil Edges'); 
    title({titlestr1, titlestr2});
end

ylabel('Stencil ID (S)'); 
xlabel('IDs of stencils which do not contain S in the stencil set');
%print('nonSymEdges.png', '-dpng', '-r300');


% 1 = size of marker
figure; 
hold off; 
spy(SparseMatA, 5); 
if (nargin < 3) 
    label1 = sprintf('Sparsity pattern of %s (Dimensions: %d x %d)', filename, m, n);
    title(label1); 
else 
    label1 = sprintf('[%s] ', testCaseName);
    label2 = sprintf('Sparsity pattern of %s (Dimensions: %d x %d)', filename, m, n);
    title({label1, label2}); 
end
axis('square');
xlabel('column');
ylabel('row'); 
%print('fullMatSpy.png', '-dpng', '-r300');


figure;
hold off; 
image(full(SparseMatA)); 
axis('square');
if (nargin < 3) 
    label1 = 'Stencil weight distribution of full system';
    title(label1); 
else 
    label1 = sprintf('[%s] ', testCaseName);
    label2 = sprintf('Stencil weight distribution %s (Dimensions: %d x %d)', filename, m, n);
    title({label1, label2}); 
end
%print('fullMatWeights.png', '-dpng', '-r300');

spMeasureSymmetry(SparseMatA);


end

