function [measure] = spMeasureSymmetry(SparseMatA)

% If the matrix is symmetric we expect B == A
% If the matrix is antisymmetric we expect B == 0
B = ( SparseMatA + SparseMatA' ) ./ 2;

% Therefore we can take the 2 norm of B and give a scalar
% measure of how symmetric our matrix is. 
measureOfSymmetry = norm(B, 1); 

fprintf(1, 'Symmetry Measure ||(A + A^T)/2||_2 = %e \n', measureOfSymmetry);
fprintf(1, 'NOTE: low Symmetry Measure implies antisymmetric matrix\n')

% Force OCTAVE to flush output. Matlab calls this by default
flush_io(1);

% We can also count the number of stencil edges which are not symmetric
end
