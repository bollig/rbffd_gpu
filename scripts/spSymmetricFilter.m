% Mask the matrix and return sparse representation of only elements which 
% do not have a weight in the corresponding transpose element. 
function [filteredMat] = spSymmetricFilter(SparseMatA) 

% Set all non-zero elements of A to 1
%C = spfun(@spmask, SparseMatA);
C = spones(SparseMatA);

% Find elements in the matrix which are not mirrored on transpose
filteredMat = (C - C');

% Now D has 0 or (+/-)1
% if < 0 then edge is not part of C. 
ind = find( filteredMat < 0 );
filteredMat(ind) = 0; 

filename='FilteredMat.mtx'; 
[err] = mmwrite(filename,filteredMat, 'Indicates non-symmetric edges. Row=StencilID, Col=Stencils which do not contain StencilID');
fprintf(1,'Wrote %s with err=%d\n', filename, err); 

fprintf(1, 'Symmetry Measure of Filtered Matrix ||f(A) - f(A)||_2 = %e\n', norm(filteredMat, 1));
flush_io(1); 

end
