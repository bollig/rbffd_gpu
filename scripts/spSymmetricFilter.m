% Mask the matrix and return sparse representation of only elements which 
% do not have a weight in the corresponding transpose element. 
function [filteredMat] = spSymmetricFilter(SparseMatA) 

% Set all non-zero elements of A to 1
C = spfun(@spmask, SparseMatA);

% Find elements in the matrix which are not mirrored on transpose
filteredMat = (C - C');

% Now D has 0 or (+/-)1
% if < 0 then edge is not part of C. 
ind = find( filteredMat < 0 );
filteredMat(ind) = 0; 

end
