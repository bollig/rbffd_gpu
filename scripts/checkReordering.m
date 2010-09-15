function [NewA] = checkReordering(SparseMatA, filename, boundaryType)

DIRICHLET = 0;
NEUMAN = 1;
ROBIN = 2;


% Direct reordering without trimming the matrix
symrcmNoTrim = symrcm(SparseMatA); 

h = figure; 
spy(SparseMatA(symrcmNoTrim, symrcmNoTrim));
label = sprintf('Symmetric Reverse Cuthill McKee Reordering for %s', filename); 
title(label); 
% Backup to disk
print('symrcmNoTrim.png', '-dpng', '-r300');

 
fid = fopen('symrcmNoTrim_permutationVector.mtx', 'wt');
fprintf(fid, '%d\n', symrcmNoTrim);
fclose(fid);


% Trim off the row and col of 1's (we can put them back in at any time)
B = SparseMatA(1:(end-1), 1:(end-1)); 
[m n] = size(SparseMatA); 
% We add in the last index so our permutation vector accounts for all
% rows/cols of the matrix. 
symrcmTrimmed = [symrcm(B), m]; 

h = figure; 
spy(SparseMatA(symrcmTrimmed, symrcmTrimmed));
label = sprintf('Symmetric Reverse Cuthill McKee Reordering for %s', filename); 
title(label); 
% Backup to disk
print('symrcmTrimmed.png', '-dpng', '-r300');
  
fid = fopen('symrcmTrimmed_permutationVector.mtx', 'wt');
fprintf(fid, '%d\n', symrcmTrimmed);
fclose(fid);

newName = sprintf('Original %s', filename); 
printBandwidth(SparseMatA, newName); 

newName = sprintf('Original %s (ignoring row/cols of 1s)', filename); 
printBandwidth(SparseMatA(1:(end-1), 1:(end-1)), newName);

newName = sprintf('SYMRCM (No Trim) %s', filename); 
printBandwidth(SparseMatA(symrcmNoTrim, symrcmNoTrim), newName);

newName = sprintf('SYMRCM (Trimmed) %s', filename); 
printBandwidth(SparseMatA(symrcmTrimmed, symrcmTrimmed), newName);

newName = sprintf('SYMRCM (Trimmed) %s (ignoring row/col of 1s)', filename); 
Atrimmed = SparseMatA(symrcmTrimmed, symrcmTrimmed);
printBandwidth(Atrimmed(1:(end-1), 1:(end-1)), newName);


NewA = SparseMatA(symrcmTrimmed, symrcmTrimmed); 

end