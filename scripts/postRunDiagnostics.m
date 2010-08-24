% NOTE: MatrixMarketFilename is the name of the MatrixMarket file containing our 
% large LHS matrix for the implicit system
function [] = postRunDiagnostics(MatrixMarketFilename)

% It is assumed mmread is on the path (available in RBF.framework/trunk/scripts)
[A, rows, cols, entries] = mmread(MatrixMarketFilename);

checkSymmetry(A); 

checkEigenvalues(A, MatrixMarketFilename);

fprintf(1, 'end\n');
end
