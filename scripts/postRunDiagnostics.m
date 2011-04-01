% NOTE: MatrixMarketFilename is the name of the MatrixMarket file containing our 
% large LHS matrix for the implicit system
function [] = postRunDiagnostics(MatrixMarketFilename, testCaseName)

if (nargin < 2) 
    testCaseName = 'Unknown Case';
end
    
% It is assumed mmread is on the path (available in RBF.framework/trunk/scripts)
[A, rows, cols, entries] = mmread(MatrixMarketFilename);

checkSymmetry(A, MatrixMarketFilename, testCaseName); 
checkReordering(A, MatrixMarketFilename, 0);
%checkEigenvalues(A, MatrixMarketFilename, testCaseName);

fprintf(1, 'end\n');
end
