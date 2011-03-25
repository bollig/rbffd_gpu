function [] = plotRelativeError(nodefilename, abs_err_file, exact_sol_file, titleprefix, testCaseName, showNodes, nbignore, ignoredNodeVal)
N = load(nodefilename); 
abserr = load(abs_err_file); 
exact = load(exact_sol_file);

if (nargin < 6) 
    showNodes = 0;
end

if (nargin > 6) 
    %N = [zeros(nbignore,1);N(nbignore+1:end,:)]; 
    if(nargin > 7)
        Z(1:nbignore,1) = ignoredNodeVal;
        Z = abs(abserr(1:length(N(:,1))))./abs(exact(1:length(N(:,1))));
        
    else 
        Z = abs(abserr(nbignore+1:length(N(:,1)),1)) ./ abs(exact(nbignore+1:length(N(:,1)),1));
        length(Z)
    end
else 
     Z = abs(abserr(1:length(N(:,1))))./abs(exact(1:length(N(:,1))));
end
Z(isnan(Z)) = -1;

plotSurf(N(:,1), N(:,2), Z(:,1), showNodes);
set(0,'defaulttextinterpreter','none'); % DISABLE LATEX SUPPORT IN TITLE

if (nargin < 5)
    titlestr=sprintf('%s: Relative Error', titleprefix); 
    title(titlestr);
else 
    titlestr1=sprintf('[%s] ', testCaseName);
    titlestr2=sprintf('%s: Relative Error', titleprefix);
    title({titlestr1,titlestr2});
end
pbaspect([1 1 1]); % Square the plot box aspect ratio

end