function [] = plotHeightfield(nodefilename, zfilename, titleprefix, testCaseName, showNodes, nbignore, ignoredNodeVal)
N = load(nodefilename); 
Z = load(zfilename); 

if (nargin < 5) 
    showNodes = 0;
end

if (nargin > 5) 
    %N = [zeros(nbignore,1);N(nbignore+1:end,:)]; 
    if(nargin > 5)
        Z(1:nbignore,1) = ignoredNodeVal;
        Z = Z(1:length(N(:,1)));
        
    else 
        Z = Z(nbignore+1:length(N(:,1)),1);
        length(Z)
    end
else 
     Z = Z(1:length(N(:,1)));
end

plotSurf(N(:,1), N(:,2), Z(:,1), showNodes);
set(0,'defaulttextinterpreter','none'); % DISABLE LATEX SUPPORT IN TITLE

if (nargin < 4)
    titlestr=sprintf('%s: %s', titleprefix, zfilename); 
    title(titlestr);
else 
    titlestr1=sprintf('[%s] ', testCaseName);
    titlestr2=sprintf('%s: %s', titleprefix, zfilename);
    title({titlestr1,titlestr2});
end
pbaspect([1 1 1]); % Square the plot box aspect ratio

end