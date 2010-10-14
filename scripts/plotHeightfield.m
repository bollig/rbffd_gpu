function [] = plotHeightfield(nodefilename, zfilename, titleprefix, testCaseName)
N = load(nodefilename); 
Z = load(zfilename); 

plotSurf(N(:,1), N(:,2), Z(1:length(N(:,1)),1));
set(0,'defaulttextinterpreter','none'); % DISABLE LATEX SUPPORT IN TITLE

if (nargin < 4)
    titlestr=sprintf('%s: %s', titleprefix, zfilename); 
else 
    titlestr=sprintf('[%s] %s: %s', testCaseName, titleprefix, zfilename);
end
title(titlestr);
pbaspect([1 1 1]); % Square the plot box aspect ratio

end