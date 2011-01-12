function [] = plotNodes(nodefilename, restrictTo2D)

if nargin < 2
    restrictTo2D = false;
end

N = load(nodefilename); 

figure; 
set(0,'defaulttextinterpreter','none'); % DISABLE LATEX SUPPORT IN TITLE

if (restrictTo2D)
    scatter(N(:,1), N(:,2), 'filled');
else 
    scatter3(N(:,1), N(:,2), N(:,3), 'filled');
end
titlestr=sprintf('Node List: %s', nodefilename); 
title(titlestr);
pbaspect([1 1 1]); % Square the plot box aspect ratio

end