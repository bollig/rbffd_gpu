function [] = plotNodes(nodefilename)

N = load(nodefilename); 

figure; 
set(0,'defaulttextinterpreter','none'); % DISABLE LATEX SUPPORT IN TITLE
scatter(N(:,1), N(:,2), 'filled')
titlestr=sprintf('Node List: %s', nodefilename); 
title(titlestr);
pbaspect([1 1 1]); % Square the plot box aspect ratio

end