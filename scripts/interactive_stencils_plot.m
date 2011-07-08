function [] = interactive_stencils_plot(nodefilename, st_size)

st_filename = sprintf('stencils_maxsz%d_%s', st_size, nodefilename);
stencils = load(st_filename);
stencils = stencils(:,2:end) + 1; 
nodes = load(nodefilename);

preview_stencils(nodes, stencils); 
% 
% for j = 1:size(stencils, 1); 
%     plot(nodes(:,1), nodes(:,2), '.'); 
%     hold on;
%     stencil = stencils(j,2:end) + 1;
%     x_j = nodes(stencil(1),:);
%     for i = 1:length(stencil)
%         x_i = nodes(stencil(i), :);
%         segment = [x_i; x_j];
%         plot(segment(:,1), segment(:,2), 'r-', 'LineWidth', 5); 
%     end
%     axis square;
%     hold off;
%     pause(0.25)  % sleep 0.5 seconds to show stencil
% end
       


end