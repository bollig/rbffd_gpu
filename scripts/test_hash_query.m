function [stencilDiff] = test_hash_query() 
desired = load('stencils_maxsz13_ellipse_cvt_300generators_uniform_density_2d_final.ascii');
adjusted = [desired(:,1) desired(:,2:end)+1];

nodes = load('ellipse_cvt_300generators_uniform_density_2d_final.ascii');
computed = neighbor_query_hash(nodes, 13,10);
% stencilDiff = zeros(300, 14);
% 
%     for i = 1:300
%          sortedStencilA = sort(adjusted(i,:),); 
%          sortedStencilB = sort(computed(i,:)); 
%          
%          stencilDiff(i,:) = sortedStencilA - sortedStencilB; 
%     end

stencilDiff = sort(adjusted, 2) - sort(computed, 2);

stencils = [computed(:,1) computed(:,2:end)-1];

for j = 1:size(stencils, 1); 
    plot(nodes(:,1), nodes(:,2), '.'); 
    hold on;
    stencil = stencils(j,2:end) + 1;
    x_j = nodes(stencil(1),:);
    for i = 1:length(stencil)
        x_i = nodes(stencil(i), :);
        segment = [x_i; x_j];
        plot(segment(:,1), segment(:,2), 'r-', 'LineWidth', 5); 
    end
    axis square;
    hold off;
    pause(0.25)  % sleep 0.5 seconds to show stencil
end
       

end