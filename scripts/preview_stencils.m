function [] = preview_stencils(nodes, stencils)

if size(nodes, 2) < 2
   nodes = [nodes zeros(size(nodes,1),1)]; 
end

min_x = min(nodes(:,1)); 
max_x = max(nodes(:,1)); 

min_y = min(nodes(:,2)); 
max_y = max(nodes(:,2)); 

for j = 1:size(stencils, 1); 
    plot(nodes(:,1), nodes(:,2), '.'); 
    hold on;
    stencil = stencils(j,1:end);
    x_j = nodes(stencil(1),:);
    max_rad = 0;
    for i = 1:length(stencil)
        x_i = nodes(stencil(i), :);
        segment = [x_i; x_j];
        
        rad = sqrt((x_i - x_j) * (x_i - x_j)');
        if (max_rad < rad) 
            max_rad = rad;
        end
        plot(segment(:,1), segment(:,2), 'r-', 'LineWidth', 3); 
        plot(x_i(1), x_i(2),'o', 'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','y','MarkerSize',5); 
        plot(x_j(1), x_j(2),'s', 'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',7);
    end
    
    [c_x, c_y, c_z] = cylinder(max_rad, 200);
    % draw circe centered at stencil center
    plot(c_x(1,:)+x_j(1), c_y(1,:)+x_j(2), '--m');
    ti = sprintf('Stencil %d', j); 
    title(ti);
    axis square;
    axis([min_x max_x min_y max_y]);
    
    hold off;
    % sleep until a key is pressed
    pause
    %pause(0.25)  % sleep 0.5 seconds to show stencil
end

end