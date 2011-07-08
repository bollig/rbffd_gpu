function [] = preview_stencils(nodes, stencils, aspect_ratio)

dim = 2; 
if size(nodes, 2) < 2
   nodes = [nodes zeros(size(nodes,1),1)]; 
   dim = 2;
end

if nargin < 3
    aspect_ratio = [1 1 1]; 
end

min_x = min(nodes(:,1)); 
max_x = max(nodes(:,1)); 

min_y = min(nodes(:,2)); 
max_y = max(nodes(:,2)); 

min_z = min(nodes(:,3)); 
max_z = max(nodes(:,3));

if max_z - min_z > 0
   dim = 3;
end


j = 1;

f = figure; 
set(f,'WindowKeyPressFcn',@scrollstencils)
plot_stencils(1);

function scrollstencils(src, evt)

%for j = 1:size(stencils, 1); 
    if strcmp(evt.Key,'leftarrow') || strcmp(evt.Key,'downarrow')
        j = j-1; 
        if (j < 1)
            j = size(stencils, 1);
        end
    elseif strcmp(evt.Key,'rightarrow') || strcmp(evt.Key,'uparrow')
        % Do not exceed bounds:
        j = j+1; 
        if j > size(stencils,1)
            j = 1; 
        end
    end
    plot_stencils(j); 
end 

function plot_stencils(j)
    if (dim < 3)
        plot(nodes(:,1), nodes(:,2), '.','MarkerSize',5); 
    else
        plot3(nodes(:,1), nodes(:,2), nodes(:,3), '.', 'MarkerSize', 5); 
    end
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
        if (dim < 3) 
            plot(segment(:,1), segment(:,2), 'r-', 'LineWidth', 2);
            plot(x_i(1), x_i(2),'o', 'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','y','MarkerSize',6);
            plot(x_j(1), x_j(2),'s', 'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',8);
        else 
            plot3(segment(:,1), segment(:,2), segment(:,3), 'r-', 'LineWidth', 2);
            plot3(x_i(1), x_i(2), x_i(3),'o', 'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','y','MarkerSize',6);
            plot3(x_j(1), x_j(2), x_j(3),'s', 'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',8);
        end 
    end
    
    if (dim < 3)
        [c_x, c_y, c_z] = cylinder(max_rad, 200);
        % draw circe centered at stencil center
        plot(c_x(1,:)+x_j(1), c_y(1,:)+x_j(2), '--m');
        ti = sprintf('Stencil %d', j);
        title(ti);
        axis square;
        axis([min_x max_x min_y max_y]);
        pbaspect(aspect_ratio)
    else
        Tes = delaunay3(nodes(:,1),nodes(:,2),nodes(:,3));
        %X = [x(:) y(:) z(:)];
        hB = tetramesh(Tes,nodes);
        bcol =[250 250 0]/256;
        alpha = 0.35; 
        set(hB,'facecolor',bcol,'facealpha', alpha, 'edgelighting','phong','facelighting','phong','LineStyle','none','marker','.','markeredgecolor','b','markersize',15);

        [c_x,c_y,c_z] = sphere(20);
        c_x = max_rad * c_x + x_j(1); 
        c_y = max_rad * c_y + x_j(2); 
        c_z = max_rad * c_z + x_j(3); 
        sp1 = surf(c_x, c_y, c_z);
        %alpha(0.5);
        alpha = 0.2;
        set(sp1,'EdgeColor',[0.5 0.5 0.5], 'EdgeAlpha', alpha,... 
            'FaceColor','m', 'FaceLighting','phong',... 
            'FaceAlpha',alpha/2); 
        ti = sprintf('Stencil %d', j);
        title(ti);
        axis square;
        axis([min_x max_x min_y max_y min_z max_z]);
        pbaspect(aspect_ratio);
    end
    
    hold off;
    % sleep until a key is pressed
%end
end
end