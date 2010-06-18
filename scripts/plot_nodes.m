function [] = plot_nodes(nb_inner_bnd, nb_outer_bnd, nb_interior, iter_max_bnd, iter_max_int, iter_step)

if (nargin < 4)
    iter_max_bnd = 100; 
end
if (nargin < 5)
    iter_max_bnd = 100; 
end
if (nargin < 6)
    iter_step = 20; 
end

for iter = 0:iter_step:iter_max_bnd

    figure(1);
    filename = sprintf('boundary_nodes_%.5d_inner_%.5d_outer_%.5d_iter.txt', nb_inner_bnd, nb_outer_bnd, iter); 

    boundary_nodes = load(filename);

    fprintf('Loaded %d interior nodes\n', length(boundary_nodes));

    %plotBoundary(boundary_nodes(1:nb_inner_bnd,:)); 
    %plotBoundary(boundary_nodes((boundary_nodes(:,1) > 0 & boundary_nodes(:,2) > 0), :));
    plotBoundary(boundary_nodes); 
    title(filename);
    axis square;
    %axis([-1 1 -1 1 -1 1]);
    pbaspect([1 1 1]);            
    axis vis3d;
    drawnow
    pause
end



for iter = 0:iter_step:iter_max_int

    figure(2);
    filename = sprintf('interior_nodes_%.5d_interior_%.5d_iter.txt', nb_interior, iter); 

    interior_nodes = load(filename);

    fprintf('Loaded %d interior nodes\n', length(interior_nodes));

    %plotInterior(interior_nodes(1:nb_inner_bnd,:)); 
    %plotInterior(interior_nodes((interior_nodes(:,1) > 0 & interior_nodes(:,2) > 0), :));
    plotInterior(boundary_nodes, interior_nodes); 
    title(filename);
    axis square;
    %axis([-1 1 -1 1 -1 1]);
    pbaspect([1 1 1]);            
    axis vis3d;
    drawnow
    pause
end

end

function [] = plotBoundary(pnodes)

    if (size(pnodes,2) > 2) 
       %scatter3(pnodes(:,1),pnodes(:,2),pnodes(:,3),18,'filled');
       tes = delaunay3(pnodes(:,1), pnodes(:,2), pnodes(:,3));
       tetramesh(tes, pnodes);
    else 
        scatter(pnodes(:,1),pnodes(:,2),18,'filled');
        %[VX,VY] = voronoi(pnodes(:,1), pnodes(:,2)); 
        %plot(VX, VY, '-', pnodes(:,1), pnodes(:,2), '.');
    end

end

function [] = plotInterior(bnodes, pnodes)

    if (size(pnodes,2) > 2) 
       scatter3(pnodes(:,1),pnodes(:,2),pnodes(:,3),18,1:length(pnodes),'filled');
       hold on;
       scatter3(bnodes(:,1),bnodes(:,2),bnodes(:,3),18,'filled');
       %tes = delaunay3(bnodes(:,1), bnodes(:,2), bnodes(:,3));
       %tetramesh(tes, bnodes);
       hold off; 
    else 
        scatter(bnodes(:,1),bnodes(:,2),18,'filled');
        hold on; 
        scatter(pnodes(:,1),pnodes(:,2),18,1:length(pnodes),'filled');
        hold off; 
        %[VX,VY] = voronoi(pnodes(:,1), pnodes(:,2)); 
        %plot(VX, VY, '-', pnodes(:,1), pnodes(:,2), '.');
    end

end


function [trimmed] = trim_nodes(bnodes)
    % Trim to single quad
    trimmed = bnodes((bnodes(:,1) > 0 & bnodes(:,2) > 0 & bnodes(:,3) > 0), :);
end