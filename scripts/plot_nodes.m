function [] = plot_nodes(nb_inner_bnd, nb_outer_bnd, iter_max, iter_step)

if (nargin < 3)
    iter_max = 100; 
end
if (nargin < 4)
    iter_step = 20; 
end

for iter = 0:iter_step:iter_max

    filename = sprintf('boundary_nodes_%.5d_inner_%.5d_outer_%.5d_iter.txt', nb_inner_bnd, nb_outer_bnd, iter); 

    boundary_nodes = load(filename);

    fprintf('Loaded %d boundary nodes\n', length(boundary_nodes));

    %plotBoundary(boundary_nodes(1:nb_inner_bnd,:)); 
    plotBoundary(boundary_nodes); 
    title(filename);
    axis square;
    pbaspect([1 1 1]);            
    axis vis3d;
    drawnow
end

end

function [] = plotBoundary(pnodes)

    if (size(pnodes,2) > 2) 
       scatter3(pnodes(:,1),pnodes(:,2),pnodes(:,3),18,'filled');
       %tes = delaunay3(pnodes(:,1), pnodes(:,2), pnodes(:,3));
       %tetramesh(tes, pnodes);
    else 
        scatter(pnodes(:,1),pnodes(:,2),18,'filled');
    end

end