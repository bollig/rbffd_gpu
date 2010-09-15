function [] = plotCVTNodes(file_prefix, nb_inner, nb_outer, nb_interior, dim)
    filename = sprintf('%s_%.5d_inner_%.5d_outer_%.5d_interior_final_%dD.ascii', file_prefix, nb_inner, nb_outer, nb_interior, dim);
    
    all_nodes = load(filename);
    
    boundary_nodes = all_nodes(1:(nb_inner + nb_outer),:);
    interior_nodes = all_nodes((nb_inner + nb_outer+1):end,:);
    

        %plotInterior(interior_nodes(1:nb_inner_bnd,:));
        %plotInterior(interior_nodes((interior_nodes(:,1) > 0 & interior_nodes(:,2) > 0), :));
    plotInteriorScatter(boundary_nodes, interior_nodes);
        title(filename);
        axis square;
        %axis([-1 1 -1 1 -1 1]);
        pbaspect([1 1 1]);
        axis vis3d;
        drawnow

end

function [] = plotInteriorScatter(bnodes, pnodes)

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
        %[VX,VY] = voronoi([bnodes(:,1);pnodes(:,1)], [bnodes(:,2);pnodes(:,2)]);
       % plot(VX, VY, '-');%, pnodes(:,1), pnodes(:,2), '.');
        hold off;
        axis([-1 1 -1 1]);
    end

end