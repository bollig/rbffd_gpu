function [] = plot_nodes(nb_inner_bnd, nb_outer_bnd, nb_interior, iter_max_bnd, iter_max_int, iter_step, dim)
% PLOT_NODES
% PLOT_NODES(nb_inner_bnd, nb_outer_bnd, nb_interior, iter_max_bnd, iter_max_int, iter_step, dim)
    if (nargin < 4)
        iter_max_bnd = 100;
    end
    if (nargin < 5)
        iter_max_int = 100;
    end
    if (nargin < 6)
        iter_step = 20;
    end
    if (nargin < 7)
        dim = 2;
    end

% 
%     for iter = 0:iter_step:iter_max_bnd
% 
%         figure(1);
%         filename = sprintf('boundary_nodes_%.5d_inner_%.5d_outer_%.5d_iter_%dD.ascii', nb_inner_bnd, nb_outer_bnd, iter, dim);
% 
%         boundary_nodes = load(filename);
% 
%         fprintf('Loaded %d boundary nodes\n', length(boundary_nodes));
% 
%         %plotBoundary(boundary_nodes(1:nb_inner_bnd,:));
%         %plotBoundary(boundary_nodes((boundary_nodes(:,1) > 0 & boundary_nodes(:,2) > 0), :));
%         plotBoundarySurface(boundary_nodes);
%         title(filename);
%         axis square;
%         %axis([-1 1 -1 1 -1 1]);
%         pbaspect([1 1 1]);
%         axis vis3d;
%         drawnow
%         pause
%     end

    boundary_nodes = []; 


    for iter = 0:iter_step:iter_max_int

        figure(2);
        filename = sprintf('interior_nodes_%.5d_interior_%.5d_iter_%dD.ascii', nb_interior, iter, dim);
        %filename = sprintf('interior_nodes_%.5d_interior_%.5d_iter_%dD.ascii', nb_interior, iter, dim);

        interior_nodes = load(filename);

        fprintf('Loaded %d interior nodes\n', length(interior_nodes));

        %plotInterior(interior_nodes(1:nb_inner_bnd,:));
        %plotInterior(interior_nodes((interior_nodes(:,1) > 0 & interior_nodes(:,2) > 0), :));
        plotInteriorScatter(boundary_nodes, interior_nodes);
        title(filename);
        axis square;
        %axis([-1 1 -1 1 -1 1]);
        pbaspect([1 1 1]);
        axis vis3d;
        drawnow
        pause
    end

end

function [data] = load_file(filename, numElements)

    if (filename(length(filename)-3 : length(filename)) == '.bin')
        data = load_binary_file(filename, numElements);
    else 
        data = load_ascii_file(filename, numElements);
    end

end

function [data] = load_ascii_file(filename, dim_num, numElements)
    %read the file back in
    fid=fopen(filename, 'rt');   %open the file
    title = fgetl(fid);  %read in the header
    if (title == 'ASCII')
        % skim off meta data comments
        
        % read data 
        [data,count]=fscanf(fid, '%f',[3,3]);   %read in data
    end
    fclose(fid);   %close the file
end

function [data] = load_binary_file(filename, dim_num, numElements)



    file = fopen(filename, 'r');
    fid = fopen('square_mat.bin','rb')   %open file
    bintitle = fread(fid, 23, 'char');   %read in the header
    title = char(bintitle')   
    data = fread(fid, [3 inf], 'int32')  %read in the data
    data_tranpose = data'   %must transpose data after reading in
    fclose(fid)   %close file
end


function [] = plotBoundaryScatter(pnodes)

    if (size(pnodes,2) > 2)
        scatter3(pnodes(:,1),pnodes(:,2),pnodes(:,3),18,'filled');
    else
        scatter(pnodes(:,1),pnodes(:,2),18,'filled');
        %[VX,VY] = voronoi(pnodes(:,1), pnodes(:,2));
        %plot(VX, VY, '-', pnodes(:,1), pnodes(:,2), '.');
    end

end

function [] = plotBoundarySurface(pnodes)

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


function [trimmed] = trim_nodes(bnodes)
    % Trim to single quad
    trimmed = bnodes((bnodes(:,1) > 0 & bnodes(:,2) > 0 & bnodes(:,3) > 0), :);
end
