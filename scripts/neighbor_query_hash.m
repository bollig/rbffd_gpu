function [stencils] = neighbor_query_hash(node_list, max_st_size, hnx)


% Dimensions of the hash overlay grid (hnx by hny by hnz regular grid
% spanning the full bounding box of the domain extent)
% NOTE: it works best for nearest neighbor if we have hnx==hny==hnz
% if nargin < 5
%     hnz = 4;
%     if nargin < 4
%         hny = 4;
%         if nargin < 3
%             hnx = 4;
%         end
%     end
% end
xmin = min(node_list(:,1));
xmax = max(node_list(:,1));

if (size(node_list, 2) > 1)
    ymin = min(node_list(:,2));
    ymax = max(node_list(:,2));
else 
    ymin = 0; 
    ymax = 0;
end

if (size(node_list, 2) > 2)
    zmin = min(node_list(:,3));
    zmax = max(node_list(:,3));
else 
    zmin = 0; 
    zmax = 0;
end
if (ymax-ymin) > 1e-8
    hny = hnx;
else 
    hny = 1;
end
if (zmax-zmin) > 1e-8
    hnz = hnx
else 
    hnz = 1;
end


nb_nodes = size(node_list,1);

stencils = zeros(nb_nodes, max_st_size);%+1);

cell_hash = zeros(hnx * hny * hnz, 1);
% list of lists
%' cell_hash.resize(hnx * hny * hnz);
cell_id_end = zeros(hnx * hny * hnz, 1);

cdx = (xmax - xmin) / hnx;
cdy = (ymax - ymin) / hny;
cdz = (zmax - zmin) / hnz;

% Foreach node:
%     determine hashid (cellid)
%          node(x,y,z) exists in cellid((x-xmin)/dx, (y-ymin)/dy, (z-zmin)/dz)
%          linearize cellid(xc, yc, zc) = ((xc*NY) + yc)*NZ + zc
%          append node to list contained in list[cellid]
for i = 1 : nb_nodes
    node = node_list(i,:);
    % xc, yc and zc are the (x,y,z) corresponding to the cell id
    % xmin,ymin,zmin are member properties of the Grid class
    % cdx,cdy,cdz are the deltaX, deltaY, deltaZ for the cell overlays
    % TODO: we note that the xc, yc and zc can be treated at binary digits
    % to select the CELL_ID (do we really need an optimization like that
    % though?)
    xc = floor((node(1) - xmin) / cdx);
    
    % This logic saves us when our nodes lie on xmax, ymax, or zmax
    % so instead of covering [n-1*dx,xmax), our cell covers [n-1*dx,xmax]
    %
    if (xc == hnx)
        xc = xc-1;
    end
    
    if (cdy > 0) 
        yc = floor((node(2) - ymin) / cdy);
        if (yc == hny)
            yc = yc-1;
        end
    else
        yc = 0;
    end
    if (cdz > 0) 
        zc = floor((node(3) - zmin) / cdz);
        if (zc == hnz)
            zc = zc-1;
        end
    else
        zc = 0;
    end
    
    % KEY: this is how we get our index for the 3D overlay grid cell
    % ZERO based cell_id (we adjust by adding 1);
    cell_id = ((xc*hny) + yc)*hnz + zc + 1;
    
    %fprintf('NODE: %f %f %f is in CELL: %d\n', node(1), node(2), node(3), cell_id);
    
    % Push back node index on cell hash
    cell_id_end(cell_id) = cell_id_end(cell_id) + 1;
    cell_hash(cell_id,cell_id_end(cell_id)) = i;
end

%fprintf('Nodes hashed, ready to query neighbors');
% TODO: Sort nodes according to hash for better access patterns

% Foreach node:
%      Generate a stencil:
%          append cell_hash[cellid(this->node)] list to candidate list
%          if (stencil_size > cell_hash.length) then
%              append 8 (or 26 if 3D) neigboring cell_hash lists to candidate list
%          end
%          sort the candidate list according to distance from node
%          select stencil_size closest matches
for  p = 1:nb_nodes
    node = node_list(p,:);
    % xc, yc and zc are the (x,y,z) corresponding to the cell id
    % xmin,ymin,zmin are member properties of the Grid class
    % cdx,cdy,cdz are the deltaX, deltaY, deltaZ for the cell overlays
    % TODO: we note that the xc, yc and zc can be treated at binary digits
    % to select the CELL_ID (do we really need an optimization like that
    % though?)
    xc = floor((node(1) - xmin) / cdx);
    
    % This logic saves us when our nodes lie on xmax, ymax, or zmax
    % so instead of covering [n-1*dx,xmax), our cell covers [n-1*dx,xmax]
    %
    if (xc == hnx)
        xc = xc-1;
    end
    
    if (cdy > 0) 
        yc = floor((node(2) - ymin) / cdy);
        if (yc == hny)
            yc = yc-1;
        end
    else
        yc = 0;
    end
    if (cdz > 0) 
        zc = floor((node(3) - zmin) / cdz);
        if (zc == hnz)
            zc = zc-1;
        end
    else
        zc = 0;
    end
    
    % KEY: this is how we get our index for the 3D overlay grid cell
    % ZERO based cell_id (we adjust by adding 1);
    node_cell_id = ((xc*hny) + yc)*hnz + zc + 1;
    
    % List of cell indices we will check
    % NOTE: in C++ we leverage std::set<size_t> here because it does NOT allow duplicates,
    % so cells are not searched twice. In Matlab we need to use 'unique' to
    % remove duplicates:
    %       [junk,index] = unique(y,'first');        %# Capture the index, ignore junk
    %        y = y(sort(index))
    % In reality, this can be further optimized by NOT appending
    % previous cells. However, this step does not cost a lot of
    % overhead for current applications.
    neighbor_cell_set = [];
    
    % Generate a list of cells to check for nearest neighbors
    % For each node expand the search until the max_st_size can be satisifed
    % DO NOT check cells with 0 node inside
    nb_neighbor_nodes_to_check = 0;
    level = 0;
    
    % TODO: cut-off search if (max_st_radius+cdx) is execeeded
    %          (requires a working impl of max_st_radius)o
    % BUGFIX: this (level < 2) guarantees we searching neighboring cells
    % in the event that we're near the boundary of a cell and the current
    % cell has more than enough nodes to exceed max_st_size.
    while (nb_neighbor_nodes_to_check < max_st_size) || (level < 2)
        level_neighbor_set = [];
        xlevel = level;
        ylevel = 0;
        zlevel = 0;
        if hny > 1
            ylevel = level;
        end
        if hnz > 1
            zlevel = level;
        end
        
        % Now count the number of nodes we'll be checking.
        % If its greater than max_st_size then we can stop expanding
        % search
        nb_neighbor_nodes_to_check = 0;
        
        %NOTE: might need a +1 here:
        for xindx = 0-xlevel : 0+xlevel
            for yindx = 0-ylevel : 0+ylevel
                for zindx = 0-zlevel : 0+zlevel
                    % Offset cell
                    xc_o = (xc + xindx);
                    yc_o = (yc + yindx);
                    zc_o = (zc + zindx);
                    
                    % If the neighbor cell is outside our overlay we ignore the task
                    if ((xc_o < 0) || (xc_o >= hnx))
                        continue;
                    end
                    
                    if ((yc_o < 0) || (yc_o >= hny))
                        continue;
                    end
                    
                    if ((zc_o < 0) || (zc_o >= hny))
                        continue;
                    end
                    
                    cell_id = ((xc_o*hny) + yc_o)*hnz + zc_o + 1;
                    
                    % only bother appending neighboring cells that contain
                    % nodes.
                    % gets all node ids contained in the cell
                    l = sum(cell_hash(cell_id,:) > 0);
                    if (l > 0)
                        %neighbor_cell_set.insert(cell_id);
                        level_neighbor_set(end+1) = cell_id;
                        nb_neighbor_nodes_to_check = nb_neighbor_nodes_to_check + l;
                    end
                end
            end
        end
        
% 
%         for it = 1:length(level_neighbor_set)
%             cell_id = level_neighbor_set(it);
%             nb_neighbor_nodes_to_check = nb_neighbor_nodes_to_check + length(cell_hash(cell_id,:));
%         end
        
        % Increase our search radius
        level = level + 1;
    end
    
    % This removes duplicates and keeps the id's ordered in the fashion
    % they were first appended to the list (helps minimize shuffling of
    % nearest neighbor indices)
    [junk,srt_index] = unique(level_neighbor_set,'first');        %# Capture the index, ignore junk
    %fprintf('NODE: %f %f %f is in CELL: %d and Querying %d distances in these CELLs:\n', node(1), node(2), node(3), node_cell_id, nb_neighbor_nodes_to_check);
    neighbor_cell_set = level_neighbor_set(sort(srt_index));
    
    % Compute distances for each neighbor and insert them into a sorted set.
    dists = [];
    neighbor_indx = [];
    d_count = 0;
    for it = 1:length(neighbor_cell_set)
        cell_id = neighbor_cell_set(it);
        for q = 1:length(cell_hash(cell_id,:))
            nid = cell_hash(cell_id,q);
            % Matlab appends 0's to all columns as we expand a list
            if nid == 0
                continue; 
            end
            neighbor = node_list(nid,:);
            sep =(node - neighbor);
            dist = sqrt(sep*sep');
            dists(end+1) = dist;
            neighbor_indx(end+1) = nid;
            d_count = d_count+1;
        end
    end
    [srted_dists, dsrt_indx] = sort(dists);
   
    % Finally, lets keep only the first max_st_size indices as our stencil
%    stencils(p,1) = max_st_size;
%    stencils(p,2:end) = neighbor_indx(dsrt_indx(1:max_st_size));
    stencils(p,1:end) = neighbor_indx(dsrt_indx(1:max_st_size));
    
end

