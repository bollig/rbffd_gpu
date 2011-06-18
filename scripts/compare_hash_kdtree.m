function [speedup] = compare_kdtree_hash()

testN = [10^2, 50^2, 100^2, 200^2 300^2 ];
testk = [ 13] ;%, 13, 27 ];
%testSubs = [ 16, 32, 64, 100, 128, 200, 256, 300, 384, 512, 1024  ]

for i = 1:length(testN)
    %fprintf('Generate node list: ');
    tic;
    nodes = [rand( testN(i),  2 ) zeros(testN(i), 1)]; %[linspace(0, 1000, N)' zeros(N, 2)];
    t_nodes = toc;
    
    testSubs = [ floor(sqrt(testN(i)) / 2) ];
    
    for j = 1:length(testk)
        %fprintf('Mex Compiled KDTree: ');
        tic;
        % KDTree is mex compiled from
        % https://sites.google.com/site/andreatagliasacchi/software/matlabk
        % d-treelibrary
        tree = kdtree_build( nodes );
        stencils = zeros(testN(i), testk(j));
        for s = 1:size(nodes, 1)
            stencils(s,:) = kdtree_k_nearest_neighbors( tree, nodes(s,:), testk(j));
        end
        t_kdtree = toc;
        
        for k = 1:length(testSubs)
            % squared because we're in 2D and we have subs_x == subs_y
            if (testSubs(k)^2 > testN(i))
                fprintf('Skipping: %d Nodes, %d Neighbors, %d Subdivisions (in Hash) [N < Subs^2]\n', testN(i), testk(j), testSubs(k));
                speedup(i,j,k) = 0;
            elseif (testk(j)*2*(testSubs(k)^2) < testN(i))
                fprintf('Skipping: %d Nodes, %d Neighbors, %d Subdivisions (in Hash) [N > neighbors*2*Subs^2]\n', testN(i), testk(j), testSubs(k));
                speedup(i,j,k) = 0;
            else
                fprintf('%d Nodes, %d Neighbors, %d Subdivisions (in Hash) =',testN(i), testk(j), testSubs(k));
                %fprintf('My Method: ');
                tic;
                hash_stencils = neighbor_query_hash(nodes, testk(j), testSubs(k));
                t_hash = toc;
                
                speedup(i,j,k) = t_kdtree / t_hash;
                fprintf('%fX (speedup)\n', speedup(i,j,k));
            end
        end
    end
end

fprintf('Final Summary:');

end
