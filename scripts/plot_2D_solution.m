function [] = plot_2D_solution(nbinner, nbouter, nbinterior, testCaseName, saveFigs, isTimeDep)

    nbnodes = nbinner + nbouter + nbinterior;

   %nodefile =
   %sprintf('nested_spheres_%0.5d_inner_%0.5d_outer_%0.5d_interior_final_2D.ascii', nbinner, nbouter, nbinterior);
   nodefile = sprintf('nested_sphere_cvt_%d_inner_%d_outer_%d_interior_2d_final.ascii', nbinner, nbouter, nbinterior);
   % nodefile = sprintf('nested_sphere_cvt_%dgenerators_2d_final.ascii', nbnodes);

    if (nargin < 5) 
        saveFigs = 1;
    end
    
    if (nargin < 6) 
       plot_2D_steady_solution(nodefile, testCaseName, nbinner+nbouter, nbinterior, saveFigs); 
    else 
        plot_2D_timedep_solution(nodefile, testCaseName);
    end
end

function plot_2D_steady_solution(nodefile, testCaseName, nbboundary, nbinterior, saveFigs)

    forceShowNodes = 1;

    % Get the number of figures before our efforts
    existingfigs = findobj('Type', 'figure');
    nefigs = length(existingfigs);

    % Do all our work
    plotNodes(nodefile); 
    plotHeightfield(nodefile, 'F.mtx', 'Implicit System RHS', testCaseName); 
    plotHeightfield(nodefile, 'X_exact.mtx','Exact Solution', testCaseName);
    plotHeightfield(nodefile, 'X_approx.mtx', 'Approximate Solution', testCaseName);
    plotHeightfield(nodefile, 'E_absolute.mtx', 'Absolute Error', testCaseName);
    plotHeightfield(nodefile, 'E_relative.mtx', 'Relative Error', testCaseName, forceShowNodes);
    plotRelativeError(nodefile, 'E_absolute.mtx', 'X_exact.mtx', 'Relative Error', testCaseName, forceShowNodes);
    % Plot heightfield, but specify that the boundary nodes should be
    % zeroed
    %plotHeightfield(nodefile, 'E_relative.mtx', 'Relative Error (Boundary Equals 0)', testCaseName, showNodes, nbboundary, 0); 
if 1
   % plotHeightfield(nodefile, 'E_relative.mtx', 'Relative Error', testCaseName, forceShowNodes);
    % Plot heightfield, but specify that the boundary nodes should be
    % excluded
    %plotHeightfield(nodefile, 'E_relative.mtx', 'Relative Error (Interior Nodes Only)', testCaseName, forceShowNodes, nbboundary); 
    postRunDiagnostics('L_host.mtx', testCaseName);

    figure; 
    solverResidual = load('BICGSTAB_RESIDUAL.mtx');
    % Plot (ri / r0) and see if it improves
    R = solverResidual ./ solverResidual(1); 
    semilogy(R,'-')
    xlabel('iteration number')
    ylabel('relative residual in solver (log(r_i/r_0))')
    label1 = sprintf('[%s] ', testCaseName); 
    title({label1, 'BICGSTAB Residual Ratio'});
end
    
    % Get the number of figures after our efforts
    allfigs = findobj('Type', 'figure'); 
    nfigs = length(allfigs);
    
    % Only sort the figures we created (presumably no other scripts were
    % run which created windows during this execution)
    tilefigs(nefigs+1:nfigs);
    
    % Now that we are displaying all figures on screen, we save them to
    % file
    if (saveFigs)
        savefigs(nefigs+1:nfigs);
    end
end


function plot_2D_timedep_solution()

    nodes = load('nested_spheres_00020_inner_00040_outer_00240_interior_final_2D.ascii')

    h = figure; 
    scatter(nodes(:,1), nodes(:,2), '.');
    axis square; 
    title('Node distribution'); 

    timestep = 0.01;

    mfig = figure;
    winsize = get(mfig,'Position'); 
    winsize(1:2) = [0 0]; 
    numframes = 1/timestep; 

    A = moviein(numframes, mfig, winsize); 
    set(mfig, 'NextPlot', 'replacechildren');

    j = 0; 
    for time_t = 0:timestep:1
        time_t
        j = j + 1;
        exact = exactSolution(nodes, time_t);
        %scatter3(nodes(:,1), nodes(:,2), exact(:));
        plotSurf(nodes(:,1), nodes(:,2), exact(:));

        A(:, j) = getframe(mfig, winsize); 
        %pause(1);
    end

    % Play movie
    %movie(mfig, A, 30, 3, winsize); 
    %mpgwrite(A,jet,'movie.mpg'); 
    movie2avi(A,'movie.avi');

end

function [solution] = exactSolution(nodes, time_t)

% Euclidean distance r
r = sqrt(nodes(:,1).*nodes(:,1) + nodes(:,2).*nodes(:,2));

% Subtle change to the original NCAR pde: replace + with - and add decay
% exponential
solution = (sin((r-1).*(r-0.5)+pi) - (r-1).*(r-0.5)).*diffusivity(nodes(:,1), nodes(:,2), time_t);

%solution = sin((r-1).*(r-0.5) + pi) * exp(time_t) + (r-1).*(r-0.5);
%saxissymmetric 
%cos wave (radius a -> b) 
%cos((x-a / (b-a))*2pi * FREQ)*e^{-alpha t} ==> 0:2pi
%der of cos satisfies neumann 
% increase frequency from 1
end

function [D] = diffusivity(x, y, t) 
decay = 1.;
UNIFORM_DIFFUSION = 0;
NONUNIFORM_IN_TIME = 0;

D = zeros(length(x),1); 
for i = 1:length(D)
   if UNIFORM_DIFFUSION
    D(i) = exp(-decay*t);
   else 
     if (x(i) > 0)
         if (y(i) > 0 && NONUNIFORM_IN_TIME)
             if (t > 0.5)
                D(i) = exp(-(1.5*decay)*t); 
             else 
                D(i) = exp(-decay*t); 
             end
         else 
             D(i) = exp(-decay*t); 
         end
     else 
         D(i) = exp(-(2*decay)*t); 
     end
   end
end
end


