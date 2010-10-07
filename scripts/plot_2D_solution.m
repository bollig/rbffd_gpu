function [] = plot_2D_solution()

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


