function [] = plotSurf(x, y, z, titlestr)
% Based on surf_from_scatter.m from Matlab fileexchange:
% http://www.mathworks.com/matlabcentral/fileexchange/5105

if (nargin < 4)
   titlestr = ''; 
end

figure;
tri = delaunay(x,y);
plot(x,y,'.')
title(titlestr); 
[r,c] = size(tri);
%disp(r);

figure;
h = trisurf(tri, x, y, z);
title(titlestr);
axis([-1 1 -1 1 min(z) max(z)]);
%axis vis3d

axis off
l = light('Position',[-1 -1 0]);
%set(gca,'CameraPosition',[208 -50 7687])
lighting phong
shading interp

colorbar EastOutside

end
