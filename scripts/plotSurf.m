function [] = plotSurf(x, y, Z, showNodes, titlestr)
% Based on surf_from_scatter.m from Matlab fileexchange:
% http://www.mathworks.com/matlabcentral/fileexchange/5105

if (nargin < 5)
   titlestr = ''; 
end

if (nargin < 4) 
    showNodes = 0; 
end

fig1 = figure;
hold on; 
% if z is longer than x and y, we assume we ignore the nodes at the front
% and match up values with the nodes at the back
nbignore = length(x) - length(Z); 

% Cap X,Y,Z is surface plot
X = x((nbignore+1):end);
Y = y((nbignore+1):end);
% Lower x,y,z is point plot3
z = [zeros(nbignore,1);Z];
z(1:nbignore,1) = max(Z);

tri = delaunay(X,Y);
p = [X(:),Y(:)];
pmid=zeros(size(tri,1),2);
for ii=1:3
   pmid=pmid+p(tri(:,ii),:)/3;  % Compute the circumcenter of the tetrahedra.
end
Ri = 0.5; % INNER RADIUS
% Only keep the tetrahedra whose circumcenters are inside shell.
tri=tri(sqrt(pmid(:,1).^2 + pmid(:,2).^2) > Ri - 1e-3,:);


% Too Many nodes will wash out our surface
if length(x) < 2000 || showNodes
    plot3(x,y,z,'.m')
end

%title(titlestr); 
[r,c] = size(tri);
%disp(r);

%figure;
h = trisurf(tri, X, Y, Z);
title(titlestr);
minz = min(z) 
maxz = max(z)
axis([-1 1 -1 1 min(z) max(z)]);
%axis vis3d

axis off
l = light('Position',[-1 -1 0]);
%set(gca,'CameraPosition',[208 -50 7687])
lighting phong
shading interp

colorbar EastOutside
hold off; 


end
