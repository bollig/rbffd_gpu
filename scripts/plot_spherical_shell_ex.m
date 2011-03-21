%function plot_spherical_shell(p,t)
% Assume p contains the n points in a n-by-3 matrix.
function plot_spherical_shell_ex(p)

% Radii of the outer and inner shells.
Ro = 0.5;
Ri = 1;

% Define a distance function for the spherical shell that is 
% non-positive in inside and zero on the boundary.
fdist = @(p) max(sqrt(p(:,1).^2 + p(:,2).^2 + p(:,3).^2)-Ro,-(sqrt(p(:,1).^2 + p(:,2).^2 + p(:,3).^2)-Ri));

t=delaunayn(p);
pmid=zeros(size(t,1),3);
for ii=1:4
   pmid=pmid+p(t(:,ii),:)/4;  % Compute the circumcenter of the tetrahedra.
end
% Only keep the tetrahedra whose circumcenters are inside shell.
t=t(feval(fdist,pmid)<-1e-3,:);

% Color for the surfaces (in RGB)
bcol=[250 250 0]/256;
icol=[250 250 0]/256;
% Color for the lines indicating where the spherical shell will be split.
lcol = [255 0 204]/255;

% First do the whole sphere
%figure
subplot(2,2,1)
triW=surftri(p,t);
hW=trimesh(triW,p(:,1),p(:,2),p(:,3));
hold off
% Set the properties of the patches.
set(hW,'facecolor',icol,'edgelighting','phong','facelighting','phong','LineStyle','none','marker','.','markeredgecolor','b','markersize',15);
axis equal
view([40.5 10])
% Add a light
camlight
% Plot line around sphere where it will be cut in half.
hold on;
thc = linspace(-pi/2,pi/2,101)';
[xc,yc,zc] = sph2cart(0*thc,thc,0*thc+Ro);
plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
[xc,yc,zc] = sph2cart(0*thc+pi,thc,0*thc+Ro);
plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
% axis off

% Now plot the spherical shell split in half.
%figure
subplot(2,2,2)
triHb=surftri(p,t);   % triangles for the inner and outer boundaries
incl = find(p(:,2) > 0);  % split the sphere open along the x-z plane.
t=t(any(ismember(t,incl),2),:);
triHb=triHb(any(ismember(triHb,incl),2),:);
triHs=surftri(p,t);        % triangles for the sides.
triHs=setdiff(triHs,triHb,'rows');
hS=trimesh(triHs,p(:,1),p(:,2),p(:,3));
hold on;
hB=trimesh(triHb,p(:,1),p(:,2),p(:,3));
hold off
% Set the properties of the patches.
set(hS,'facecolor',icol,'edgelighting','phong','facelighting','phong','LineStyle','none','marker','.','markeredgecolor','k','markersize',15);
set(hB,'facecolor',bcol,'edgelighting','phong','facelighting','phong','LineStyle','none','marker','.','markeredgecolor','b','markersize',15);
axis equal
view([40.5 10])
% Add a light
camlight
hold on;
% Plot the outline of the wedge.
% plotWedgeOutline;
% axis off

%
% Extract out a wedge
%
% Data file containing the nodes.
%load SphericalShellEx
[lam,th] = cart2sph(p(:,1),p(:,2),p(:,3));
incl = find(lam >= 0 & lam <= pi/6 & th >= 0 & th <= pi/4);
p = p(incl,:);
[lam,th,r] = cart2sph(p(:,1),p(:,2),p(:,3));
ido = find(r > Ro - 0.01);
idi = find(r < Ri + 0.01);
% Plot the interior and boundary points as different colors
%figure;
subplot(2,2,4)
plot3(p(:,1),p(:,2),p(:,3),'k.','MarkerSize',15), hold on
plot3(p(ido,1),p(ido,2),p(ido,3),'b.','MarkerSize',15)
plot3(p(idi,1),p(idi,2),p(idi,3),'b.','MarkerSize',15)
% Plot the outline of the wedge.
% plotWedgeOutline;

% Determine the nodes for a 32-by-32 stencil located at the center of 
% the wedge.
fdc = zeros(1,3);
[fdc(1),fdc(2),fdc(3)] = sph2cart(pi/12,pi/8,(Ri+Ro)/2);
% Compute the closest node to the center.
re2 = sum((repmat(fdc,[length(p) 1]) - p).^2,2);
[temp,id] = sort(re2);
nn = 32;
% Plot the nodes of the stencil with the other nodes.
% plot3(p(id(1),1),p(id(1),2),p(id(1),3),'ro','MarkerSize',9,'LineWidth',2)
% plot3(p(id(2:nn+1),1),p(id(2:nn+1),2),p(id(2:nn+1),3),'go','MarkerSize',9,'LineWidth',2)
% view([40.5 10])
% axis off

% Plot the nodes of the stencil.
%figure,
subplot(2,2,3)
plot3(p(id(1),1),p(id(1),2),p(id(1),3),'k.','MarkerSize',15), hold on
plot3(p(id(1),1),p(id(1),2),p(id(1),3),'ro','MarkerSize',9,'LineWidth',2)
plot3(p(id(2:nn+1),1),p(id(2:nn+1),2),p(id(2:nn+1),3),'k.','MarkerSize',15)
plot3(p(id(2:nn+1),1),p(id(2:nn+1),2),p(id(2:nn+1),3),'go','MarkerSize',9,'LineWidth',2)
view([40.5 10])
% axis off


    function plotWedgeOutline
        % Plot the wedge lines;
        thc = linspace(0,pi/4,11)';
        lamc = linspace(-0.1,pi/6,11)';
        rc = linspace(Ri,Ro,11)';
        [xc,yc,zc] = sph2cart(0*lamc + lamc(1),0*thc + thc(1),rc);
        plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
        [xc,yc,zc] = sph2cart(0*lamc + lamc(end),0*thc + thc(1),rc);
        plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
        [xc,yc,zc] = sph2cart(0*lamc + lamc(1),0*thc + thc(end),rc);
        plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
        [xc,yc,zc] = sph2cart(0*lamc + lamc(end),0*thc + thc(end),rc);
        plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
        [xc,yc,zc] = sph2cart(0*lamc + lamc(1),thc,0*rc + rc(1));
        plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
        [xc,yc,zc] = sph2cart(0*lamc + lamc(end),thc,0*rc + rc(1));
        plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
        [xc,yc,zc] = sph2cart(0*lamc + lamc(1),thc,0*rc + rc(end));
        plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
        [xc,yc,zc] = sph2cart(0*lamc + lamc(end),thc,0*rc + rc(end));
        plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
        [xc,yc,zc] = sph2cart(lamc,thc(1),0*rc + rc(end));
        plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
        [xc,yc,zc] = sph2cart(lamc,thc(end),0*rc + rc(end));
        plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
        [xc,yc,zc] = sph2cart(lamc,thc(1),0*rc + rc(1));
        plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
        [xc,yc,zc] = sph2cart(lamc,thc(end),0*rc + rc(1));
        plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
    end
end

% t=t(any(ismember(t,incl),2),:);
% tri1=tri1(any(ismember(tri1,incl),2),:);
% h2=trimesh(tri1,p(:,1),p(:,2),p(:,3));
% set(h2,'facecolor',icol,'edgelighting','phong','facelighting','phong','LineStyle','none','marker','.','markeredgecolor','b','markersize',15);





% % Plot the nodes
% [lam,th,r] = cart2sph(p(:,1),p(:,2),p(:,3));
% % Indicies for the nodes on the top.
% idt = find( r > Ro - 0.01);
% % Nodes on the top;
% pt = p(idt,:);
% % Indicies for the nodes on the top and half the sphere
% idht = find(pt(:,2) > 0);
% h3 = scatter3(pt(idht,1),pt(idht,2),pt(idht,3),50,0*pt(idht,3)+Ri,'.');
% 
% % Indicies for the nodes on the bottom.
% idb = find( r < Ri + 0.01);
% % Nodes on the top;
% pb = p(idb,:);
% % Indicies for the nodes on the bottom and half the sphere
% idhb = find(pb(:,2) > 0);
% h3 = scatter3(pb(idhb,1),pb(idhb,2),pb(idhb,3),50,0*pb(idhb,3)+Ri,'.');

% 
% 
%   tri1=surftri(p,t);
%   if nargin>2 & ~isempty(expr)
%     incl=find(eval(expr));
%     t=t(any(ismember(t,incl),2),:);
%     tri1=tri1(any(ismember(tri1,incl),2),:);
%     tri2=surftri(p,t);
%     tri2=setdiff(tri2,tri1,'rows');
%     h=trimesh(tri2,p(:,1),p(:,2),p(:,3));
%     set(h,'facecolor',icol,'edgecolor','k');
%     hold on
%   end
%   h=trimesh(tri1,p(:,1),p(:,2),p(:,3));
%   hold off
%   set(h,'facecolor',bcol,'edgecolor','k');
%   axis equal
%   cameramenu
% 
% 
% dim=size(p,2);
% switch dim
%  case 2
%   trimesh(t,p(:,1),p(:,2),0*p(:,1),'facecolor','none','edgecolor','k');
%   view(2)
%   axis equal
%   axis off
%  case 3
%   tri1=surftri(p,t);
%   if nargin>2 & ~isempty(expr)
%     incl=find(eval(expr));
%     t=t(any(ismember(t,incl),2),:);
%     tri1=tri1(any(ismember(tri1,incl),2),:);
%     tri2=surftri(p,t);
%     tri2=setdiff(tri2,tri1,'rows');
%     h=trimesh(tri2,p(:,1),p(:,2),p(:,3));
%     set(h,'facecolor',icol,'edgecolor','k');
%     hold on
%   end
%   h=trimesh(tri1,p(:,1),p(:,2),p(:,3));
%   hold off
%   set(h,'facecolor',bcol,'edgecolor','k');
%   axis equal
%   cameramenu
%  otherwise
%   error('Unimplemented dimension.');
% end
