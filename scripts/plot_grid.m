function [] = plot_grid(filename, dim, n_stop)

nodes = load(filename); 
if nargin > 2
    boundary = nodes(1:n_stop, :);
else
    boundary = nodes;
end

if dim == 1
    y = zeros(1,length(nodes(:,1))); 
    plot(nodes(:,1),y, '.'); 
elseif dim == 2
    plot(nodes(:,1), nodes(:,2), '.'); 
elseif dim == 3
    subplot(2,1,1);
    plot_3d_subgrid(boundary, dim, 0);
    
    boundary = nodes(find(boundary(:,2) <= 0),:);

    subplot(2,1,2); 
    plot_3d_subgrid(boundary,dim, 0);
    
%    subplot(1,2,2); 
%    plot_3d_subgrid(nodes,dim, 1);
   
%    subplot(1,2,4); 
%    plot_3d_subgrid(boundary,dim, 1);
end
end

function [] = plot_3d_subgrid(nodes, dim, type)
    if nargin < 3
        type = 0
    end 

    if type == 0
        %plot_3d_shell(nodes);
        plot_3d_hull(nodes); 
    else 
        plot3(nodes(:,1), nodes(:,2), nodes(:,3), '.');
    end
    pbaspect([1 0.5 0.5]);
    axis tight

end

function [] = plot_3d_hull(p) 
% Color for the surfaces (in RGB)
bcol=[250 250 0]/256;
icol=[250 250 0]/256;
% Color for the lines indicating where the spherical shell will be split.
lcol = [255 0 204]/255;

    K = convhulln(p);
    hW = trisurf(K,p(:,1),p(:,2),p(:,3));
    
hold off
% Set the properties of the patches.
set(hW,'facecolor',icol,'FaceAlpha',0.75,'edgelighting','phong','facelighting','phong','LineStyle','none','marker','.','markeredgecolor','b','markersize',15);
axis equal
view([40.5 10])
% Add a light
camlight headlight
% Plot line around sphere where it will be cut in half.
hold on;
%thc = linspace(-pi/2,pi/2,101)';
%[xc,yc,zc] = sph2cart(0*thc,thc,0*thc+Ro);
%plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
%[xc,yc,zc] = sph2cart(0*thc+pi,thc,0*thc+Ro);
%plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
%plot3(p(:,1), p(:,2), p(:,3),'--','LineWidth',2,'Color',lcol)
%plot3(p(:,1), p(:,2), p(:,3),'.','Color',lcol)
% axis off
hold off;

end


function [] = plot_3d_shell(p) 
% Color for the surfaces (in RGB)
bcol=[250 250 0]/256;
icol=[250 250 0]/256;
% Color for the lines indicating where the spherical shell will be split.
lcol = [255 0 204]/255;

t=delaunayn(p);
pmid=zeros(size(t,1),3);
for ii=1:4
   pmid=pmid+p(t(:,ii),:)/4;  % Compute the circumcenter of the tetrahedra.
end
% Only keep the tetrahedra whose circumcenters are inside shell.
%t=t(feval(fdist,pmid)<-1e-3,:);
% First do the whole sphere
%figure
%subplot(2,2,1)
%triW=surftri(p,t);
%hW=trimesh(triW,p(:,1),p(:,2),p(:,3));
hW = tetramesh(t, p);

hold off
% Set the properties of the patches.
set(hW,'facecolor',icol,'FaceAlpha',0.75,'edgelighting','phong','facelighting','phong','LineStyle','none','marker','.','markeredgecolor','b','markersize',15);
axis equal
view([40.5 10])
% Add a light
camlight headlight
% Plot line around sphere where it will be cut in half.
hold on;
%thc = linspace(-pi/2,pi/2,101)';
%[xc,yc,zc] = sph2cart(0*thc,thc,0*thc+Ro);
%plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
%[xc,yc,zc] = sph2cart(0*thc+pi,thc,0*thc+Ro);
%plot3(xc,yc,zc,'--','LineWidth',2,'Color',lcol)
%plot3(p(:,1), p(:,2), p(:,3),'--','LineWidth',2,'Color',lcol)
% axis off
hold off;

end 