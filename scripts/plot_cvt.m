function [] = plot_cvt(iter, step)
% PLOT_CVT : octave program to draw mesh as a series of points
% PLOT_CVT(iter, step)
% iter => the total number of iterations the CVT ran
% step => the number of iterations run between intermediate file outputs
%       (default step == 20)

if (nargin < 2) || isempty(step)
  step = 20
end
if (nargin < 1) || isempty(iter)
  iter = 1000
end

for i = 0:step:iter
    file=sprintf('voronoi_tmp_%.5d.txt',i)
    %file='voronoi_tmp.txt';
    %file='cvt_circle.txt';

    pts = load(file);

    figure(1);
    X = pts(:,1);
    Y = pts(:,2);
    Z = zeros(size(pts,1), 1); 
    if size(pts,2) > 2
        Z = pts(:,3);
    end
    
    subplot(2,1,1);
    plot3(X, Y, Z, 'ko','markerfacecolor','k'); 
    xlabel('x'); ylabel('y'); zlabel('z');
    axis([0 1 0 1 0 1]);
    
    if (size(pts,2) > 2)
    
        subplot(2,1,2);
        plotVoronoi(X, Y, Z); 
        axis([0 1 0 1 0 1]);
    else
        [VX,VY] = voronoi(X, Y); 
        plot(VX, VY, '-', X, Y, '.');
        xlabel('x'); ylabel('y'); zlabel('z');
        axis([0 1 0 1]);
    end
    
    drawnow
    pause;
end
end

function [] = plotVoronoi(x, y, z)

plot3(x,y,z,'Marker','.','MarkerEdgeColor','r','MarkerSize',10, 'LineStyle', 'none') 
X=[x(:) y(:) z(:)]; 
[V,C]=voronoin(X); 
V(1,:) = 1;
V(V > 1) = 1;
V(V < 0) = 0;
for k=1:length(C) 
    if all(C{k}~=1) 
       VertCell = V(C{k},:); 
       KVert = convhulln(VertCell); 
       patch('Vertices',VertCell,'Faces',KVert,'FaceColor','g','FaceAlpha',1.0) 
    end 
end
% T = delaunay3( x, y, z, {'Qt', 'Qbb', 'Qc', 'Qz'} );
%tetramesh(T,X);
xlabel('x'); ylabel('y'); zlabel('z');
end
