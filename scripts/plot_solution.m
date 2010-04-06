% octave program to draw mesh as a series of points

%file='voronoi_tmp.txt';
%file='cvt_circle.txt';
file = 'FINAL_SOLUTION.txt';
pts = load(file);

efile = 'FINAL_ERROR.txt'; 
epts = load(efile); 


%newplot;
x = epts(:,1);
y = epts(:,2);
z = epts(:,3);

% interpolate solution to regular mesh for visualization
% Borrowed from
% http://www.mathworks.com/access/helpdesk/help/techdoc/visualize/f0-45715.html
xlin = linspace(-1,1,1000);
ylin = linspace(-0.5,0.5,500);

[X,Y] = meshgrid(xlin,ylin);

Z = griddata(x,y,z,X,Y,'cubic');


h = figure(1);
%set(h, 'Position' , [60 60 1024 300]);

%subplot(1,2,2);
hold on; 
%plot3(x,y,z,'*'); 
errPlot = surf(X,Y,Z);
set(errPlot, 'FaceColor', 'interp', 'EdgeColor', 'none');

%scatter3(x,y,z,1,'filled');
%shading interp;
set(gca, 'PlotBoxAspectRatio', [1.0 0.5 1]);
set(gca,'FontSize', 18);
colormap jet;
xlabel('Pos X'); 
ylabel('Pos Y'); 
zlabel('Temperature');
title('Solution Error at t=0.08 (iter: 1000)');
vcb = colorbar('vert');
set(gca,'FontSize', 18);
set(get(vcb, 'YLabel'), 'String', 'Error','FontSize', 14);
set(gca,'FontSize', 18);
view(2)
hold off; 




%subplot(1,2,1);
figure(2)
plot(x,y,'*');
set(gca, 'PlotBoxAspectRatio', [1 0.5 1]);
set(gca,'FontSize', 18);
xlabel('Pos X');
ylabel('Pos Y');
title('RBF Centers'); 
vcb = colorbar('vert');
colorbar('off');

