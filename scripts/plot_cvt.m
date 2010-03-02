% octave program to draw mesh as a series of points

file='voronoi_tmp.txt';
%file='cvt_circle.txt';
pts = load(file);

newplot;
x = pts(:,1);
y = pts(:,2);
plot(x,y,'*');
