% octave program to draw mesh as a series of points
function [] = plot_cvt(iter, step)
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

    [VX,VY] = voronoi(X, Y); 

    plot(VX, VY, '-', X, Y, '.');
    axis([0 1 0 1]);
    drawnow
    pause;
end
end
