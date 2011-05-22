function [] = plot_grid(filename, dim)

nodes = load(filename); 

if dim == 1
    y = zeros(1,length(nodes(:,1))); 
    plot(nodes(:,1),y, '.'); 
elseif dim == 2
    plot(nodes(:,1), nodes(:,2), '.'); 
elseif dim == 3
    plot3(nodes(:,1), nodes(:,2), nodes(:,3), '.');
end

end