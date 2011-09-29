function[centers]=interpolate2D_ga(gen_rand)

if nargin < 1
    gen_rand = 0;
end
%close all; 

evalRBF = @(mymatrix, myepsilon) GA(mymatrix, myepsilon);

epsilon = 2;

N = 100;
M = N;

a = -2; 
b = 2;

%[x,y] = meshgrid(linspace(a,b,N),linspace(a,b,M));

%centers = load('FINAL_SOLUTION.txt'); 
%centers = (0.75*a) + 0.75*(b-a).* rand(NUM_CENTERS,2)
centers = [ 0.0 0.0; 
            1.0 0.0;
            0.0 1.0; 
            1.0 1.0;
            -1.0 0.0;
            0.0 -1.0;
            -1.0 -1.0;
            -1.0 1.0;
            1.0 -1.0;];
% randomly displace regular grid       
if (gen_rand)
%centers = centers + (0.375 - (0.75).*rand(size(centers)));
centers = (1.5 - 3.*rand(gen_rand, 2));
NUM_CENTERS = size(centers,1);
centers
solution = ones(NUM_CENTERS,1) + 2*rand(NUM_CENTERS,1)+1;
solution
else 
    centers = [

    0.2082    0.5459
   -1.3847   -0.3259
   -0.7872   -1.2306
    1.4780   -1.2273
   -0.5401   -0.2748
   -0.6179    0.5023
   -0.4354   -1.0592
   -0.1569    0.1728
    0.8457   -1.2131
   -0.8171    1.4005
    0.8159   -0.0973
    0.3874   -0.6495
   -1.1728    0.9621
   -1.0691    0.4904
    0.2927    0.9369];


solution = [
    2.6439
    2.8077
    3.0971
    2.0975
    3.1055
    2.5496
    2.4830
    2.4863
    2.3083
    3.9128
    3.8713
    3.6374
    3.4565
    2.3516
    2.7207
    ];
NUM_CENTERS = size(centers,1);
end

local_x = zeros(NUM_CENTERS,M,N);
local_y = zeros(NUM_CENTERS,M,N);
distMatrix = zeros(NUM_CENTERS,N,M);
for k = 1:NUM_CENTERS
    % These scalars added and subtracted control footprint of basis
    % function in left panel
    c_x1 = centers(k,1)-1; 
    c_x2 = centers(k,1)+1;
    c_y1 = centers(k,2)-1; 
    c_y2 = centers(k,2)+1;
    
    % Show local support only
    [lx,ly] = meshgrid(linspace(c_x1,c_x2,N),linspace(c_y1,c_y2,M));
    local_x(k,:,:) = lx;
    local_y(k,:,:) = ly;
    for i = 1:N
      for j = 1:M
            distMatrix(k,i,j) = sqrt(((lx(i,j) - centers(k,1)).^2) + ((ly(i,j) - centers(k,2)).^2));
        end
    end
end
RBFs = evalRBF(distMatrix,epsilon);
RBFs(isnan(RBFs)) = 0;

linewidth = 1;
fontsize = 18;

figure;
set(gcf, 'Position', [100, 900, 1024, 440]);
% Plot RBFs under solution point stems
subplot(1,2,1);
%figure(1)
stem3(centers(1:NUM_CENTERS,1),centers(1:NUM_CENTERS,2),solution,'fill','ro','LineWidth',linewidth,'MarkerEdgeColor',[1.0 0.0 1.0],'MarkerFaceColor', [0.0 1.0 0.0],'MarkerSize',7);
hold on;
for i = 1:NUM_CENTERS 
    lx(:,:) = local_x(i,:,:);
    ly(:,:) = local_y(i,:,:);
    p1=surf(lx,ly,squeeze(RBFs(i,:,:)), 'LineWidth', linewidth);
    set(p1, 'FaceColor', 'interp','EdgeColor','None');
end
hold off;
set(gca,'FontSize',fontsize);
%shading interp;
colormap(bone); %winter
invgray = flipud(get(gcf,'colormap'));
set(gcf,'colormap',invgray);
% Turn off the 3D bounding box so we can see things easier
set(gca,'Box','off');
%axis([-2 2 -2 2 1 3.0]);
%title('');
xlabel('x');
ylabel('y');
zlab = sprintf('%s,  (%s = %g)', '$\phi (r)=e^{-(\epsilon r)^2} $', '$\epsilon$', epsilon);
title(zlab, 'Interpreter', 'Latex','FontSize',24);
axis tight
grid on; 
view(3);
pbaspect([1 1 0.65])


% Plot interpolant connecting stems
% Form real distance matrix
realDistMatrix = zeros(NUM_CENTERS,NUM_CENTERS);
for i = 1:NUM_CENTERS
    for j = 1:NUM_CENTERS
        % Dist matrix for center to center interaction
        if (i == j) 
            realDistMatrix(i,i) = 0;
        else    
            realDistMatrix(i,j) = sqrt(((centers(i,1) - centers(j,1)).^2) + ((centers(i,2) - centers(j,2)).^2));
        end
    end
end

% Here we have 3x1 zero vector for a_0 + a_1 * x + a_2 * y;
RHS = [solution; zeros(3,1)];
% P = [1 X Y]
P = [ones(NUM_CENTERS, 1) centers(1:NUM_CENTERS, 1) centers(1:NUM_CENTERS, 2)];

Phi = evalRBF(realDistMatrix, epsilon);
Phi(isnan(Phi)) = 0; 


A = [Phi  P; 
    P'  zeros(3,3)];

coeffs = A\RHS;

% Reconstruct surface
reconstrMatrix = zeros(N*M, NUM_CENTERS);
% Reconstruct over full domain.
[x,y] = meshgrid(linspace(a,b,N),linspace(a,b,M));
for k = 1:NUM_CENTERS
for i = 1:N
    for j = 1:M
            reconstrMatrix((j-1)*M + i,k) = sqrt(((x(i,j) - centers(k,1)).^2) + ((y(i,j) - centers(k,2)).^2));
        end
    end
end

%C = coeffs(1:size(coeffs,1)-3);
EM = evalRBF(reconstrMatrix, epsilon);
EM(isnan(EM)) = 0; 
EvalMatrix = [EM ones(N*M,1) reshape(x, [N*M 1]) reshape(y, [N*M 1])];

z = reshape( (EvalMatrix * coeffs), N, M);
%figure(2)
subplot(1,2,2);
stem3(centers(1:NUM_CENTERS,1),centers(1:NUM_CENTERS,2),solution,'fill','ro','LineWidth',linewidth,'MarkerEdgeColor',[1.0 0.0 0.0],'MarkerFaceColor', [0.0 1.0 0.0],'MarkerSize',7);
%for i = 1:NUM_CENTERS 
%    p1=surf(x,y,squeeze(RBFs(i,:,:)), 'LineWidth', linewidth);
%    set(p1, 'FaceColor', 'interp','EdgeColor','None');
%end
hold on;
p1 = surfc(x, y, z);
set(p1, 'FaceColor', 'interp','EdgeColor','None');
hold off;
set(gca,'FontSize',fontsize);
% Turn off the 3D bounding box so we can see things easier
set(gca,'Box','off');
%shading interp;
colormap(bone); %winter
invgray = flipud(get(gcf,'colormap'));
set(gcf,'colormap',invgray);
%axis([-2 2 -2 2 1 3.0]);
%title('');
xlabel('x');
ylabel('y');
title('$\hat{f}_N = \sum_{j=1}^{N} w_j \Phi_j(r)$', 'Interpreter', 'latex','FontSize',24);
axis tight
grid on; 
view(3);
pbaspect([1 1 0.65])
end


function[phi] = MQ(r,eps)
    phi=sqrt(1+(eps*r).^2);
end

function[phi] = IMQ(r,eps)
    phi=1./(sqrt(1.+(eps.*r).^2));
end

function[phi] = GA(r,eps)
    phi=exp(-(eps.*r).^2);
end

function[phi] = TPS(r,eps)
    phi=((eps.*r).^2) .* log(eps.*r);
end

function[phi] = W2(r,eps)
    phi=((1-eps.*r).^4) .* (4.*eps.*r + 1);
end