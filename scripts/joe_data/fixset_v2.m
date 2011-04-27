function [peb] = fixset_v2(x,ir,or)

[n,d]=size(x);
r = 0;
for i = 1:d
    r = r+x(:,i).^2;
end
r = sqrt(r);
ind_or = find(abs(r-or)<10^-2);

for i = 1:d
    x(ind_or,i) = x(ind_or,i)./r(ind_or);
end

peb = x(ind_or,:);
