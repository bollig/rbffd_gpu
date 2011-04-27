function [peb1,peb2,peb3,pib1,pib2,pib3,ip] = fixset_v3(x,ir,or,h)

[n,d]=size(x);
r = 0;
for i = 1:d
    r = r+x(:,i).^2;
end
r = sqrt(r);
ind_or = find(abs(r-or)<0.3*10^-1);
ind_ir = find(abs(r-ir)<0.3*10^-1);

for i = 1:d
    x(ind_or,i) = x(ind_or,i)./r(ind_or);
    x(ind_ir,i) = x(ind_ir,i)./r(ind_ir);
end

peb1 = x(ind_or,:);
peb2 = (or-h)*x(ind_or,:);
peb3 = (or+h)*x(ind_or,:);

pib1 = ir*x(ind_ir,:);
pib2 = (ir+h)*x(ind_ir,:);
pib3 = (ir-h)*x(ind_ir,:);


j = [1:length(x)]';
i = zeros(length(x),1);
i(ind_or) = ind_or;
i(ind_ir) = ind_ir;

boolean = (i ~= j);
ind = find(boolean ==1);
ip = x(ind,:);


