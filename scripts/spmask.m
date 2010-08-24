function [mask] = spmask(x) 
	if (abs(x) > 0.) 
		mask = 1;
	else 
		mask = 0;
	end
end

