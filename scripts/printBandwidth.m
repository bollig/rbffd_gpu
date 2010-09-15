function [upperBandwidth] = printBandwidth(A, filename) 
[m n] = size(A);

% value within: [0, n/2 + 1] 
upperBandwidth = 0; 
lowerBandwidth = 0; 
for i = 1:m
    % Looking at the upper bandwidth
    for j = i:n
        if abs(A(i, j)) > 0
        % Element is nonzero    
        upperBandwidth = max([(j - i), upperBandwidth]); 
        end
    end
    
    % Looking at the lower bandwidth
    for j = 1:i
        if abs(A(i, j)) > 0
            % Element is nonzero    
            lowerBandwidth = max([(i - j), lowerBandwidth]); 
        end
    end
end

fprintf(1,'Upper bandwidth of %s is: %d\n', filename, upperBandwidth); 
fprintf(1,'Lower bandwidth of %s is: %d\n', filename, lowerBandwidth); 
flush_io(1); 
end