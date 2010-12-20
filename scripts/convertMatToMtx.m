%% convertMatToMtx.m  --- Evan Bollig, 2010
% Converts .mat format files to a directory of .mtx files. 
% The .mtx format is plain ascii tuples separated by '\n' with tuple
% elements separated by spaces
% 
% If this routine is run with no arguments a user interface appears for
% the user to select one or more files to convert
%
% Argument matFilename specifies the full/absolute path to a file to
% convert
function [] = convertMatToMtx(matFilename)
    if (nargin < 1)
        % The path is in pth
        [filename,pth] = uigetfile('*.mat', 'Pick one or more MAT-files', 'MultiSelect', 'on');
        if isequal(filename,0) || isequal(pth,0)
            disp('Operation cancelled');
            return
        end
    else
        filename = matFilename;
    end

    % Decompose the file
    if (iscell(filename))
        for i = 1:length(filename)
            decompressFile(filename{i});
        end
    else
        decompressFile(filename);
    end
end


function [] = decompressFile(filename)
[pathstr,file_no_ext,ext,versn] = fileparts(filename);

% Make a directory matching matFilename
mkdir(file_no_ext);

% Foreach variable inside matFilename
data = load(filename)

if (isstruct(data))
    % Write each field to a separate file.
    names = fieldnames(data);
    for i = 1:length(names)
        writeToFile(getfield(data, char(names(i))), char(strcat(file_no_ext, '/', names(i), '.mtx')));
    end
else
    fprintf('NOT SUPPORTED YET (See Evan Bollig)\n');
end
end

function [] = writeToFile(data, filename)

fprintf('.....Writing ''%s''\n', filename);

% TODO: write tuple dimensions to the file header, plus number of tuples

fid = fopen(filename,'w+');

[m n] = size(data);

for i = 1:m
    for j = 1:n
        fprintf(fid, '%g ', data(i,j));
    end
    fprintf(fid, '\n');
end

fclose(fid);

end