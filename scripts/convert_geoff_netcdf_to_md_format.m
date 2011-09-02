%% convertGeoffNetCDFToMDFormat.m  --- Evan Bollig, 2011
% Converts .nc format files provided by Geoff Womeldorff to the MD node
% format (4 doubles per line: (x,y,z,JUNK)
% File input is binary, output is ascii. 
% 
% If this routine is run with no arguments a user interface appears for
% the user to select one or more files to convert
%
% Argument matFilename specifies the full/absolute path to a file to
% convert
function [] = convertGeoffNetCDFToMDFormat(matFilename)
    if (nargin < 1)
        % The path is in pth
        [filename,pth] = uigetfile('*.nc', 'Pick one or more NetCDF-files', 'MultiSelect', 'on');
        if isequal(filename,0) || isequal(pth,0)
            disp('Operation cancelled');
            return
        end
    else
        filename = matFilename;
    end

    % Decompress the binary file into .ascii and .mtx parts
    if (iscell(filename))
        for i = 1:length(filename)
            decompressFile(filename{i});
        end
    else
        decompressFile(filename);
    end
    
    % Merge 
end


function [] = decompressFile(filename)
[pathstr,file_no_ext,ext,versn] = fileparts(filename);

% Make a directory matching matFilename
mkdir(file_no_ext);

% Foreach variable inside matFilename
ncid = netcdf.open( filename, 'NC_NOWRITE' );
%[varname, xtype, dimids, numatts] = netcdf.inqVar(ncid,0)

% Geoff told me [x,y,z]Cell are the CVT generators. The [x,y,z]Vertices
% are vertices of polygons. 
varid = netcdf.inqVarID(ncid,'xCell');
xdata = netcdf.getVar(ncid,varid);
varid = netcdf.inqVarID(ncid,'yCell');
ydata = netcdf.getVar(ncid,varid);
varid = netcdf.inqVarID(ncid,'zCell');
zdata = netcdf.getVar(ncid,varid);
nodes = [xdata(:) ydata(:) zdata(:) xdata(:)];
tempfilename = strcat('/tmp/', file_no_ext, '_nodes.ascii')
destfilename = strcat(file_no_ext, '_nodes.ascii');
% Write to local disk and transfer to once written
writeToFile(nodes, char(tempfilename));
movefile(char(tempfilename), char(destfilename));

%data = struct('nodes', nodes);
% 
% if (isstruct(data))
%     % Write each field to a separate file.
%     names = fieldnames(data);
%     for i = 1:length(names)
%         d = getfield(data, char(names(i)));
%         % If data is > 3 dimensions we assume its a full matrix. We write
%         % this to matrix market format
%         if size(d,2) > 3
%             writeToMMFile(d, char(strcat(file_no_ext, '/', names(i), '.mtx')));
%         else 
%            writeToFile(d, char(strcat(file_no_ext, '/', names(i), '.ascii')));
%         end
%     end
% else
%     fprintf('NOT SUPPORTED YET (See Evan Bollig)\n');
% end
end

function [] = writeToMMFile(data, filename)
    fprintf('.....Writing ''%s''\n', filename);
    A = sparse(data);
    figure
    spy(A);
    comment = str2mat('Auto-Converted Matrix From convertMatToAscii (Evan Bollig)',[' ',date]);
    mmwrite(filename,A,comment);
end

function [] = writeToFile(data, filename)

fprintf('.....Writing ''%s''\n', filename);

% TODO: write tuple dimensions to the file header, plus number of tuples

fid = fopen(filename,'w+');

[m n] = size(data);

for i = 1:m
    %for j = 1:n
        fprintf(fid, '%24.16e ', data(i,:));
    %end
    fprintf(fid, '\n');
end

fclose(fid);

end