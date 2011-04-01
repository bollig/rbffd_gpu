function tilefigs(figs)
%SAVEFIGS Tile all open figure windows around on the screen.
%
%   SAVEFIGS saves all open figures.
%
%   SAVEFIGS(FIGS) can be used to specify which figures that should be
%   tiled. Figures are not sorted when specified.
%
%   See also TILEFIGS, .

%   Author:	 Evan Bollig 
%   Time-stamp:	 2010-10-13 11:40:11 +0500
%   E-mail:	 bollig@scs.fsu.edu
%   URL:	 http://sc.fsu.edu/~bollig

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get the handles to the figures to process.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~nargin				% If no input arguments...
   figs = findobj('Type', 'figure');	% ...find all figures.
   figs	= sort(figs);
else
    figsNums = figs;
   figs = findobj('Type', 'figure');	% ...find all figures.
   figs	= sort(figs);
   figs = figs(figsNums); % save only a subset of them
end

if isempty(figs)
   disp('No open figures or no figures specified.');
   return
end

nfigs = length(figs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put the figures where they belong.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   for i = 1 : nfigs
       % Get the axes element from the figure; get the title object from
       % the axes; get the string from the title
       titlestr = get(get(get(figs(i), 'CurrentAxes'), 'Title'), 'String');
       title = get(get(figs(i), 'CurrentAxes'), 'Title');
%       oldfontsize = get(get(get(figs(i), 'CurrentAxes'), 'Title'), 'FontSize');
 %      set(figs(i),'FontSize', 18);

       window_dim = get(figs(i), 'Position'); 
       % Increase window width to 800x600
       set(figs(i), 'Position', [100, 900, 1024, 768]); 
       
       if iscell(titlestr)
           titlestr = strcat(titlestr{1:end});
       end
       
       titlestr = strrep(titlestr,', ','_');
       titlestr = strrep(titlestr,' ','');
       titlestr = strrep(titlestr,'[','');
       titlestr = strrep(titlestr,'-','');
       titlestr = strrep(titlestr,']','_');
       titlestr = strrep(titlestr,'(','_');
       titlestr = strrep(titlestr,')','_');
       titlestr = strrep(titlestr,':','__');
       % Trim any file extensions in labels
       titlestr = regexprep(titlestr,'\.(\w*)','');
       
       filename = sprintf('%s', titlestr);
       %print(figs(i), '-zbuffer','-dpdf','-r600', filename);
       exportfig(figs(i),filename,'Format','png','FontMode','fixed','FontSize','18','Color','rgb','Renderer','zbuffer','Resolution',600,'bounds','loose');
       % Export a figure with separate text so text is visible at printer 
       % resolution. 
       % exportfig(gcf,'test.eps','separatetext',1);
       
       % Put things back the way we found them: 
 %      set(title,'FontSize', oldfontsize);
       set(figs(i), 'Position', window_dim); 
   end
end
