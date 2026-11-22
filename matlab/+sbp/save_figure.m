function save_figure(fig, scriptFile, name)
%SAVE_FIGURE  Save a figure as out/matlab/<name>.png.
%   sbp.save_figure(gcf, mfilename('fullpath'), 'fig-name')
%   Before saving, gives the figure the companion's common look, the same
%   as Python's sbp.style(): plain titles, no box, lines 1.5 points wide
%   and the colour-blind-friendly Okabe-Ito colour order (blue, orange,
%   green, vermillion, purple, sky blue, yellow), see sbp.palette.

okabe_ito = sbp.palette();
for ax = findall(fig, 'Type', 'axes')'
    colororder(ax, okabe_ito);   % recolours lines with default colours
    ax.TitleFontWeight = 'normal';
    box(ax, 'off');
end
for h = findall(fig, 'Type', 'line')'
    if h.LineWidth == 0.5 && ~strcmp(h.LineStyle, 'none')
        h.LineWidth = 1.5;   % MATLAB's default is thinner than Python's
    end
end
for t = findall(fig, 'Type', 'tiledlayout')'
    t.Title.FontWeight = 'normal';
end

file = fullfile(sbp.out_dir(scriptFile), [name '.png']);
if exist('exportgraphics', 'file')
    exportgraphics(fig, file, 'Resolution', 150);
else  % older MATLAB, or GNU Octave
    print(fig, file, '-dpng', '-r150');
end
end
