function cmap = colormap(name, n)
%COLORMAP  Matplotlib's colormaps in MATLAB, so both languages match.
%   cmap = sbp.colormap('viridis')   sequential (Python: cmap="viridis")
%   cmap = sbp.colormap('RdBu_r')    diverging, blue-white-red
%                                    (Python: cmap="RdBu_r")
%   Both are readable with red-green colour blindness. Append _r to
%   'viridis' to reverse it. n is the number of colours (default 256).

if nargin < 2, n = 256; end
switch erase(name, '_r')
    case 'viridis'
        anchors = [0.267 0.005 0.329; 0.283 0.141 0.458; 0.254 0.265 0.530;
                   0.207 0.372 0.553; 0.164 0.471 0.558; 0.128 0.567 0.551;
                   0.135 0.659 0.518; 0.267 0.749 0.441; 0.478 0.821 0.318;
                   0.741 0.873 0.150; 0.993 0.906 0.144];
    case 'RdBu'   % stored blue to red, i.e. matplotlib's RdBu_r
        anchors = [0.020 0.188 0.380; 0.127 0.396 0.669; 0.263 0.576 0.765;
                   0.566 0.769 0.869; 0.820 0.898 0.941; 0.969 0.966 0.965;
                   0.992 0.859 0.780; 0.955 0.642 0.506; 0.839 0.376 0.302;
                   0.692 0.092 0.168; 0.404 0.000 0.122];
        if ~endsWith(name, '_r'), anchors = flipud(anchors); end
    otherwise
        error('sbp:colormap', 'Unknown colormap ''%s''.', name);
end
if strcmp(name, 'viridis_r'), anchors = flipud(anchors); end
x = linspace(0, 1, size(anchors, 1));
cmap = interp1(x, anchors, linspace(0, 1, n));
end
