function d = data_dir()
%DATA_DIR  The companion's shared data folder (models, datasets).

d = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'data');
end
