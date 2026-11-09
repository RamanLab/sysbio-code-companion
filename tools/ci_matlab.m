function ci_matlab()
%CI_MATLAB  What the GitHub Actions MATLAB job runs.
%   Runs the examples that do not need the COBRA Toolbox first, and only
%   then sets up the COBRA Toolbox (if it was downloaded to ~/cobratoolbox)
%   and runs the examples that do, so that a problem with the COBRA Toolbox
%   cannot affect the other examples.

top = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(top, 'tools'));
fprintf('MATLAB %s\n', version);
failed = run_examples([], 'nocobra');

cobra = fullfile(getenv('HOME'), 'cobratoolbox');
if exist(cobra, 'dir')
    here = pwd;
    try
        cd(cobra);
        initCobraToolbox(false);
        if ~changeCobraSolver('glpk', 'LP', 0)
            changeCobraSolver('matlab', 'LP');
        end
        fprintf('::notice title=COBRA Toolbox::ready\n');
    catch err
        msg = strrep(getReport(err, 'extended', 'hyperlinks', 'off'), ...
            newline, '%0A');
        fprintf('::error title=COBRA Toolbox setup::%s\n', msg);
    end
    cd(here);
    failed = [failed, run_examples([], 'cobra')];
end

if ~isempty(failed)
    error('ci_matlab:failed', '%d example(s) failed', numel(failed));
end
end

