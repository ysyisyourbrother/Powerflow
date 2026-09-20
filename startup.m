%% startup.m — Octave 启动时自动执行
%% 作用: 会话开始时把 MATPOWER 和本项目算法目录加进搜索路径

this_dir = fileparts(mfilename('fullpath'));
mp = fullfile(this_dir, 'matpower');

% 按 MATPOWER 官方 install_matpower 的推荐目录精确加载。
% 不要用 genpath 全量加载 —— 仓库里有 785 个重名 .m 文件(测试目录 */lib/t/ 与
% 各子模块之间),全量加载会让 which() 找到错误版本,测试套件会直接失败。
mp_dirs = {
    'lib', 'lib/t', 'data', 'examples', ...
    'most/lib', 'most/lib/t', 'most/examples', ...
    'mp-opt-model/lib', 'mp-opt-model/lib/t', 'mp-opt-model/examples', ...
    'mips/lib', 'mips/lib/t', 'mips/examples', ...
    'mptest/lib', 'mptest/lib/t'};
for k = 1:numel(mp_dirs)
    addpath(fullfile(mp, mp_dirs{k}));
end
clear mp mp_dirs k

% 项目自己的算法代码:放在搜索路径最前面。
% 若这里出现与 MATPOWER 同名的函数(例如 newtonpf.m),会覆盖官方版本。
src_dir = fullfile(this_dir, 'src');
if exist(src_dir, 'dir')
    addpath(genpath(src_dir), '-begin');
end
clear this_dir src_dir
