%% smoke_test.m — 环境自检 / 潮流计算冒烟测试
%% 用途: 验证 Octave + MATPOWER 装好了、能算出正确结果。
%% 运行:  ./octave.sh src/examples/smoke_test.m

define_constants;   % 引入 PD/PG/VM/VA/PF/PT 等列索引常量,避免记列号

fprintf('\n===== 1. 基本潮流计算 (case9) =====\n');
results = runpf('case9');
fprintf('收敛: %d,  迭代次数: %d\n', results.success, results.iterations);
fprintf('总发电: %.2f MW   总负荷: %.2f MW\n', ...
    sum(results.gen(:, PG)), sum(results.bus(:, PD)));
% case9 参考值: 总发电 319.64 MW, 负荷 315.00 MW
assert(results.success == 1);
assert(abs(sum(results.gen(:, PG)) - 319.6413) < 0.01);
fprintf('  [OK] 与官方参考值一致\n');

fprintf('\n===== 2. 母线电压 =====\n');
fprintf('母线 5 电压幅值: %.4f p.u.\n', results.bus(5, VM));

fprintf('\n===== 3. 修改算例: 负荷 +20%% 再算 =====\n');
mpc = loadcase('case14');
mpc.bus(:, PD) = mpc.bus(:, PD) * 1.2;   % 有功负荷 ×1.2
mpc.bus(:, QD) = mpc.bus(:, QD) * 1.2;   % 无功负荷 ×1.2
r2 = runpf(mpc);
fprintf('负荷加重后总发电: %.2f MW\n', sum(r2.gen(:, PG)));

fprintf('\n===== 4. 换算法: 高斯-赛德尔 / 快速解耦 =====\n');
mpopt = mpoption('pf.alg', 'GS', 'verbose', 0);     % GS   = 高斯-赛德尔
rg = runpf('case14', mpopt);
mpopt = mpoption('pf.alg', 'FDXB', 'verbose', 0);   % FDXB = 快速解耦(XB 版)
rf = runpf('case14', mpopt);
fprintf('牛顿法 迭代 %d 次, GS 迭代 %d 次, FDXB 迭代 %d 次\n', ...
    r2.iterations, rg.iterations, rf.iterations);

fprintf('\n===== 5. 画电压曲线并存成图片 =====\n');
figure('visible', 'off');   % 无界面环境, 画到内存
bar(results.bus(:, VM));
title('case9 bus voltage magnitude'); xlabel('bus'); ylabel('V (p.u.)');
out = fullfile(fileparts(mfilename('fullpath')), 'voltage_case9.png');
print('-dpng', '-r100', out);
fprintf('已保存: %s\n', out);

fprintf('\n[PASS] 环境正常,以上全部通过。\n');
