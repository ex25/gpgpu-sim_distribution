# GPGPU-Sim Hopper PTX 分支

这是配对的 `research/hopper-ptx` 内层仓库。开始工作前读取：

1. `.codex/README.md`
2. `../../../.codex/rules/hopper-ptx.md`
3. `../../../.codex/plans/hopper-ptx.md`
4. 当前阶段的 `../../../.codex/docs/implementation/*.md`

本仓库负责 PTX lexer/parser、functional execution、`warp_inst_t` 连接、shader/LDST
时序和 GPGPU-Sim 局部验证。每次报告都要记录本仓库 commit，并与外层 Accel-Sim
commit 配对。不要把现有实验分支的提交默认为研究基线。
