# GPGPU-Sim 局部研究入口

当前分支：`research/hopper-ptx`。

当前基线：内层 `dev`，commit `91880c53`。外层完整规则、计划、证据和阶段文档位于
`../../../.codex/`，本目录只记录 GPGPU-Sim 局部上下文。

重点代码区域：

- `src/cuda-sim/ptx.l`、`ptx.y`、`opcodes.def`：PTX 前端。
- `src/cuda-sim/instructions.cc`：功能执行。
- `src/abstract_hardware_model.h`：`warp_inst_t`、TMA、GMMA、cluster、mbarrier 字段。
- `src/gpgpu-sim/shader.cc`：shader/LDST 时序和 mbarrier 状态更新。
- `../trace-driven/trace_driven.cc`：SASS trace 到共享 instruction 的动态字段来源。

当前只允许执行外层计划指定的阶段；验收记录必须写回外层阶段文档。
