# Mooncake KV Pool 部署 TODO

## 0. 编译构建 Mooncake tq4-rc1

| 任务 | 状态 | 备注 |
|------|------|------|
| 编译构建 | ✅ 完成 | `build_mooncake.sh`, Mooncake v0.3.11-rc1, USE_ASCEND_DIRECT=ON |
| 安装部署 | ✅ 完成 | `install_mooncake.sh`, store/engine 导入成功 |

## 1. scenario_standalone_kvboth

| 任务 | 状态 | 备注 |
|------|------|------|
| 部署脚本 | ✅ 通过 | `mooncake.sh`, 基于新编译 Mooncake v0.3.11-rc1 |
| Smoke Test | ✅ 通过 | 见 `smoke_test.output` |
| Benchmark Test | ⬜ 待开展 | |

## 2. scenario_multi_instance_pool

| 任务 | 状态 | 备注 |
|------|------|------|
| 部署脚本 | ✅ 通过 | `mooncake.sh`, 基于新编译 Mooncake v0.3.11-rc1 |
| Smoke Test | ✅ 通过 | 见 `smoke_test.output` |
| 池化验证 | ⬜ 待开展 | 跨实例池化，见 `test_pool.sh`, 当前 External prefix cache hit rate = 0% |
| Benchmark Test | ⬜ 待开展 | |

## 3. scenario_pd_disaggregation

| 任务 | 状态 | 备注 |
|------|------|------|
| 部署脚本 | ✅ 通过 | `mooncake.sh`, 基于新编译 Mooncake v0.3.11-rc1 |
| Smoke Test | ✅ 通过 | 见 `smoke_test.output` |
| 池化验证 | ⬜ 待开展 | PD 分离池化 |
| Benchmark Test | ⬜ 待开展 | |
