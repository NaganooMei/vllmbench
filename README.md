# vllmbench

通过 `vllm bench serve` 对比 HBM prefix cache、UCM Host Cache 和 UCM Posix 命中性能。Python 主程序及表格导出使用标准库，运行环境需要已有的 vLLM CLI，Python 版本要求 3.10 或更新。

## 配置

修改仓库中的 `bench.json`，填写服务地址、模型、DP 数以及全部 case。每个 case 提供 `input_len`、`output_len`、`hit_rate`、`concurrency`、`num_requests`，长度单位均为 token，命中率范围为 0～1。`concurrency` 和 `num_requests` 均按整个客户端测量批次计算。

每轮自动生成随机 seed，每个 case 使用独立 seed。预埋和测量使用相同 seed 和请求数，随机长度波动固定为 0。可在配置中设置整数 `seed` 复现指定实验，默认采用随机生成。

可选配置字段如下。

```json
{
  "routing": {"mode": "auto"},
  "block_tokens": 1152,
  "results_dir": "results",
  "seed_settle_seconds": 1,
  "reset_timeout_seconds": 60,
  "metrics": {
    "enabled": false,
    "url": "http://localhost:18081/metrics",
    "settle_seconds": 1,
    "timeout_seconds": 10
  }
}
```

`block_tokens` 应填写服务的实际 LCM 缓存 block 长度。默认值为 1，服务最终按自己的缓存粒度决定命中长度。目标前缀按 block 向下取整。100% 目标会给最终 token 重算预留空间，实际比例以指标为准。模型的最大上下文需要容纳输入加输出。

## HBM 命中

在开启 HBM prefix caching 的基线服务上执行一次。

```bash
python bench.py hbm bench.json
```

自动对每个 case 在所有 DP 分别预热公共前缀，再运行正式测量。正式请求共享公共前缀，剩余后缀各自生成。为了保持 vLLM 随机前缀的生成对应，预热使用与测量相同的请求数。

## UCM Host Cache 命中

在 UCM 服务上执行一次。

```bash
python bench.py cache bench.json
```

每个 case 的全部请求前缀会在每个 DP 都预埋一遍，然后调用 `POST /reset_prefix_cache?reset_external=false`，清理 HBM 前缀缓存，接着测量。每个 DP 的 Host Cache 需要容纳该 case 的完整预埋集合。reset 返回失败时自动重试至超时。

内部 LB 会把 reset 广播到全部 DP。本地核对的 vLLM 版本只返回第一个 DP 的 reset 结果，开启 metrics 后还应检查测量窗口中的实际 HBM 命中。

## UCM Posix 命中

第一条命令预埋全部 case，等待 UCM 写出完成后正常停服，保留 Posix 文件和相同的 key 配置并重启服务，再执行第二条命令。

```bash
python bench.py posix-seed bench.json
# 等待 UCM 写出完成，停服并重启
python bench.py posix-run bench.json
```

第二条命令自动读取匹配配置的最新成功预埋记录及其 seed。也可以用 `--run results/<seed-run>` 指定历史预埋记录。每次重测 Posix 都需要重新建立冷 HBM/Host Cache 状态。`seed_settle_seconds` 是预埋后的等待时间，写出完成仍需结合 UCM 现场状态确认。

## 自动 LB 和固定 DP

默认模式下，预埋覆盖每个 DP，正式请求交给内部 LB 分配。固定 DP 可通过命令行选择。

```bash
python bench.py cache bench.json --dp-rank 0
```

也可在配置中写入 `"routing": {"mode": "fixed", "rank": 0}`。固定模式的预埋与测量都指向同一个 DP，汇总表保留其路由范围。

## 可选 metrics

添加 `--collect-metrics` 即可采集正式命中轮次。

服务需要开放 `/metrics`，UCM 的分层统计和耗时分布需要服务端启用相应的 UCM metrics。客户端采集选项控制本轮快照与报表导出。

```bash
python bench.py hbm bench.json --collect-metrics
python bench.py cache bench.json --collect-metrics
python bench.py posix-run bench.json --collect-metrics
```

预热、预埋完成后至少等待 `metrics.settle_seconds`，reset 完成后采集 before 快照，正式 bench 完成后等待同样的时间再采集 after 快照。请按服务的指标发布周期设置等待时间。采集窗口应使用空闲的专用测试服务，避免其他请求混入。

每个 case 的 `metrics/` 包含原始及过滤后的 Prometheus 快照、按 rank 的 Connector/CacheStore/Posix CSV 与 Excel 报表、直方图增量 avg/p90/p99、命中率 JSON。metrics 关闭时仍输出 bench 性能表，实际命中率列保留为空。

复用了用户 UCM 分支中的 `scripts/collect_ucm_metrics.py`，来源提交为 `3f83ab3ee73fce8fc20c9194eb6f642b3c7c491d`。仓库内的 `collect_ucm_metrics.py` 保持来源文件内容。配置中的 `metrics.collector_path` 可以指向具有相同导出 API 的新版脚本。

HBM 和 external 命中率来自 vLLM token 计数，按汇总后的分子与分母计算。Host Ready 占比以 load shard 为单位。Posix backend 提交数及读取字节单独展示，pending 共享 slot 与实际底层读取分别记录。指标缺失、计数器回退或分母为零时相关结果留空。

## 结果

每次运行自动生成该轮及整个 results 目录的 `summary.csv`、`summary.xlsx`，并保存每个 case 的 bench JSON、日志、完整命令、seed 和实验 manifest。失败请求及 metrics 采集异常会保存在结果中，并使命令返回非零状态。

可以重新汇总已有结果。

```bash
python bench.py report results/
```

原来的 Bash 入口也会调用通用工具，`ucm-seed` 和 `ucm-run` 分别映射到 `posix-seed`、`posix-run`。`BENCH_CONFIG` 可以指定配置文件，原有的 MODEL、SERVED_MODEL、HOST、PORT、SEED、BLOCK_TOKENS、RESULT_DIR 和 DRY_RUN 环境变量继续生效。

无请求预览可使用 `--dry-run`。`posix-run` 预览读取已有预埋记录。默认用完整 CLI 帮助检测参数。缺少 `--ready-check-timeout-sec` 时，通过当前 Python 环境中的同一个 vLLM CLI 入口，仅替换可识别的 `wait_for_endpoint` 试跑 hook，避免首条测量请求提前热缓存。未识别的实现会在发请求前报错。

## 本地检查

```bash
python -m unittest discover -s tests -v
```

测试使用模拟 bench 和 HTTP 服务，覆盖工作流与导出，真实 GPU/NPU 性能由远端服务验证。
