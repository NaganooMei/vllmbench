# vllmbench

通过 `vllm bench serve` 对比 HBM prefix cache、UCM Host Cache 和 UCM Posix 命中性能。Python 主程序及表格导出使用标准库，运行环境需要已有的 vLLM CLI，Python 版本要求 3.10 或更新。

## 配置

修改仓库中的 `bench.json`，填写服务地址、模型、DP 数以及全部 case。每个 case 提供 `input_len`、`output_len`、`hit_rate`、`concurrency`、`num_requests`，长度单位均为 token，命中率范围为 0～1。`concurrency` 和 `num_requests` 均按整个客户端测量批次计算。

每轮自动生成随机 seed，每个 case 使用独立 seed。预埋和测量使用相同 seed 和请求数，随机长度波动固定为 0。可在配置中设置整数 `seed` 复现指定实验，默认采用随机生成。

UCM 预埋直接把 `--random-input-len` 设为目标命中长度，正式测量设为完整输入长度，两阶段均使用 `--dataset-name random --random-prefix-len 0`。例如 32K、90% 的预埋长度是 29491，正式长度是 32768。

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

先在每个目标 DP 上执行独立的 8K 预热组，再依次处理正式 case。每个 case 只向每个 DP 发送 **1 条**请求来预埋共享前缀，预埋并发为 1；正式测量仍使用 case 配置中的请求数与并发，所有请求共享公共前缀，剩余后缀各自生成。

HBM 预埋通过兼容入口传入 `--num-prompts 1 --prefix-source-num-prompts N`，其中 N 为正式组请求数。入口使用正式组的 seed 和 N 生成数据，仅保留第 1 条交给 vLLM 发送，避免直接改变生成请求数导致共享前缀变化。额外参数由入口消费，不会传给原生 vLLM CLI。Cache、Posix 的各请求独立前缀仍需全量预埋。

## UCM Host Cache 命中

在 UCM 服务上执行一次。

```bash
python bench.py cache bench.json
```

每个 case 的全部请求前缀会在每个 DP 都预埋一遍，然后调用 `POST /reset_prefix_cache?reset_external=false`，清理 HBM 前缀缓存，接着测量。每个 DP 的 Host Cache 需要容纳该 case 的完整预埋集合。reset 返回失败时自动重试至超时。

内部 LB 会把 reset 广播到全部 DP。本地核对的 vLLM 版本只返回第一个 DP 的 reset 结果，还应检查测量窗口中的实际 HBM 命中。

## UCM Posix 命中

第一条命令预埋全部 case，等待 UCM 写出完成后正常停服，保留 Posix 文件和相同的 key 配置并重启服务，再执行第二条命令。

```bash
python bench.py posix-seed bench.json
# 等待 UCM 写出完成，停服并重启
python bench.py posix-run bench.json
```

第二条命令自动读取匹配配置的最新成功预埋记录及其 seed。也可以用 `--run results/<seed-run>` 指定历史预埋记录。每次重测 Posix 都需要重新建立冷 HBM/Host Cache 状态。`seed_settle_seconds` 是预埋后的等待时间，写出完成仍需结合 UCM 现场状态确认。

## UCM Posix 清缓存对照组（不重启）

在已包含 UCM 清 Cache 功能的服务上执行：

```bash
python bench.py posix-reset bench.json
```

一条命令自动执行独立加载预热，再对每个 case 在每个目标 DP 预埋全部请求的目标前缀（例如 90% 输入），等待 `seed_settle_seconds`，调用下面的接口，成功后用相同 seed 和相同请求数发送全长正式请求：

```bash
curl -X POST 'http://localhost:18081/reset_prefix_cache?reset_external=true'
```

该接口需要清空 HBM 和 UCM 内存 Cache，保留后端 Posix 文件。reset 返回失败会重试至 `reset_timeout_seconds`，失败或超时则停止该轮，不执行正式测量。预埋等待时间应足够让 Posix 异步写入完成；该组不停止或重启服务，也不需要先跑 `posix-seed`。

预热组同样先预埋，再清 HBM 和 UCM Cache，然后执行加载预热。每个正式 case 的计数窗口从 reset 成功之后开始，预热和清理不计入正式性能结果。汇总表的 Mode 为 `posix-reset`，与停服重启的 `posix` 组区分；支持默认全 DP 预埋加自动 LB、`--dp-rank` 和 `--collect-metrics`。

HBM hit、External hit、Host Ready 和 Posix bytes 可用于核对加载来源；接口成功不等同于已验证 Posix 命中。此流程清的是 UCM 内存缓存，不清操作系统文件页缓存，因此不能视为与服务重启完全相同的底层 I/O 冷态。

## 自动 LB 和固定 DP

默认模式下，预埋覆盖每个 DP，正式请求交给内部 LB 分配。固定 DP 可通过命令行选择。

```bash
python bench.py cache bench.json --dp-rank 0
```

也可在配置中写入 `"routing": {"mode": "fixed", "rank": 0}`。固定模式的预埋与测量都指向同一个 DP，汇总表保留其路由范围。

## 独立预热

HBM、Cache 和 Posix 命中测试都在全部正式 case 之前增加独立预热。沿用配置字段 `ucm_warmup`，默认在每个目标 DP 上执行两组，每组 8K 输入、4 请求、4 并发、约 90% 前缀命中、输出 1 token。预热数据使用独立 seed，日志和 bench JSON 放在该轮的 `warmup_001/` 等目录中，正式汇总及 metrics 采集窗口从后续正式 case 开始。

```json
"ucm_warmup": {
  "enabled": true,
  "rounds": 2,
  "input_len": 8192,
  "output_len": 1,
  "hit_rate": 0.9,
  "concurrency": 4,
  "num_requests": 4
}
```

HBM 模式先向每个目标 DP 发送 1 条预热组的共享前缀，再发送完整的预热组请求命中 HBM，不调用 reset。Cache 模式先预埋预热前缀、reset HBM，再向每个 DP 发送预热加载请求。Posix 的 `posix-seed` 同时预埋预热数据并保存其参数与 seed，重启后的 `posix-run` 先从 Posix 加载这些独立前缀，然后直接进入正式测试。每组预热使用不同 seed。自动 LB 模式覆盖所有 DP，固定 DP 模式覆盖指定 DP，预热组命中阶段每次 bench 的并发为配置中的 `concurrency`。

`enabled` 设为 `false` 可关闭所有模式的独立预热组，正式 case 的前缀预埋仍会执行。Posix 重启后的预热参数和 seed 采用预埋记录中的值。历史预埋记录仍可继续测量，缺少预热数据时会提示本轮跳过加载预热，重新执行 `posix-seed` 可生成预热数据。

## 可选 metrics

默认就会在每个正式 case 前后各读取一次 `/metrics`，用计数器增量计算 HBM 和 external 实际命中率，打印到终端并写入汇总 CSV、Excel。不需要添加 `--collect-metrics`，也不导出详细快照或耗时分布。该默认读取失败时显示 `n/a` 和原因，性能测试继续执行。

实际命中率需要服务开放 `/metrics` 并提供对应的 vLLM prefix cache 计数。添加 `--collect-metrics` 才启用详细快照和报表导出；UCM 的分层统计和耗时分布需要服务端启用相应的 UCM metrics。

```bash
python bench.py hbm bench.json --collect-metrics
python bench.py cache bench.json --collect-metrics
python bench.py posix-run bench.json --collect-metrics
```

默认命中率统计和详细采集都只覆盖正式测量窗口。预热、预埋完成后至少等待 `metrics.settle_seconds`，Cache 模式 reset 完成后读取 before 计数，正式 bench 完成后等待同样的时间再读取 after 计数。请按服务的指标发布周期设置等待时间。采集窗口应使用空闲的专用测试服务，避免其他请求混入。

启用详细采集时，每个 case 的 `metrics/` 包含原始及过滤后的 Prometheus 快照、按 rank 的 Connector/CacheStore/Posix CSV 与 Excel 报表、直方图增量 avg/p90/p99、命中率 JSON。不开详细采集时，实际命中率仍保存在 `case_result.json` 和汇总表中。

复用了用户 UCM 分支中的 `scripts/collect_ucm_metrics.py`，来源提交为 `3f83ab3ee73fce8fc20c9194eb6f642b3c7c491d`。仓库内的 `collect_ucm_metrics.py` 保持来源文件内容。配置中的 `metrics.collector_path` 可以指向具有相同导出 API 的新版脚本。

HBM 和 external 命中率来自 vLLM token 计数，按汇总后的分子与分母计算。Host Ready 占比以 load shard 为单位。Posix backend 提交数及读取字节单独展示，pending 共享 slot 与实际底层读取分别记录。指标缺失、计数器回退或分母为零时相关结果留空。

## 结果

每次运行自动生成该轮及整个 results 目录的 `summary.csv`、`summary.xlsx`，并保存每个 case 的 bench JSON、日志、完整命令、seed 和实验 manifest。失败请求及显式启用的详细 metrics 采集异常会保存在结果中，并使命令返回非零状态。默认轻量计数读取异常只标记命中率不可用。

终端的 `Model` 列和 CSV、Excel 的 `served_model` 列使用本轮 `bench.json` 的 `served_model`。重新汇总旧结果时，会从该轮保存的 `manifest.json` 恢复模型名称。

终端和 CSV、Excel 汇总同时记录 TTFT、TPOT（均值，单位 ms）、输出吞吐量 `Output tok/s` 和总 token 吞吐量 `Total tok/s`。总吞吐量使用 vLLM 的 `total_token_throughput`，即成功请求的输入加输出 token 数除以正式 bench 时长，输入包含命中的缓存 token；输出吞吐量单独展示。上述性能指标无需开启 `--collect-metrics`。

终端还显示 Target hit、HBM hit、External hit、Host Ready 和 Posix bytes。实际 0 命中显示 `0.0%`。缺少数据的列显示 `n/a`，Metrics 列标明计数读取或详细采集状态。

可以重新汇总已有结果。旧版未写入汇总的 TPOT 或吞吐量会从仍然存在的原始 bench JSON 中补齐，无需重新压测。

```bash
python bench.py report results/
```

原来的 Bash 入口也会调用通用工具，`ucm-seed` 和 `ucm-run` 分别映射到 `posix-seed`、`posix-run`。`BENCH_CONFIG` 可以指定配置文件，原有的 MODEL、SERVED_MODEL、HOST、PORT、SEED、BLOCK_TOKENS、RESULT_DIR 和 DRY_RUN 环境变量继续生效。

无请求预览可使用 `--dry-run`。`posix-run` 预览读取已有预埋记录。默认用完整 CLI 帮助检测参数。缺少 `--ready-check-timeout-sec` 时，通过当前 Python 环境中的同一个 vLLM CLI 入口，仅替换可识别的 `wait_for_endpoint` 试跑 hook，避免首条测量请求提前热缓存。未识别的实现会在发请求前报错。

## 实验锁

同一配置的实验不能重叠运行，避免 reset、预埋或 metrics 窗口互相干扰。锁由操作系统持有，正常结束、强制终止或机器重启后都会释放；磁盘上的 `.lock` 文件会保留，可直接再次运行，不需要删除。

旧版只写 PID 文件。新版发现旧锁中的 PID 已不存在时会自动接管；PID 仍存在时保守拒绝。此时用 `cat 报错中的完整锁路径` 查看 PID，再用 `ps -p PID -o pid,args` 确认对应测试是否仍在运行。只有确认该文件是旧版残留且原测试已结束，才删除这一份旧锁。不要删除正在运行的新版实验的锁文件。

## 本地检查

```bash
python -m unittest discover -s tests -v
```

测试使用模拟 bench 和 HTTP 服务，覆盖工作流与导出，真实 GPU/NPU 性能由远端服务验证。
