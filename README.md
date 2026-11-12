# vllmbench

通过 `vllm bench serve` 对比 HBM prefix cache、UCM Host Cache 和 UCM Posix 命中性能。Python 主程序及表格导出使用标准库，运行环境需要已有的 vLLM CLI，Python 版本要求 3.10 或更新。

## 配置

修改仓库中的 `bench.json`，填写服务地址、模型、DP 数以及全部 case。每个 case 提供 `input_len`、`output_len`、`hit_rate`、`concurrency`、`num_requests`，长度单位均为 token，命中率范围为 0～1。`concurrency` 和 `num_requests` 均按整个客户端测量批次计算。

示例模型路径默认为 `/models/xxx`，运行时替换为实际模型目录。

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

先在每个目标 DP 上执行独立的 8K 预热组，再依次处理正式 case。HBM、Cache、Posix 均使用每条请求各自的随机前缀，不设置公共前缀（`--random-prefix-len 0`）。HBM 的流程与 Cache 相同，只是不调用 reset：每个目标 DP 都预埋该 case 的完整请求集合，输入长度为目标命中部分（例如 90%），输出 1 token；随后保留 HBM 缓存，使用相同 seed、相同请求数发送全长正式请求。

上述非公共前缀模式的预埋在各 DP 上依次执行，每个 DP 都使用 case 配置的完整并发和请求数，不按 DP 数拆分。例如 40 请求、40 并发、2 DP，会先向 DP0 发 40 请求、并发 40，再向 DP1 发同样的 40 请求、并发 40。正式组仍使用 case 配置中的并发和请求数，默认走内部 LB。各 DP 的 HBM 需要容纳完整预埋集合，否则实际命中率可能低于目标值。

## DeepSeek V4 MTP HBM 命中

使用同一份 `bench.json` 执行全部 case。

```bash
python bench.py v4hbm bench.json
```

`v4hbm` 用于本次验证的 V4 MTP/DSpark 配置，HBM 边界为 4096 token，尾块为 32 token。先将 `input_len × hit_rate` 按 4096 向下取整，得到 `expected_hbm_tokens`，再加 32 得到实际预埋长度 `seed_tokens`。此模式固定使用上述粒度。

| 正式输入 | 目标比例 | 预期 HBM token | 实际预埋输入 |
|---|---|---|---|
| 32768 | 90% | 28672 | 28704 |
| 131072 | 90% | 114688 | 114720 |
| 8192（默认预热） | 90% | 4096 | 4128 |

每个目标 DP 依次预埋全部独立前缀，使用配置中的完整请求数和并发，输出 1 token。正式请求使用相同 seed 和请求数，按配置的全长输入、输出及并发执行，默认走内部 LB，也支持 `--dp-rank`。独立预热采用相同的取整加 32 规则，默认一轮。

100% 等目标若加 32 后超过正式输入，会先将 HBM 目标降至可容纳尾块的最大 4096 边界。取整结果为 0 时跳过预埋。例如正式输入 32768、目标 100%，实际预埋仍为 28704，预期 HBM 比例为 87.5%。

manifest、case_result.json 及 CSV/XLSX 汇总记录 `seed_tokens` 和 `expected_hbm_tokens`，终端表格 Mode 为 `v4hbm`。32K、90% 的预期 HBM 比例为 87.5%，实际命中率由正式轮次指标给出。各 DP 需要容纳完整预埋集合。本模式针对当前 4096/32 配置，已实测单前缀逐 DP 命中，完整并发集合的硬件结果仍需按实际容量验证。

## HBM 公共前缀命中

使用相同的 `bench.json`，一次执行其中全部 case：

```bash
python bench.py hbm-shared bench.json
```

每个 case 的所有正式请求共享一条随机前缀，长度为 `seed_tokens`，即输入长度乘目标命中率，再按 `block_tokens` 向下取整。每个目标 DP 只预埋 1 条请求、并发 1、输出 1 token，输入为公共前缀加 1 token；正式组保留配置的完整输入长度、输出长度、并发和请求数，默认走内部 LB。全程不调用 reset。支持带 UCM 和不带 UCM 的服务，以及 `--dp-rank` 固定 DP。

例如 32K、90% 命中：预埋使用 `--random-prefix-len 29491 --random-input-len 1`，正式组使用 `--random-prefix-len 29491 --random-input-len 3277`，正式总输入仍为 32768。每个 case 和预热组使用独立 seed，新执行默认生成新 seed。

单请求预埋通过仓库内的 `bench_compat.py` 调用原有 vLLM CLI：按正式请求数生成随机样本以保持公共前缀的随机数状态一致，实际只发送第一条。不能直接把原生命令的 `--num-prompts` 改为 1，否则部分 vLLM 版本生成的公共前缀会改变。

默认独立预热也采用公共前缀，每个 DP 先预埋 1 条，再发送配置中的 8K 预热请求组，默认一轮。正式轮次的命中率、TTFT、TPOT 和吞吐继续自动汇总，表格 Mode 为 `hbm-shared`，与原有 `hbm` 区分。实际命中率仍以指标为准。Cache 和 Posix 继续采用原有的独立前缀方式。

## DeepSeek V4 MTP 公共前缀命中

```bash
python bench.py hbm-shared-v4 bench.json
```

结合公共前缀和 `v4hbm` 的 4096 向下取整再加 32 规则。每个 case 的所有请求共享 `seed_tokens` 长度的前缀，每个目标 DP 只预埋 1 条、并发 1、输出 1 token，全程不调用 reset。预埋和正式组使用相同 seed，并按正式请求数生成样本后只发送第一条预埋请求，以保持公共前缀一致。

例如正式输入 32768、目标 90%，预期 HBM token 为 28672，公共前缀和预埋总输入均为 28704。预埋使用 `--random-prefix-len 28704 --random-input-len 0`；正式组使用 `--random-prefix-len 28704 --random-input-len 4064`，保留配置中的请求数、并发及输出长度。vLLM 的 random 数据集支持非空公共前缀加零长度独立部分。正式请求默认走内部 LB，也支持 `--dp-rank`。

默认一轮独立 8K 预热同样按该规则执行，公共前缀及预埋输入为 4128，预期 HBM token 为 4096。100% 和小输入的边界处理与 `v4hbm` 相同；取整为 0 时跳过预埋。汇总 Mode 为 `hbm-shared-v4`，并记录实际预埋长度及预期 HBM token。32K、90% 对应的预期 HBM 比例为 87.5%，实际命中率仍以正式组指标为准。此模式沿用已验证的 V4 边界规则，公共前缀组合尚需目标服务验证。

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

Cache 和 `posix-reset` 模式都严格检查响应体：只有 `{"success": true}` 或 JSON 布尔值 `true` 才继续；明确返回 `false` 时重试。空 HTTP 200、无效 JSON、缺少布尔型 `success` 等情况立即报错，不能验证清理结果时不进入正式测试。旧服务需要修改 reset 接口以返回引擎清理结果。终端会打印 `Reset response: success=true/false (reset_external=...)`，方便核对；客户端仍受服务端 DP 返回值聚合范围限制。

预热组同样先预埋，再清 HBM 和 UCM Cache，然后执行加载预热。每个正式 case 的计数窗口从 reset 成功之后开始，预热和清理不计入正式性能结果。汇总表的 Mode 为 `posix-reset`，与停服重启的 `posix` 组区分；支持默认全 DP 预埋加自动 LB、`--dp-rank` 和 `--collect-metrics`。

HBM hit、External hit、Host Ready 和 Posix bytes 可用于核对加载来源；接口成功不等同于已验证 Posix 命中。此流程清的是 UCM 内存缓存，不清操作系统文件页缓存，因此不能视为与服务重启完全相同的底层 I/O 冷态。

## 自动 LB 和固定 DP

默认模式下，预埋覆盖每个 DP，正式请求交给内部 LB 分配。固定 DP 可通过命令行选择。

```bash
python bench.py cache bench.json --dp-rank 0
```

也可在配置中写入 `"routing": {"mode": "fixed", "rank": 0}`。固定模式的预埋与测量都指向同一个 DP，汇总表保留其路由范围。

## 独立预热

HBM、Cache 和 Posix 命中测试都在全部正式 case 之前增加独立预热。沿用配置字段 `ucm_warmup`，默认在每个目标 DP 上执行 1 组，只预埋该组预热数据；每组 8K 输入、4 请求、4 并发、约 90% 前缀命中、输出 1 token。预热数据使用独立 seed，日志和 bench JSON 放在该轮的 `warmup_001/` 等目录中，正式汇总及 metrics 采集窗口从后续正式 case 开始。

```json
"ucm_warmup": {
  "enabled": true,
  "rounds": 1,
  "input_len": 8192,
  "output_len": 1,
  "hit_rate": 0.9,
  "concurrency": 4,
  "num_requests": 4
}
```

HBM 模式先向每个目标 DP 预埋预热组全部请求的独立前缀，再发送全长的预热组请求命中 HBM，不调用 reset。Cache 模式先预埋预热前缀、reset HBM，再向每个 DP 发送预热加载请求。Posix 的 `posix-seed` 同时预埋预热数据并保存其参数与 seed，重启后的 `posix-run` 先从 Posix 加载这些独立前缀，然后直接进入正式测试。每组预热使用不同 seed。自动 LB 模式覆盖所有 DP，固定 DP 模式覆盖指定 DP，预热组命中阶段每次 bench 的并发为配置中的 `concurrency`。

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

终端的 `Date (CST)` 列显示每轮开始时间，使用北京时间，格式为 `月-日 时:分`（例如 `11-07 14:35`），从原始 run ID 的 UTC 时间转换，重新汇总不会改成当前时间。终端的 `Model` 列和 CSV、Excel 的 `served_model` 列使用本轮 `bench.json` 的 `served_model`。重新汇总旧结果时，会从该轮保存的 `manifest.json` 恢复模型名称。

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
