# vllmbench

通过 `vllm bench serve` 对比 HBM prefix cache、UCM Host Cache 和 UCM Posix 命中性能。Python 主程序及表格导出使用标准库，运行环境需要已有的 vLLM CLI，Python 版本要求 3.10 或更新。

## 快速使用

先修改 [bench.json](bench.json) 中的 `base_url`、`model`、`served_model`、`dp` 和 `cases`，在已安装 vLLM 的 Python 环境运行。

```bash
# 预览将要执行的命令
python bench.py cache bench.json --dry-run
# 正式运行并输出汇总表
python bench.py cache bench.json
# V4 公共前缀 HBM 测试
python bench.py hbm-shared-v4 bench.json
# 完整输入的裸推基线
python bench.py hbm-no-cache bench.json
```

一次命令执行配置中的全部 case，生成一个 run 目录。每个正式 case 在汇总表占一行，预埋和预热各自保存日志。

## 模式总览

| 模式 | 预埋范围与顺序 | 预埋请求数和并发 | 正式前的处理 |
| --- | --- | --- | --- |
| `hbm` | 每个 DP，逐 DP 执行 | 每个 DP 使用 case 完整请求数和并发 | 保留 HBM |
| `v4hbm` | 每个 DP，逐 DP 执行，使用 V4 长度规则 | 每个 DP 使用 case 完整请求数和并发 | 保留 HBM |
| `hbm-shared` | 每个 DP，并行启动 | 每个 DP 1 条、并发 1，所有请求共用前缀 | 保留 HBM |
| `hbm-shared-v4` | 每个 DP，逐 DP 执行，使用 V4 长度规则 | 每个 DP 1 条、并发 1，所有请求共用前缀 | 保留 HBM |
| `hbm-no-cache` | 直接发送正式组 | case 完整请求数和并发 | 直接测量 |
| `cache` | 每个 DP，并行启动 | 每个 DP 使用 case 完整请求数和并发 | `reset_external=false`，清 HBM |
| `posix-seed` | 完整集合只发送一次，内部 LB 分配 | case 完整请求数和并发 | 保存预埋记录，供重启后使用 |
| `posix-run` | 从预埋记录读取 seed | 正式组使用 case 请求数和并发 | 重启后先执行加载预热 |
| `posix-reset` | 完整集合只发送一次，内部 LB 分配 | case 完整请求数和并发 | `reset_external=true`，清 HBM 和 UCM Cache |

例如 DP4、40 请求、40 并发，`cache` 会同时向四个 DP 各发一组 40 条预埋请求；`posix-seed` 和 `posix-reset` 总共只预埋 40 条。正式组总共使用配置的 40 请求、40 并发。定向预埋通过 `X-data-parallel-rank` 指定 DP，`--dp-rank` 可将流程固定到一个 DP。

## 配置

修改仓库中的 `bench.json`，填写服务地址、模型、DP 数以及全部 case。每个 case 提供 `input_len`、`output_len`、`hit_rate`、`concurrency`、`num_requests`，长度单位均为 token，命中率范围为 0～1。`concurrency` 和 `num_requests` 均按整个客户端测量批次计算。

示例模型路径默认为 `/models/xxx`，运行时替换为实际模型目录。

每次执行默认生成随机 seed，每个 case 和预热组各用独立 seed。同一组预埋与正式测量复用 seed，随机长度波动固定为 0。普通模式两阶段使用相同请求数，公共前缀模式按正式组的请求数生成样本后取第一条预埋，保持公共前缀一致。配置中的整数 `seed` 可用于复现实验，默认也可显式写为 `"random"`。

UCM 预埋直接把 `--random-input-len` 设为目标命中长度，正式测量设为完整输入长度，两阶段均使用 `--dataset-name random --random-prefix-len 0`。例如 32K、90% 的预埋长度是 29491，正式长度是 32768。

可选配置字段如下。

```json
{
  "routing": {"mode": "auto"},
  "block_tokens": 1,
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

`block_tokens` 是可选的预埋长度对齐参数，默认值为 1。普通命中模式按它向下取整，服务端继续使用实际 KV block 粒度。`v4hbm` 和 `hbm-shared-v4` 固定使用 4096 对齐加 32 的规则。100% 目标会为尾部重算预留空间，实际比例由正式轮次指标给出。模型的最大上下文需要容纳输入加输出。

## HBM 命中

在开启 HBM prefix caching 的基线服务上执行一次。

```bash
python bench.py hbm bench.json
```

先在每个目标 DP 上执行独立的 8K 预热组，再依次处理正式 case。HBM、Cache、Posix 均使用每条请求各自的随机前缀，不设置公共前缀（`--random-prefix-len 0`）。HBM 的流程与 Cache 相同，只是不调用 reset：每个目标 DP 都预埋该 case 的完整请求集合，输入长度为目标命中部分（例如 90%），输出 1 token；随后保留 HBM 缓存，使用相同 seed、相同请求数发送全长正式请求。

`hbm` 和 `v4hbm` 的预埋在各 DP 上依次执行，每个 DP 都使用 case 配置的完整并发和请求数，不按 DP 数拆分。例如 40 请求、40 并发、2 DP，会先向 DP0 发 40 请求、并发 40，再向 DP1 发同样的 40 请求、并发 40。正式组仍使用 case 配置中的并发和请求数，默认走内部 LB。各 DP 的 HBM 需要容纳完整预埋集合，否则实际命中率可能低于目标值。Cache 的各 DP 预埋采用并行流程，Posix 将完整集合一次交给内部 LB 分发，详见下文。

## HBM 裸推性能

```bash
python bench.py hbm-no-cache bench.json
```

每个 case 直接发送完整正式请求，使用配置中的输入长度、输出长度、并发和请求数，公共前缀长度为 0。整个流程只运行正式组，预埋、独立预热及 reset 均跳过，配置中的 `ucm_warmup.enabled=true` 也会在该模式下关闭。每个 case 默认使用独立随机 seed，支持固定 seed 和 `--dp-rank`。

`hit_rate` 在该模式下可省略，使用已有配置时按 0 处理，汇总中的目标命中率和预埋长度均为 0。正式轮次的实际命中率、TTFT、TPOT、输出吞吐和总吞吐按原有方式采集，汇总 Mode 为 `hbm-no-cache`，支持 `--collect-metrics`。

裸推基线应使用关闭 prefix caching、停用 UCM 的服务配置。此模式发送完整正式请求，服务端已有的缓存配置仍然生效，实际命中情况由指标给出。

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

自动 LB 配置下，各 DP 的公共前缀预埋并行启动，每个预埋命令分别携带 `X-data-parallel-rank=0/1/...`，将请求指定到不同 DP。全部预埋成功完成后，再等待 `seed_settle_seconds` 与 `metrics.settle_seconds` 中的较大值，然后进入下一阶段；任一 DP 失败则停止该轮，不执行正式测试。正式 case 和预热组的预埋均采用此流程。`--dp-rank` 仍只预埋指定 DP。每个 DP 的命令、日志和结果分别保存在 `seed_dp<rank>.*`。`--dry-run` 按 rank 顺序打印这些命令。

例如 32K、90% 命中：预埋使用 `--random-prefix-len 29491 --random-input-len 1`，正式组使用 `--random-prefix-len 29491 --random-input-len 3277`，正式总输入仍为 32768。每个 case 和预热组使用独立 seed，新执行默认生成新 seed。

单请求预埋通过仓库内的 `bench_compat.py` 调用原有 vLLM CLI：按正式请求数生成随机样本以保持公共前缀的随机数状态一致，实际只发送第一条。不能直接把原生命令的 `--num-prompts` 改为 1，否则部分 vLLM 版本生成的公共前缀会改变。

默认独立预热也采用公共前缀，每个 DP 先预埋 1 条，再发送配置中的 8K 预热请求组，默认一轮。正式轮次的命中率、TTFT、TPOT 和吞吐继续自动汇总，表格 Mode 为 `hbm-shared`，与原有 `hbm` 区分。实际命中率仍以指标为准。Cache 和 Posix 继续采用原有的独立前缀方式。

## DeepSeek V4 MTP 公共前缀命中

```bash
python bench.py hbm-shared-v4 bench.json
```

结合公共前缀和 `v4hbm` 的 4096 向下取整再加 32 规则。每个 case 的所有请求共享 `seed_tokens` 长度的前缀，每个目标 DP 只预埋 1 条、并发 1、输出 1 token，全程不调用 reset。预埋和正式组使用相同 seed，并按正式请求数生成样本后只发送第一条预埋请求，以保持公共前缀一致。

各 DP 按 rank 顺序逐个预埋：DP0 成功完成后才发送 DP1，依此类推；每条请求仍携带对应的 rank header。全部成功后等待稳定时间，再进入下一阶段；任一 DP 失败即停止，不再向后续 DP 预埋，也不执行正式测试。正式 case 和预热组的预埋均采用此流程。`--dp-rank` 仍只覆盖指定 DP。

例如正式输入 32768、目标 90%，预期 HBM token 为 28672，公共前缀和预埋总输入均为 28704。预埋使用 `--random-prefix-len 28704 --random-input-len 0`；正式组使用 `--random-prefix-len 28704 --random-input-len 4064`，保留配置中的请求数、并发及输出长度。vLLM 的 random 数据集支持非空公共前缀加零长度独立部分。正式请求默认走内部 LB，也支持 `--dp-rank`。

默认一轮独立 8K 预热同样按该规则执行，公共前缀及预埋输入为 4128，预期 HBM token 为 4096。100% 和小输入的边界处理与 `v4hbm` 相同；取整为 0 时跳过预埋。汇总 Mode 为 `hbm-shared-v4`，并记录实际预埋长度及预期 HBM token。32K、90% 对应的预期 HBM 比例为 87.5%，实际命中率仍以正式组指标为准。此模式沿用已验证的 V4 边界规则，公共前缀组合尚需目标服务验证。

## UCM Host Cache 命中

在 UCM 服务上执行一次。

```bash
python bench.py cache bench.json
```

每个 case 的全部请求前缀会在每个 DP 都预埋一遍，然后调用 `POST /reset_prefix_cache?reset_external=false`，清理 HBM 前缀缓存，接着测量。每个 DP 的 Host Cache 需要容纳该 case 的完整预埋集合。reset 返回失败时自动重试至超时。

`cache` 同时启动各目标 DP 的预埋，每组用独立的 `X-data-parallel-rank` header 指定 DP，输入内容及 seed 相同，每个 DP 保留配置中的完整请求数和并发。例如 DP4、40 请求、40 并发，会同时向四个 DP 各发 40 条，全服务预埋并发最高 160；正式测试仍总共 40 请求、40 并发。预热组的预埋也并行。各 DP 全部成功后等待稳定时间，再执行 reset；任一 DP 失败则停止该轮。不同 case 依次执行，`--dp-rank` 只覆盖指定 DP。每个 DP 的结果与日志独立保存。

内部 LB 会把 reset 广播到全部 DP。本地核对的 vLLM 版本只返回第一个 DP 的 reset 结果，还应检查测量窗口中的实际 HBM 命中。

## UCM Posix 命中

第一条命令预埋全部 case，等待 UCM 写出完成后正常停服，保留 Posix 文件和相同的 key 配置并重启服务，再执行第二条命令。

各 DP 需要访问相同的 Posix 文件并使用兼容的 KV key/分片配置。每个 case 只发送完整请求集合一次，默认由内部 LB 分配到各 DP，保留配置中的完整请求数和并发，输出 1 token。例如 DP4、40 请求、40 并发，总共预埋 40 条。预热数据也只预埋一组并走内部 LB。自动路由的预埋命令省略 rank header，命令、日志和结果保存为 `seed_auto.*`。`--dp-rank` 或固定路由配置会改为在指定 DP 上预埋，保存为 `seed_dp<rank>.*`。失败的预埋不会发布为可用记录。重启后的加载预热覆盖各目标 DP，正式组默认走内部 LB。

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

一条命令自动执行独立加载预热，再对每个 case 将完整请求集合的目标前缀一次交给内部 LB 预埋（例如 90% 输入），使用配置中的完整请求数和并发。固定路由时使用指定 DP。等待 `seed_settle_seconds` 后调用下面的接口，成功后用相同 seed 和相同请求数发送全长正式请求：

```bash
curl -X POST 'http://localhost:18081/reset_prefix_cache?reset_external=true'
```

该接口需要清空 HBM 和 UCM 内存 Cache，保留后端 Posix 文件。reset 返回失败会重试至 `reset_timeout_seconds`，失败或超时则停止该轮，不执行正式测量。预埋等待时间应足够让 Posix 异步写入完成；该组不停止或重启服务，也不需要先跑 `posix-seed`。

Cache 和 `posix-reset` 模式都严格检查响应体：只有 `{"success": true}` 或 JSON 布尔值 `true` 才继续；明确返回 `false` 时重试。空 HTTP 200、无效 JSON、缺少布尔型 `success` 等情况立即报错，不能验证清理结果时不进入正式测试。旧服务需要修改 reset 接口以返回引擎清理结果。终端会打印 `Reset response: success=true/false (reset_external=...)`，方便核对；客户端仍受服务端 DP 返回值聚合范围限制。

预热组同样整组预埋一次并走内部 LB，再清 HBM 和 UCM Cache，然后向各目标 DP 执行加载预热。每个正式 case 的计数窗口从 reset 成功之后开始，预热和清理不计入正式性能结果。汇总表的 Mode 为 `posix-reset`，与停服重启的 `posix` 组区分，支持 `--dp-rank` 和 `--collect-metrics`。各 DP 需要访问相同的 Posix 文件并使用兼容的 KV key/分片配置。

HBM hit、External hit、Host Ready 和 Posix bytes 可用于核对加载来源；接口成功不等同于已验证 Posix 命中。此流程清的是 UCM 内存缓存，不清操作系统文件页缓存，因此不能视为与服务重启完全相同的底层 I/O 冷态。

## 自动 LB 和固定 DP

自动路由下，HBM 和 Cache 的预埋覆盖每个 DP，Posix 的预埋整组交给内部 LB 分配。正式请求也交给内部 LB 分配。固定 DP 可通过命令行选择。

```bash
python bench.py cache bench.json --dp-rank 0
```

也可在配置中写入 `"routing": {"mode": "fixed", "rank": 0}`。固定模式的预埋与测量都指向同一个 DP，汇总表保留其路由范围。

## 独立预热

HBM、Cache 和 Posix 命中测试都在全部正式 case 之前增加独立预热。沿用配置字段 `ucm_warmup`，默认在每个目标 DP 上执行 1 组，只预埋该组预热数据；每组 8K 输入、4 请求、4 并发、约 90% 前缀命中、输出 1 token。预热数据使用独立 seed，日志和 bench JSON 放在该轮的 `warmup_001/` 等目录中，正式汇总及 metrics 采集窗口从后续正式 case 开始。

```json
{
  "ucm_warmup": {
    "enabled": true,
    "rounds": 1,
    "input_len": 8192,
    "output_len": 1,
    "hit_rate": 0.9,
    "concurrency": 4,
    "num_requests": 4
  }
}
```

普通 HBM 模式先向每个目标 DP 预埋预热组全部独立前缀。公共前缀 HBM 模式每个 DP 预埋一条，顺序或并发方式见模式总览。随后各目标 DP 依次执行完整预热请求组。Cache 模式先并行完成各 DP 的预埋、reset HBM，再向每个 DP 发送预热加载请求。Posix 的 `posix-seed` 同时预埋预热数据并保存其参数与 seed，重启后的 `posix-run` 先从 Posix 加载这些独立前缀，然后直接进入正式测试。每组预热使用不同 seed。自动 LB 模式覆盖所有 DP，固定 DP 模式覆盖指定 DP，预热组命中阶段每次 bench 的并发为配置中的 `concurrency`。

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

同时输出输入吞吐 `Input tok/s` 和估算 Prefill 吞吐 `Prefill est tok/s`，单位均为 token/s。输入吞吐按总吞吐减输出吞吐计算。Prefill 估算值按成功请求的输入 token 总数除以成功请求的 TTFT 总和计算，TTFT 使用秒；也等于成功请求的平均输入长度除以平均 TTFT。它包含命中缓存的输入 token，以及 TTFT 中的排队和加载耗时，属于按请求聚合的估算口径。CSV/Excel 字段分别为 `input_tokens_per_s` 和 `prefill_tokens_per_s_estimated`。缺少所需数据或 TTFT 总和为 0 时估算值留空，终端显示 `n/a`。`report` 可从历史结果的原始 bench JSON 补算新列。

终端还显示 Target hit、HBM hit、External hit、Host Ready 和 Posix bytes。实际 0 命中显示 `0.0%`。缺少数据的列显示 `n/a`，Metrics 列标明计数读取或详细采集状态。

可以重新汇总已有结果。旧版未写入汇总的 TPOT 或吞吐量会从仍然存在的原始 bench JSON 中补齐，无需重新压测。

```bash
python bench.py report results/
```

原来的 Bash 入口也会调用通用工具，`ucm-seed` 和 `ucm-run` 分别映射到 `posix-seed`、`posix-run`。`BENCH_CONFIG` 可以指定配置文件，原有的 MODEL、SERVED_MODEL、HOST、PORT、SEED、BLOCK_TOKENS、RESULT_DIR 和 DRY_RUN 环境变量继续生效。

无请求预览可使用 `--dry-run`。`posix-run` 预览读取已有预埋记录。默认用完整 CLI 帮助检测参数。缺少 `--ready-check-timeout-sec` 时，通过当前 Python 环境中的同一个 vLLM CLI 入口，仅替换可识别的 `wait_for_endpoint` 试跑 hook，避免首条测量请求提前热缓存。未识别的实现会在发请求前报错。

## 指标口径与结果定位

| 列 | 计算口径 |
| --- | --- |
| `Output tok/s` | 成功请求的输出 token 总数 ÷ 正式测试整组耗时 |
| `Total tok/s` | 成功请求的输入加输出 token 总数 ÷ 正式测试整组耗时 |
| `Input tok/s` | 总吞吐减输出吞吐 |
| `Prefill est tok/s` | 成功请求的输入 token 总数 ÷ 成功请求的 TTFT 总和，TTFT 使用秒 |
| `HBM hit` | 正式窗口的 HBM hit token 增量 ÷ HBM query token 增量 |
| `External hit` | 正式窗口的 external hit token 增量 ÷ HBM query token 增量 |
| `Host Ready` | 加载时已经 Ready 的 Host Cache shard 数 ÷ 成功加载 shard 数 |
| `Posix bytes` | `ucm:posix_s2h_bytes_total` 在正式窗口内的增量，汇总各 worker，单位 bytes |

吞吐的输入 token 包含缓存命中的部分。`Prefill est tok/s` 的分母累计每条请求的 TTFT，包含排队和 KV 加载耗时。AISBench 工具的 Prefill 指标请在其原生性能报告中查看，两套工具的汇总列说明见各自 README。

`external_hit_rate_remaining` 另存在 CSV/Excel 中，分母为 external query token。`External hit` 包含 UCM Host Cache 和 Posix 的命中贡献，加载来源可结合 `Host Ready` 与 `Posix bytes` 判断。目标 90% 会受到预埋对齐、缓存驻留及实际请求复用影响，正式轮次输出实际命中率。

默认轻量计数读取也会填充 `Posix bytes`，服务端提供该计数器时可直接查看。真实零读取显示 `0`，指标缺失显示 `n/a`。

### 输出配置为 128，为什么有 1 token 的结果

| 文件或阶段 | 输出长度 |
| --- | --- |
| `seed_dp*.json`、`seed_auto.json` | 固定 1 token |
| `warmup_*/warmup_load_dp*.json` | 默认 1 token，使用预热配置 |
| 正式 case 下的 `measure.json` | 使用该 case 的 `output_len` |

正式请求包含 `--ignore-eos`，脚本逐条检查 `output_lens`。配置 128 而正式输出长度为 1 时，会记录 `unexpected output token counts` 并结束本轮。对应的 `measure.command.json` 和 `measure.log` 可用于核对正式命令与服务返回。

## 实验锁

同一配置的实验不能重叠运行，避免 reset、预埋或 metrics 窗口互相干扰。锁由操作系统持有，正常结束、强制终止或机器重启后都会释放；磁盘上的 `.lock` 文件会保留，可直接再次运行，不需要删除。

旧版只写 PID 文件。新版发现旧锁中的 PID 已不存在时会自动接管；PID 仍存在时保守拒绝。此时用 `cat 报错中的完整锁路径` 查看 PID，再用 `ps -p PID -o pid,args` 确认对应测试是否仍在运行。只有确认该文件是旧版残留且原测试已结束，才删除这一份旧锁。不要删除正在运行的新版实验的锁文件。

## 本地检查

```bash
python -m unittest discover -s tests -v
```

测试使用模拟 bench 和 HTTP 服务，覆盖工作流与导出，真实 GPU/NPU 性能由远端服务验证。
