# 实现与使用方法

仓库包含六个 Python 包文件、一个测试文件、打包配置和拟议验收配置。四个功能模块分别处理数据审计、校准选择、名称策略和候选筛选；`cli.py` 提供数据分区审计与实测候选筛选两个命令。实现文件保留原代码包字节。

## 安装与入口

工具运行要求 Python 3.10 或更新版本，功能模块只使用标准库。`pyproject.toml` 使用 setuptools 打包。Windows PowerShell 中可按以下步骤安装：

```powershell
git clone https://github.com/GuoJianFei666/damo-yolo-aerial-detection.git
Set-Location damo-yolo-aerial-detection
py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe -m aerial_quant.cli --help
```

CLI 的任务名为 `audit-splits` 或 `select`，均要求 `--input` 与 `--output`。输出目录需要提前创建，程序不会创建父目录。

## 数据审计

[data_audit.py](../src/aerial_quant/data_audit.py) 包含三个主要接口：

| 接口 | 输入 | 输出 |
|---|---|---|
| `parse_annotation(text)` | VisDrone 八字段标注文本 | `Box` 列表；保留忽略区和其他类别，错误行抛出异常 |
| `image_summary(boxes, image_width, image_height, input_size=640, tiny_short_side=16.0)` | 标注框、原图尺寸与模型输入尺度 | 可评价目标、类别、遮挡、缩放后 tiny 目标及越界框统计 |
| `audit_splits(partitions)` | 五分区清单 | 分区数量、通过的检查及未覆盖项目；重叠或无效输入抛出异常 |

`audit_splits` 接收如下五个 JSON 键。每个键的值为非空记录列表，每条记录包括 `image_id`、64 位小写十六进制 `sha256` 和 `group_id`。哈希由实际图像字节计算，`group_id` 由数据整理者提供。

| 分区键 | 当前代码规则 |
|---|---|
| `train` | 训练图像清单 |
| `calibration` | 必须与 `train` 中对应图像的 ID、哈希及分组完全一致 |
| `select` | 用于候选选择 |
| `confirm` | 与选择阶段隔离的确认分区 |
| `test` | 最终测试分区 |

除训练与其校准子集外，各分区按 ID、精确图像哈希和给定分组互斥。当前接口不发现近重复图像，也不核实分组标注。项目当前研究说明采用官方训练、开发和最终测试流程，接入时需明确这些流程与旧五分区接口的对应关系。

准备真实清单 `partitions.json` 后运行：

```powershell
New-Item -ItemType Directory -Force reports | Out-Null
& .\.venv\Scripts\python.exe -m aerial_quant.cli audit-splits --input partitions.json --output reports/split-audit.json
```

审计通过会输出 `status`、`counts`、`checks` 和 `not_checked`。`image_summary` 中的 tiny 统计使用缩放后短边阈值，与标准 AP 指标分别记录。

## 覆盖校准选择

[calibration.py](../src/aerial_quant/calibration.py) 的接口为：

```python
select_coverage(rows, budget, minimum_coverage, max_per_group, seed=0)
```

候选记录包含 `image_id`、`group_id`、`split="train"` 和 `tags` 列表。`minimum_coverage` 给出各标签的最低覆盖数量，`max_per_group` 限制每个采集分组进入校准集的数量。函数返回选中的图像 ID。

算法每步选择覆盖缺口得分最高的可用样本，以种子和图像 ID 的哈希打破同分，直至达到预算。分组上限阻止凑足预算或最终覆盖不足时，函数抛出异常；贪心选择失败不等于约束必然无解。

在自己的数据处理脚本中，可读取候选及约束后调用现有接口：

```python
import json
from pathlib import Path
from aerial_quant.calibration import select_coverage

rows = json.loads(Path("calibration_candidates.json").read_text(encoding="utf-8"))
cfg = json.loads(Path("calibration_settings.json").read_text(encoding="utf-8"))
selected_ids = select_coverage(rows, **cfg)
Path("calibration_ids.json").write_text(
    json.dumps(selected_ids, ensure_ascii=False, indent=2), encoding="utf-8"
)
```

配置键对应 `budget`、`minimum_coverage`、`max_per_group` 和可选的 `seed`。本工具只有 Python 接口，未增加校准 CLI 子命令。

## 显式模块名称策略

[policy.py](../src/aerial_quant/policy.py) 使用模块名称清单组织部分量化策略：

1. `module_names_hash(available_names)` 对排序后的名称计算 SHA256。
2. 策略包含 `schema_version=1`、`module_names_sha256` 和显式 `quantize` 名称列表。
3. `resolve_policy(policy, available_names)` 或 `load_policy(path, available_names)` 检查版本、清单哈希、重复及未知名称，返回排序后的量化模块名。

空 `quantize` 列表可明确表示不量化模块的基线。`conv2d_constructor_kwargs` 保留卷积的 groups、dilation 等构造参数，供调用方构造兼容量化层。该模块不会导出 ONNX、生成 TensorRT 引擎或验证实际层精度。

## 从实测候选中筛选

[selection.py](../src/aerial_quant/selection.py) 的 `select_measured` 根据完整候选记录筛选：

| 字段 | 当前接口要求 |
|---|---|
| `candidate_id` | 候选唯一名称；默认基线名为 `fp16` |
| `data_origin` | CLI 使用 `measured_hardware`；构造数据只供测试显式放行 |
| `partition` | 必须为 `select` |
| `protocol_id`、`hardware_id` | 同一比较中一致的协议与设备标识 |
| `checkpoint_sha256`、`evaluation_manifest_sha256` | 同一比较中一致的模型和评价清单标识 |
| `engine_sha256` | 每个候选的非空引擎标识 |
| `timing_scope` | 必须为 `decode_excluded_e2e`，含预处理、传输、推理和后处理，不含视频解码 |
| `ap`、`ap_small` | 有限数值，范围为 0—100 |
| `group_recall` | 非空分组召回字典，各候选使用相同分组键，数值范围为 0—100 |
| `latency_ms` | 正有限时延样本，默认至少 1,000 个 |

输入文件 `measurements.json` 为上述候选记录的 JSON 列表，必须包含一个指定基线。程序计算各候选的 P95，并检查 AP、旧 `ap_small`、最差分组召回的退化及最低时延降幅，从满足约束的记录中选择 P95 最低者。

```powershell
& .\.venv\Scripts\python.exe -m aerial_quant.cli select --input measurements.json --output reports/selection.json --baseline fp16 --constraints configs/acceptance_proposed.json
```

返回 `candidate_found` 或 `no_feasible_candidate`，并列出逐候选拒绝原因。当前 `ap_small` 是原接口字段，尚未改为研究说明中的 `R_tiny`。AP 与时延需要调用方测量，工具不生成检测指标或硬件记录。

[acceptance_proposed.json](../configs/acceptance_proposed.json) 保留原拟议阈值：总体与 small AP 最大下降各 1.0 个百分点、分组召回最大下降 2.0 个百分点、最低时延降幅 0.15。这些数值是筛选配置。

## 测试源码与历史记录

[tests/test_research_kit.py](../tests/test_research_kit.py) 覆盖构造标注、分区交叉、覆盖约束、模块策略、候选筛选及卷积参数保留等情况。测试依赖 pytest 和 PyTorch，可在具备依赖的独立环境运行：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

[历史记录](../reports/historical-20261017/README.md) 保存原包的 2026-10-17 测试输出与环境。2026-10-22 本次发布只读取、复制和核对文件，未执行原项目、安装依赖或重新运行测试。
