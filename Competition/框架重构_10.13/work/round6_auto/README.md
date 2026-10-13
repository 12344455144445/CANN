> 当前状态：用户已停止自动测评。以下是归档技术资料；未经用户重新授权，不得执行比赛提交。新候选尚未经过 CANN 编译及 NPU 验证。

# 第六轮自动评测驱动

这是当前 Codex 会话调用的本地 Windows/Edge 驱动与离线证据工具。没有常驻服务、隐藏比赛 API、云端 NPU 或自动登录功能。UI 执行需要当前会话的桌面权限；登录失效、验证码或平台限流出现时保留现有记录并停止点击。

## 已实现

- `cli.py`：当前日全账号历史核对、预算与源码审查门禁、源证据绑定、15 点结果解析、统一参考耗时的估分及 CSV。年月日和时间统一为 `Asia/Shanghai`（UTC+8）；旧日期预算不会被重置成“其他人提交 0 次”。
- `run_ui.ps1`：`RefreshHistory`、`Submit`、`ReadResult` 三个入口，句柄、页面地址、按钮名称和坐标来自配置。提交前保存原源码，粘贴后逐字复核，在临时标签页二次读取全账号历史，再复核编辑器，最后写 `submission_attempt.json` 并只点击一次。
- `ui.py`：最小键鼠操作层，必须由已获取互斥锁的 PowerShell 驱动调用。
- `test_integrity.py`：独立临时目录中的完整性测试，不访问真实账号、浏览器或网络。

所有入口使用同一个 Windows 命名互斥锁 `Local\CANNJudge.SubmissionUI`。旧第五轮驱动没有此锁，因此不得和本驱动同时操作浏览器。本地锁无法约束其他电脑和其他不遵守此锁的进程；二次账号历史核对及预留预算用于降低队友并发的影响。

## 本机入口

`config.local.json` 保存此次已确认的本机参数，且不含账号 ID 或登录信息。重新启动浏览器后必须重新校准 `browser_hwnd`；界面布局变化后重新校准坐标。不可把示例中的零句柄直接用作真实配置。

```powershell
python -X utf8 work/round6_auto/test_integrity.py

powershell -NoProfile -ExecutionPolicy Bypass -File work/round6_auto/run_ui.ps1 -Action RefreshHistory -ConfigPath work/round6_auto/config.local.json

powershell -NoProfile -ExecutionPolicy Bypass -File work/round6_auto/run_ui.ps1 -Action Submit -ConfigPath work/round6_auto/config.local.json -CandidateFolder work/<新候选目录>

powershell -NoProfile -ExecutionPolicy Bypass -File work/round6_auto/run_ui.ps1 -Action ReadResult -ConfigPath work/round6_auto/config.local.json -CandidateFolder work/<同一候选目录>

python -X utf8 work/round6_auto/cli.py analyze --config work/round6_auto/config.local.json --candidate work/<控制目录> --candidate work/<候选目录> --baseline work/<控制目录> --output work/round6_auto/runtime/analysis.json
```

每次重复测评必须新建目录，复制冻结源码、完成对应审查并保留新目录自己的证据。已有 `submission_attempt.json` 或 `submission_info.json` 的目录不能再次点击提交。编译失败也计入当天预算。

## 候选审查格式

`kernel.asc` 应为 UTF-8、无 BOM、LF。`review_checks.json` 至少包含：

```json
{
  "review_pass": true,
  "sha256": "当前 kernel.asc 原始字节的 SHA256",
  "checks": {"precision": true, "memory": true, "dispatch": true}
}
```

`required_review_checks` 可指定本轮新框架必须通过的审查名；默认空列表兼容既有审查字段。此驱动不会用第五轮 shape audit 对新框架进行错误的结构断言；精度、内存、形状分派及并发正确性必须由候选专用审查工具完成并如实写入 review。`review_pass=true` 是审查产物，不等于驱动自行完成了源码语义证明。

候选目录名和 `quarantine_sha256` 同时隔离已知不安全源码，改名不能清除已知问题。`post_submission_finding` 或 `quarantined` 会阻止提交及推荐。

## 预算及历史证据

历史从第一页连续读取到前一天，或到有明确总条数证明的最后一页。每次扫描使用新的 UUID 目录，拒绝混入旧页。若分页期间总条数发生变化、日期错误、页序错误、时间顺序不降序、边界不完整或观察超过默认 60 秒，则阻止提交并要求重新扫描。

有真实提交 ID 时按 ID 去重；当前网页复制文本未显示 ID，则按唯一提交时间核对。无 ID 的同秒多行不能据此判为同一次，也不能盲计为多次；自动流程会停止，需要先从真实页面采集行 ID/结果路由，再补全 `entries`。一页可以保存 `entries:[{"id":"页面数字ID","url":"真实独立结果URL","submitted_at":"2026-10-13T12:00:00+08:00"}]`。网页数字 ID 与 URL 的 24 位 Mongo 路由 token 是不同标识，分别保存并核对。

当前配置使用历史工作策略每日 50 次、预留 10 次，现行提交页没有明示该上限，因此 `platform_limit_verified=false`。有出处、当天日期和开始时刻的 `limit_assumption` 将此次探索限制为最多 4 次本机提交；不会伪称上限已验证。先完成控制和候选小规模试验，发现平台限额/频率提示时立即停止。进一步扩展需更新有依据的预算策略和实际账号历史。

提交间隔 180 秒是本工作流的保守策略，不是对平台规则的声明。等待后会重新读取账号历史，且在真正点击前再做门禁。任何日期的“已开始提交但没有独立 URL”记录都继续阻止后续提交，不能跨日遗忘。只在有明确证据且已经审阅时才允许将失败前置步骤记录为 `not_submitted_verified`，并附 `resolution:{"reviewed":true,"evidence":"证据说明"}`；不能用此状态清除已经可能发生的点击。

## 结果判断与恢复

推荐资格同时需要：源码 SHA 与审查 SHA 一致、网页复制源码与当前源码一致、结果 URL 与保存的提交 URL 一致、页面数字 ID 与已捕获 ID 一致（旧记录没有捕获数字 ID 时保留兼容）、状态 Pass、唯一顺序点号 1..15、每点 Pass、错误率 0、耗时与参考耗时为正。

`Running 0/0` 和不完整 Pass 页面继续轮询。超时后保留 URL，下次仅恢复 `ReadResult`。若点击后没有 URL，保存未决记录并核对真实历史；不可自动重试提交。原始网页文本、提交链接和历史证据只保存于本地 `runtime/` 和候选目录，命令输出省略账号和原始网页信息。

性能分数来自统一控制参考耗时的公式估算，不能冒充队伍实时排名或保证稳定提升。先重复成对测量，观察各测试点和噪声，再决定推荐源码。此驱动不负责 GitHub 上传、合并 PR 或替换默认分支源码。
