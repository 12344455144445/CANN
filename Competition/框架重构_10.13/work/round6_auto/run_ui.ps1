param(
    [Parameter(Mandatory=$true)][ValidateSet('RefreshHistory','Submit','ReadResult')][string]$Action,
    [Parameter(Mandatory=$true)][string]$ConfigPath,
    [string]$CandidateFolder
)
$ErrorActionPreference = 'Stop'
$taskCfgFile = (Resolve-Path -LiteralPath $ConfigPath).Path
$taskCfg = Get-Content -LiteralPath $taskCfgFile -Raw -Encoding utf8 | ConvertFrom-Json
$taskPython = 'python'
$taskCli = Join-Path $PSScriptRoot 'cli.py'
$taskUi = Join-Path $PSScriptRoot 'ui.py'
$taskRuntime = Join-Path $PSScriptRoot 'runtime'
$taskBeijingZone = [TimeZoneInfo]::FindSystemTimeZoneById('China Standard Time')
New-Item -ItemType Directory -Path $taskRuntime -Force | Out-Null
if (-not $taskCfg.browser_hwnd -or -not $taskCfg.ui_layout_verified) { throw 'Configure and verify browser HWND and UI layout before desktop automation' }
if ($Action -ne 'RefreshHistory') {
    if (-not $CandidateFolder) { throw 'CandidateFolder required' }
    $taskCandidate = (Resolve-Path -LiteralPath $CandidateFolder).Path
}
Add-Type -AssemblyName UIAutomationClient
$taskWindow = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$taskCfg.browser_hwnd)
if ($taskWindow.Current.Name -notlike ('*' + $taskCfg.browser_title_contains + '*')) { throw 'Configured window title does not match the intended browser' }

function Task-Now { return [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $taskBeijingZone).ToString("yyyy-MM-ddTHH:mm:ss") + '+08:00' }
function Task-WriteJson([string]$Path, $Value) {
    $taskJson = ConvertTo-Json -InputObject $Value -Depth 20
    $taskTemp = $Path + '.new'
    [IO.File]::WriteAllText($taskTemp, $taskJson, (New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $taskTemp -Destination $Path -Force
}
function Task-Ui([string]$Command, [string[]]$Arguments=@()) {
    & $taskPython -X utf8 $taskUi --hwnd $taskCfg.browser_hwnd --expected-title $taskCfg.browser_title_contains $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('Desktop helper failed: ' + $Command) }
}
function Task-Url {
    Task-Ui 'copyurl'
    return (Get-Clipboard -Raw).Trim()
}
function Task-AssertUrl([string]$Expected) {
    if ((Task-Url) -cne $Expected) { throw 'Browser address is not the intended page; no submit allowed' }
}
function Task-CopyPage([object[]]$Point) {
    Task-Ui 'click' @([string]$Point[0], [string]$Point[1])
    Task-Ui 'copyall'
    return Get-Clipboard -Raw
}
function Task-Find([string]$Name) {
    return $taskWindow.FindFirst([System.Windows.Automation.TreeScope]::Descendants,
        (New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::NameProperty, $Name)))
}
function Task-Invoke($Element) {
    if ($null -eq $Element -or -not $Element.Current.IsEnabled) { throw 'Required UI element is missing or disabled' }
    $taskPattern = $null
    if ($Element.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$taskPattern)) {
        $taskPattern.Invoke()
    } else {
        $taskRect = $Element.Current.BoundingRectangle
        if ($taskRect.IsEmpty -or $taskRect.Width -lt 1 -or $taskRect.Height -lt 1) { throw 'UI element has no usable rectangle' }
        Task-Ui 'click' @([string][int]($taskRect.X+$taskRect.Width/2), [string][int]($taskRect.Y+$taskRect.Height/2))
    }
}
function Task-RefreshHistory([bool]$NewTab=$false) {
    $taskScanId = [Guid]::NewGuid().ToString('N')
    $taskScanFolder = Join-Path $taskRuntime ('history_' + $taskScanId)
    New-Item -ItemType Directory -Path $taskScanFolder | Out-Null
    $taskToday = (Task-Now).Substring(0,10)
    $taskRowsSeen = 0
    $taskBoundary = $false
    $taskOpened = $false
    try {
        for ($taskPage=1; $taskPage -le $taskCfg.max_history_pages; $taskPage++) {
            $taskHistoryUrl = $taskCfg.history_url_template.Replace('{page}', [string]$taskPage)
            if ($NewTab -and $taskPage -eq 1) { Task-Ui 'newtab' @($taskHistoryUrl); $taskOpened=$true }
            else { Task-Ui 'navigate' @($taskHistoryUrl) }
            Task-AssertUrl $taskHistoryUrl
            $taskHistoryText = $null
            for ($taskTry=0; $taskTry -lt 8; $taskTry++) {
                $taskHistoryText = Task-CopyPage $taskCfg.history_focus
                if ($taskHistoryText -match [regex]::Escape($taskCfg.history_ready_text)) { break }
                Start-Sleep -Seconds 1
            }
            if ($taskHistoryText -notmatch [regex]::Escape($taskCfg.history_ready_text)) { throw 'History table is not loaded; no account budget assumption allowed' }
            $taskTimes = [regex]::Matches($taskHistoryText, '\d{4}[/-]\d{2}[/-]\d{2} \d{2}:\d{2}:\d{2}')
            $taskRowsSeen += $taskTimes.Count
            foreach ($taskTime in $taskTimes) {
                if ($taskTime.Value.Substring(0,10).Replace('/','-') -lt $taskToday) { $taskBoundary=$true }
            }
            $taskTotal = [regex]::Match($taskHistoryText, '共\s*(\d+)\s*条')
            $taskTotalValue = $null
            if ($taskTotal.Success) { $taskTotalValue = [int]$taskTotal.Groups[1].Value }
            $taskPageRecord = @{scan_id=$taskScanId;page=$taskPage;url=$taskHistoryUrl;observed_at=(Task-Now);text=$taskHistoryText}
            if ($null -ne $taskTotalValue) { $taskPageRecord.total_count=$taskTotalValue }
            Task-WriteJson (Join-Path $taskScanFolder ('page'+$taskPage+'.json')) $taskPageRecord
            if ($taskBoundary -or ($null -ne $taskTotalValue -and $taskRowsSeen -eq $taskTotalValue)) { break }
        }
        $taskHistoryOutput = Join-Path $taskScanFolder 'complete_history.json'
        & $taskPython -X utf8 $taskCli history --pages $taskScanFolder --scan-id $taskScanId --page-size $taskCfg.history_page_size --output $taskHistoryOutput | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Account history is incomplete or ambiguous; automatic submission blocked' }
        return $taskHistoryOutput
    } finally {
        if ($taskOpened) {
            $taskCurrent = Task-Url
            if ($taskCurrent.StartsWith('https://cannjudge.cn/me/submissions?')) { Task-Ui 'closetab' }
            else { throw 'Unexpected history-tab address; leave it visible for inspection' }
        }
    }
}
function Task-Gate([string]$History) {
    $taskGateFile = Join-Path $taskCandidate 'budget_gate.json'
    & $taskPython -X utf8 $taskCli gate --config $taskCfgFile --candidate $taskCandidate --history $History --output $taskGateFile | Out-Host
    if ($LASTEXITCODE -notin @(0,3)) { throw 'Source evidence or account budget gate failed' }
    return Get-Content -LiteralPath $taskGateFile -Raw -Encoding utf8 | ConvertFrom-Json
}

# The mutex name is shared across configurations, candidates, browser windows and local Codex processes.
# Old round5 scripts do not acquire it: no other UI driver may run in parallel.
$taskMutex = New-Object Threading.Mutex($false, 'Local\CANNJudge.SubmissionUI')
$taskLocked = $false
try {
    try { $taskLocked=$taskMutex.WaitOne(0) }
    catch [Threading.AbandonedMutexException] { $taskLocked=$true }
    if (-not $taskLocked) { throw 'Another local process owns the CANNJudge browser; retry only after it finishes' }
    if ($Action -eq 'RefreshHistory') {
        $taskHistory = Task-RefreshHistory
        Write-Output ('Complete history saved locally: ' + (Split-Path -Leaf (Split-Path -Parent $taskHistory)))
    } elseif ($Action -eq 'Submit') {
        if (Test-Path -LiteralPath (Join-Path $taskCandidate 'submission_attempt.json')) { throw 'Existing submit attempt; reconcile instead of retrying' }
        # First full scan handles new-day/team-budget changes and spacing before editing.
        $taskHistory = Task-RefreshHistory
        $taskGate = Task-Gate $taskHistory
        while ($taskGate.minimum_wait_seconds -gt 0) {
            Write-Output ('Keeping local policy spacing: ' + $taskGate.minimum_wait_seconds + ' seconds')
            Start-Sleep -Seconds ([Math]::Min(30, $taskGate.minimum_wait_seconds))
            $taskHistory = Task-RefreshHistory
            $taskGate = Task-Gate $taskHistory
        }
        Task-Ui 'navigate' @($taskCfg.submit_url)
        Task-AssertUrl $taskCfg.submit_url
        $taskTab = $null
        for ($taskTry=0; $taskTry -lt 10 -and $null -eq $taskTab; $taskTry++) {
            $taskTab = Task-Find $taskCfg.kernel_tab_name
            if ($null -eq $taskTab) { Start-Sleep -Seconds 1 }
        }
        Task-Invoke $taskTab
        Start-Sleep -Seconds 2
        $taskOriginal = Task-CopyPage $taskCfg.editor_focus
        [IO.File]::WriteAllText((Join-Path $taskCandidate 'pre_edit_source.asc'), $taskOriginal, (New-Object Text.UTF8Encoding($false)))
        $taskExpected = [IO.File]::ReadAllText((Join-Path $taskCandidate 'kernel.asc'), [Text.Encoding]::UTF8)
        $taskClipboardReady=$false
        for ($taskTry=0; $taskTry -lt 5; $taskTry++) {
            try { Set-Clipboard -Value $taskExpected; $taskClipboardReady=$true; break }
            catch { if ($taskTry -eq 4) { throw }; Start-Sleep -Milliseconds 300 }
        }
        if (-not $taskClipboardReady) { throw 'Clipboard unavailable; no submit sent' }
        Task-Ui 'replaceall'
        $taskActual = Task-CopyPage $taskCfg.editor_focus
        if ($taskActual.Replace("`r`n","`n") -cne $taskExpected) { throw 'Browser source differs from reviewed candidate; no submit sent' }
        # Reconcile again after editing and waiting. A temporary history tab preserves the source editor.
        $taskHistory = Task-RefreshHistory $true
        Task-AssertUrl $taskCfg.submit_url
        $taskActual = Task-CopyPage $taskCfg.editor_focus
        if ($taskActual.Replace("`r`n","`n") -cne $taskExpected) { throw 'Source changed while refreshing budget; no submit sent' }
        [IO.File]::WriteAllText((Join-Path $taskCandidate 'web_verified.asc'), $taskActual, (New-Object Text.UTF8Encoding($false)))
        $taskGate = Task-Gate $taskHistory
        if ($taskGate.minimum_wait_seconds -gt 0) { throw 'Spacing changed; no submit sent' }
        $taskButton=$null
        foreach ($taskName in $taskCfg.submit_button_names) { $taskButton=Task-Find $taskName; if ($null -ne $taskButton) { break } }
        if ($null -eq $taskButton) { throw 'Submit button not found; no submit sent' }
        $taskAttempt = @{state='submit_invocation_started';started_at=(Task-Now);source_sha256=$taskGate.source_sha256}
        Task-WriteJson (Join-Path $taskCandidate 'submission_attempt.json') $taskAttempt
        Task-Invoke $taskButton
        $taskSubmissionUrl=$null
        for ($taskTry=0; $taskTry -lt 15; $taskTry++) {
            Start-Sleep -Seconds 1
            $taskCurrent=Task-Url
            if ($taskCurrent -match '^https://cannjudge\.cn/(?:.*/)?submission/(?:[0-9a-f]{24}|\d+)(?:[/?#]|$)') { $taskSubmissionUrl=$taskCurrent; break }
        }
        if (-not $taskSubmissionUrl) { throw 'Submit clicked but no independent URL captured; preserve attempt and reconcile visibly; never click Submit again blindly' }
        $taskCapturedInfo=@{url=$taskSubmissionUrl;captured_at=(Task-Now);source_sha256=$taskGate.source_sha256}
        # Route token and numeric display ID are independent. Capture the latter when the page has loaded.
        $taskCaptureText=Task-CopyPage $taskCfg.result_focus
        Task-AssertUrl $taskSubmissionUrl
        $taskDisplayId=[regex]::Match($taskCaptureText, '(?:提交\s*ID|Submission ID)\s*[:：]\s*(\d+)')
        if ($taskDisplayId.Success) { $taskCapturedInfo.submission_id=$taskDisplayId.Groups[1].Value }
        Task-WriteJson (Join-Path $taskCandidate 'submission_info.json') $taskCapturedInfo
        $taskAttempt.state='url_captured';$taskAttempt.url=$taskSubmissionUrl
        Task-WriteJson (Join-Path $taskCandidate 'submission_attempt.json') $taskAttempt
        Write-Output 'Source bytes verified; one submission URL captured and stored locally.'
    } else {
        $taskInfo = Get-Content -LiteralPath (Join-Path $taskCandidate 'submission_info.json') -Raw -Encoding utf8 | ConvertFrom-Json
        if ($taskInfo.url -notmatch '^https://cannjudge\.cn/(?:.*/)?submission/(?:[0-9a-f]{24}|\d+)(?:[/?#]|$)') { throw 'Stored submission URL is not a live validated CANNJudge result route' }
        Task-Ui 'navigate' @($taskInfo.url)
        $taskDeadline=[DateTime]::UtcNow.AddSeconds($taskCfg.result_timeout_seconds)
        do {
            Task-AssertUrl $taskInfo.url
            $taskResultText=Task-CopyPage $taskCfg.result_focus
            if ($taskResultText -notmatch '提交\s*ID|Submission ID') { throw 'Copied page is not submission evidence; inspect the visible page' }
            Task-WriteJson (Join-Path $taskCandidate 'platform_result.json') @{url=$taskInfo.url;observed_at=(Task-Now);text=$taskResultText}
            & $taskPython -X utf8 $taskCli result --config $taskCfgFile --candidate $taskCandidate | Out-Host
            if ($LASTEXITCODE -eq 0) { break }
            if ($LASTEXITCODE -ne 4) { throw 'Result evidence parser failed' }
            if ([DateTime]::UtcNow -ge $taskDeadline) { throw 'Result timeout; keep captured URL and resume read later, do not resubmit this folder' }
            Start-Sleep -Seconds ([Math]::Min(30, $taskCfg.result_poll_seconds))
            Task-Ui 'refresh'
        } while ($true)
    }
} finally {
    if ($taskLocked) { $taskMutex.ReleaseMutex() }
    $taskMutex.Dispose()
}
