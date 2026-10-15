param([ValidateSet('v82','v83','v84','v85')][string]$Version='v84',[switch]$CheckOnly)
$ErrorActionPreference='Stop'
try {
    $manifest=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'manifest.json') -Raw | ConvertFrom-Json
    $expected=if($Version -eq 'v82'){$manifest.base_sha256}else{$manifest.versions.$Version.sha256}
    $path=Join-Path $PSScriptRoot ($Version+'/kernel.asc')
    $hash=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if($hash -ne $expected){throw 'Source integrity check failed'}
    $text=[IO.File]::ReadAllText($path,[Text.Encoding]::UTF8)
    if(([regex]::Matches($text,'extern "C" void run_kernel\(')).Count -ne 1){throw 'Kernel entry missing or duplicated'}
    Write-Host ('Verified '+$Version+' / '+(Get-Item -LiteralPath $path).Length+' bytes / '+$hash)
    if($CheckOnly){exit 0}
    Set-Clipboard -Value $text
    $actual=(Get-Clipboard -Raw).Replace("`r`n","`n").TrimEnd("`r","`n")
    if($actual -cne $text.Replace("`r`n","`n").TrimEnd("`r","`n")){throw 'Clipboard verification failed'}
    Write-Host ('SUCCESS: complete '+$Version+' source copied.') -ForegroundColor Green
    Write-Host 'Click inside kernel.asc editor; Ctrl+A then Ctrl+V. Check the end before submitting.'
    Write-Host 'v82 passed submission 724207. v83/v84/v85 still require device evaluation.'
}catch{Write-Host $_.Exception.Message -ForegroundColor Red;exit 1}
