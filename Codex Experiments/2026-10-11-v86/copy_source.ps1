param([ValidateSet('v84','v86')][string]$Version='v86',[switch]$CheckOnly)
$ErrorActionPreference='Stop'
try {
    $manifest=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'manifest.json') -Raw | ConvertFrom-Json
    $expected=if($Version -eq 'v84'){$manifest.base_sha256}else{$manifest.source_sha256}
    $path=Join-Path $PSScriptRoot ($Version+'/kernel.asc')
    if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $expected){throw 'Source integrity mismatch'}
    $text=[IO.File]::ReadAllText($path,[Text.Encoding]::UTF8)
    if(([regex]::Matches($text,'extern "C" void run_kernel\(')).Count -ne 1){throw 'Entry contract missing'}
    Write-Host ('Verified '+$Version+' '+(Get-Item -LiteralPath $path).Length+' bytes')
    if($CheckOnly){exit 0}
    Set-Clipboard -Value $text
    if((Get-Clipboard -Raw).Replace("`r`n","`n").TrimEnd("`r","`n") -cne $text.Replace("`r`n","`n").TrimEnd("`r","`n")){throw 'Clipboard mismatch'}
    Write-Host ('SUCCESS: complete '+$Version+' copied. Paste inside kernel.asc editor using Ctrl+A, Ctrl+V.') -ForegroundColor Green
    Write-Host 'V86 device compilation/correctness/performance are pending.'
}catch{Write-Host $_.Exception.Message -ForegroundColor Red;exit 1}
