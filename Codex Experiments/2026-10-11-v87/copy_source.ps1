param([ValidateSet('v86','v87')][string]$Version='v87',[switch]$CheckOnly)
$ErrorActionPreference='Stop'
try {
    $manifest=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'manifest.json') -Raw | ConvertFrom-Json
    $expected=if($Version -eq 'v86'){$manifest.base_sha256}else{$manifest.source_sha256}
    $path=Join-Path $PSScriptRoot ($Version+'/kernel.asc')
    if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $expected){throw 'Source integrity mismatch'}
    $text=[IO.File]::ReadAllText($path,[Text.Encoding]::UTF8)
    Write-Host ('Verified '+$Version+' '+(Get-Item -LiteralPath $path).Length+' bytes')
    if($CheckOnly){exit 0}
    Set-Clipboard -Value $text
    if((Get-Clipboard -Raw).Replace("`r`n","`n").TrimEnd("`r","`n") -cne $text.Replace("`r`n","`n").TrimEnd("`r","`n")){throw 'Clipboard mismatch'}
    Write-Host ('SUCCESS: complete '+$Version+' copied. Paste inside kernel.asc editor using Ctrl+A, Ctrl+V.') -ForegroundColor Green
    Write-Host 'V87 device validation and performance remain pending.'
}catch{Write-Host $_.Exception.Message -ForegroundColor Red;exit 1}
