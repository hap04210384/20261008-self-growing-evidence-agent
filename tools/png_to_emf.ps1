# png_to_emf.ps1 —— 用 PowerPoint COM 把 PNG 导出为 Word 兼容的 EMF
# 用法: powershell -ExecutionPolicy Bypass -File png_to_emf.ps1 <输入.png> <输出目录>
# 产出: <输出目录>/幻灯片1.EMF（单页幻灯片与图同尺寸，图铺满页面）
# 背景: LibreOffice --convert-to emf 产出的 EMF 会让 Word 打开文档时崩溃
#       （2026-10-07 实测），Office 自家导出的 EMF 无此问题。
param(
    [Parameter(Mandatory = $true)][string]$PngPath,
    [Parameter(Mandatory = $true)][string]$OutDir
)
$job = Start-Job -ScriptBlock {
    param($png, $outDir)
    Add-Type -AssemblyName System.Drawing
    $img = [System.Drawing.Image]::FromFile($png)
    $wPt = $img.Width * 0.75   # px -> pt (96dpi)
    $hPt = $img.Height * 0.75
    $img.Dispose()
    $ppt = New-Object -ComObject PowerPoint.Application
    $pres = $ppt.Presentations.Add()
    $pres.PageSetup.SlideWidth = $wPt
    $pres.PageSetup.SlideHeight = $hPt
    $slide = $pres.Slides.Add(1, 12)
    $slide.Shapes.AddPicture($png, $false, $true, 0, 0, $wPt, $hPt) | Out-Null
    $pres.SaveAs($outDir, 23)   # 23 = ppSaveAsEMF
    $pres.Close()
    $ppt.Quit()
    "EMF EXPORTED -> $outDir"
} -ArgumentList $PngPath, $OutDir
if (Wait-Job $job -Timeout 120) { Receive-Job $job } else { Write-Error "TIMEOUT" }
Remove-Job $job -Force
Stop-Process -Name POWERPNT -Force -ErrorAction SilentlyContinue
