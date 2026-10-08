# png_to_emf.ps1 -- export PNG to Word-compatible EMF via PowerPoint COM
# usage: powershell -ExecutionPolicy Bypass -File png_to_emf.ps1 <input.png> <output_dir>
# output: <output_dir>/slide1.emf (single-slide deck sized to the image)
# note: LibreOffice --convert-to emf produces EMFs that crash Word (2026-10-07);
#       Office-exported EMFs are safe. PowerPoint COM rejects forward slashes
#       in AddPicture/SaveAs paths (2026-10-08), so normalize to backslashes.
param(
    [Parameter(Mandatory = $true)][string]$PngPath,
    [Parameter(Mandatory = $true)][string]$OutDir
)
$PngPath = $PngPath -replace '/', '\'
$OutDir  = $OutDir  -replace '/', '\'
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
