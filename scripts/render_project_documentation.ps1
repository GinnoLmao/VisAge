param(
    [string]$Docx,
    [string]$Pdf
)
$ErrorActionPreference = 'Stop'
$visageProjectRoot = Split-Path $PSScriptRoot
if (-not $Docx) { $Docx = Join-Path $visageProjectRoot 'output/docx/VisAge_Project_Documentation.docx' }
if (-not $Pdf) { $Pdf = Join-Path $visageProjectRoot 'output/pdf/VisAge_Project_Documentation.pdf' }
$visageWord = $null
$visageDocument = $null
try {
    $visageWord = New-Object -ComObject Word.Application
    $visageWord.Visible = $false
    $visageWord.DisplayAlerts = 0
    $visageDocument = $visageWord.Documents.Open($Docx, $false, $false)
    $visageDocument.Repaginate()
    [void]$visageDocument.Fields.Update()
    foreach ($visageToc in $visageDocument.TablesOfContents) { $visageToc.Update() }
    foreach ($visageStyleName in @('Title','Subtitle','Heading 1','Heading 2','TOC 1','TOC 2')) {
        $visageStyle = $visageDocument.Styles.Item($visageStyleName)
        $visageStyle.Font.Color = 0
        $visageStyle.Font.Name = 'Calibri'
        $visageStyle.Font.Italic = 0
        $visageStyle.ParagraphFormat.Borders.Enable = 0
    }
    $visageDocument.Repaginate()
    foreach ($visageToc in $visageDocument.TablesOfContents) { $visageToc.UpdatePageNumbers() }
    [void]$visageDocument.Fields.Update()
    $visageDocument.Save()
    $visageDocument.ExportAsFixedFormat($Pdf,17)
    Get-Item -LiteralPath $Pdf | Select-Object Name,Length
} finally {
    if ($null -ne $visageDocument) { $visageDocument.Close(0); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($visageDocument) }
    if ($null -ne $visageWord) { $visageWord.Quit(0); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($visageWord) }
}
