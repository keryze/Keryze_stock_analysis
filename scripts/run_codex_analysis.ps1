$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$recommendDir = Join-Path $scriptDir "recommend"

$dateFull = Get-Date -Format "yyyyMMdd"

$promptFile = Join-Path $recommendDir "analysis_prompt.md"
$reportCandidates = @(
    (Join-Path $recommendDir "daily_recommendation_$dateFull.txt"),
    (Join-Path $recommendDir "daily_recommendation.txt")
)

if (-not (Test-Path $promptFile)) {
    throw "Prompt file not found: $promptFile"
}

$reportFile = $reportCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $reportFile) {
    throw "Daily recommendation report not found."
}

$promptTemplate = Get-Content -Path $promptFile -Raw -Encoding UTF8
$reportContent = Get-Content -Path $reportFile -Raw -Encoding UTF8
$codexCmd = Get-Command codex.exe -ErrorAction SilentlyContinue
$codexExe = $null

if ($codexCmd) {
    $codexExe = $codexCmd.Source
} else {
    $fallbackCodexExe = Join-Path $env:USERPROFILE ".vscode/extensions/openai.chatgpt-26.5309.21912-win32-x64/bin/windows-x86_64/codex.exe"
    if (Test-Path $fallbackCodexExe) {
        $codexExe = $fallbackCodexExe
    }
}

if (-not $codexExe) {
    throw "codex.exe not found. Please install Codex CLI or update the fallback path in run_codex_analysis.ps1."
}

$finalPrompt = $promptTemplate + "`r`n`r`n" + "Below is today's long-term stock selection report. Please analyze it based on the prompt above." + "`r`n`r`n" + $reportContent

$outputFile = Join-Path $recommendDir "codex_analysis_result_$dateFull.txt"
$tempPromptFile = Join-Path $recommendDir "codex_prompt_$dateFull.tmp.txt"

[System.IO.File]::WriteAllText($tempPromptFile, $finalPrompt, [System.Text.UTF8Encoding]::new($false))

try {
  python -X utf8 -c "import pathlib,sys; sys.stdout.write(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))" $tempPromptFile | & $codexExe exec `
    --skip-git-repo-check `
    --cd $scriptDir `
    -s read-only `
    -o $outputFile `
    -
}
finally {
  if (Test-Path $tempPromptFile) {
    Remove-Item $tempPromptFile -Force
  }
}

Write-Host "Prompt file: $promptFile"
Write-Host "Report file: $reportFile"
Write-Host "Output file: $outputFile"
