# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ashley R. Thomas

# The tutorial video uses the following one-liner:
#
#   $outputDir = "C:\demo\project-sessions-output\$((Get-Date).ToString("yyyyMMdd-HHmmss"))\" ; mkdir $outputDir ; Write-Host "outputDir=$outputDir" ; $env:PYTHONUNBUFFERED = "1"; python C:\demo\llm-prompts-and-tools\scripts\ollama_summarize_claude_markdown.py C:\demo\project-sessions-md\ -o $outputDir 2>&1 | Tee-Object -FilePath "$outputDir\log.txt"
#
# This script does the same thing with less typing (and adds model/parallel/
# resume/diag-dir options), e.g.:
#
#   .\run_ollama_summarize_claude_markdown.ps1 C:\demo\project-sessions-md C:\demo\project-sessions-output

<#
.SYNOPSIS
    Run ollama_summarize_claude_markdown.py against a folder of Claude markdown,
    writing results into a fresh timestamped subdirectory.

.DESCRIPTION
    Creates <OutputRootDirectory>\<yyyyMMdd-HHmmss>\ as the output directory,
    then runs the summarizer (located side-by-side with this script) with output
    unbuffered, teeing all output to log.txt inside that directory.

.PARAMETER SourceProjectMarkdownDirectory
    Path to a Claude markdown file or a folder of .md/.txt files to summarize.

.PARAMETER OutputRootDirectory
    Parent directory under which a timestamped run subdirectory is created.

.PARAMETER Model
    Short name (or full Ollama tag) of the model to use; known short names are
    mapped to their full tag before being passed to the summarizer, and any
    other value is passed through as-is. Tested with gemma4:e4b-it-q8_0,
    llama3.1:8b, phi4:14b, gpt-oss:20b. Defaults to 'gemma4'.

.PARAMETER Parallel
    Number of concurrent Ollama requests. Defaults to 1.

.PARAMETER Resume
    Skip source files whose <basename>_summary.md already exists in the output
    directory. Off by default.

.PARAMETER DiagDir
    If set, write per-chunk/per-group/final diagnostic outputs under this
    directory. Unset by default (no diagnostics captured).

.EXAMPLE
    .\run_ollama_summarize_claude_markdown.ps1 c:\sourcedir c:\outputroot

.EXAMPLE
    .\run_ollama_summarize_claude_markdown.ps1 c:\sourcedir c:\outputroot -Parallel 1 -Resume
#>

param(
    [Parameter(Mandatory)]
    [string] $SourceProjectMarkdownDirectory,

    [Parameter(Mandatory)]
    [string] $OutputRootDirectory,

    [ArgumentCompletions(
        "gemma4", "gemma4:e4b-it-q8_0",
        "phi4", "phi4:14b",
        "llama3.1", "llama3.1:8b",
        "gpt-oss", "gpt-oss:20b"
    )]
    [string] $Model = "gemma4",

    [int] $Parallel = 1,

    [switch] $Resume,

    [string] $DiagDir
)

$ErrorActionPreference = "Stop"

# Map friendly model names to their full Ollama tags. Anything not listed here
# is passed through verbatim, so any valid Ollama tag works. Add entries (and
# completion suggestions above) as more shorthands are useful.
$modelTags = @{
    "gemma4"   = "gemma4:e4b-it-q8_0"
    "phi4"     = "phi4:14b"
    "llama3.1" = "llama3.1:8b"
    "gpt-oss"  = "gpt-oss:20b"
}
$modelTag = if ($modelTags.ContainsKey($Model)) { $modelTags[$Model] } else { $Model }

# The Python script lives next to this one.
$scriptDir = $PSScriptRoot
$summarizer = Join-Path $scriptDir "ollama_summarize_claude_markdown.py"

# Each run gets its own timestamped output directory.
$timestamp = (Get-Date).ToString("yyyyMMdd-HHmmss")
$outputDir = Join-Path $OutputRootDirectory $timestamp
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
Write-Host "outputDir=$outputDir"

# Build the argument list, only including optional flags when they're requested.
# A blank/unset DiagDir or an off Resume means "don't pass it" so the Python
# script falls back to its own defaults (no diag dir, no resume).
$pyArgs = @(
    $summarizer
    $SourceProjectMarkdownDirectory
    "-o", $outputDir
    "-m", $modelTag
    "-p", $Parallel
)
if ($Resume) {
    $pyArgs += "--resume"
}
if ($DiagDir) {
    $pyArgs += "--diag-dir", $DiagDir
}

$env:PYTHONUNBUFFERED = "1"
$logFile = Join-Path $outputDir "log.txt"

# Run the summarizer. A thrown CommandNotFoundException (no python at all) or a
# non-zero exit (Microsoft Store stub, ModuleNotFoundError, bad path, etc.) both
# mean the run didn't complete — surface the same troubleshooting hint.
$failed = $false
try {
    python @pyArgs 2>&1 | Tee-Object -FilePath $logFile
    if ($LASTEXITCODE -ne 0) { $failed = $true }
}
catch {
    Write-Warning $_
    $failed = $true
}

if ($failed) {
    Write-Host ""
    Write-Warning @"
The summarizer did not run to completion. Common causes:
  - Python isn't on your PATH. Install it, or activate your virtual environment first.
  - The active environment is missing 'requests':  pip install requests
See $logFile for the underlying error.
"@
    exit 1
}
