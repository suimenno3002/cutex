param(
    [string]$Gpu = "L4",
    [int]$M = 4096,
    [int]$N = 4096,
    [int]$Warmup = 10,
    [int]$Iterations = 100,
    [switch]$ForceRetune,
    [switch]$Trace,
    [switch]$DumpIr
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$projectRoot = Split-Path -Parent $PSScriptRoot
$modalArgs = @(
    "run",
    (Join-Path $projectRoot "modal_app.py"),
    "--gpu", $Gpu,
    "--m", $M,
    "--n", $N,
    "--warmup", $Warmup,
    "--iterations", $Iterations
)
if ($ForceRetune) {
    $modalArgs += "--force-retune"
}
if ($Trace) {
    $modalArgs += "--trace"
}
if ($DumpIr) {
    $modalArgs += "--dump-ir"
}

Push-Location $projectRoot
try {
    & uv run modal @modalArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
