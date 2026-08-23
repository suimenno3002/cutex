param(
    [int]$M = 16384,
    [int]$N = 16384,
    [int]$K = 16384,
    [int]$Warmup = 20,
    [int]$Iterations = 200,
    [double]$GpuWarmupSeconds = 10.0,
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
    (Join-Path $projectRoot "modal_dense_gemm.py"),
    "--m", $M,
    "--n", $N,
    "--k", $K,
    "--warmup", $Warmup,
    "--iterations", $Iterations,
    "--gpu-warmup-seconds", $GpuWarmupSeconds
)
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
