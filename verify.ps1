param(
    [ValidateSet("fast_demo", "formal")]
    [string]$Mode = "fast_demo",
    [string]$PythonExecutable = "C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$OutputDirectory = Join-Path $ProjectRoot "results\$Mode"

Push-Location $ProjectRoot
try {
    & $PythonExecutable -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Unit tests failed" }

    & $PythonExecutable main_demo.py --mode $Mode --output-dir $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw "Demo run failed" }

    & $PythonExecutable verify_results.py $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw "Result acceptance failed" }
}
finally {
    Pop-Location
}
