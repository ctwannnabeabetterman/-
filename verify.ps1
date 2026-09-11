param(
    [ValidateSet("fast_demo", "formal")]
    [string]$Mode = "fast_demo",
    [string]$PythonExecutable = "C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$ResultsRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot "results"))
$OutputDirectory = [System.IO.Path]::GetFullPath((Join-Path $ResultsRoot $Mode))
$ExpectedPrefix = $ResultsRoot + [System.IO.Path]::DirectorySeparatorChar

if (-not $OutputDirectory.StartsWith($ExpectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean output outside the project results directory: $OutputDirectory"
}

Push-Location $ProjectRoot
try {
    & $PythonExecutable -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Unit tests failed" }

    if (Test-Path -LiteralPath $OutputDirectory) {
        Remove-Item -LiteralPath $OutputDirectory -Recurse -Force
    }
    & $PythonExecutable main_demo.py --mode $Mode --output-dir $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw "Demo run failed" }

    & $PythonExecutable verify_results.py $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw "Result acceptance failed" }
}
finally {
    Pop-Location
}
