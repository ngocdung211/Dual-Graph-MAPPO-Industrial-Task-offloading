param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [Parameter(Mandatory = $true)]
    [string]$ArtifactRoot,
    [Parameter(Mandatory = $true)]
    [string]$RunName
)

$ErrorActionPreference = "Stop"
$RunsRoot = Join-Path $ArtifactRoot "runs"
$Destination = Join-Path $RunsRoot $RunName
if (Test-Path -LiteralPath $Destination) {
    throw "Drive run already exists: $Destination"
}

New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null
$Temporary = Join-Path $RunsRoot (".$RunName.tmp-" + [guid]::NewGuid())
try {
    Copy-Item -LiteralPath $Source -Destination $Temporary -Recurse -Force
    Move-Item -LiteralPath $Temporary -Destination $Destination
} catch {
    if (Test-Path -LiteralPath $Temporary) {
        Remove-Item -LiteralPath $Temporary -Recurse -Force
    }
    throw
}

Write-Output $Destination
