param(
    [string[]]$Lanes = @("1w", "1d", "4h", "1h", "15m"),
    [switch]$DeleteBranches
)

$ErrorActionPreference = "Stop"

$repoRoot = (& git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
    throw "Not inside a git repository."
}

Set-Location $repoRoot

$laneRoot = Join-Path $repoRoot "lanes"
foreach ($lane in $Lanes) {
    $branch = "lane/$lane"
    $path = Join-Path $laneRoot $lane
    if (Test-Path $path) {
        & git worktree remove --force $path
    }
    if ($DeleteBranches) {
        $branchExists = (& git show-ref --verify --quiet "refs/heads/$branch"; $LASTEXITCODE) -eq 0
        if ($branchExists) {
            & git branch -D $branch
        }
    }
}
