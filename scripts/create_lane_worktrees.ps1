param(
    [string]$BaseBranch = "private-shop-core",
    [string[]]$Lanes = @("1w", "1d", "4h", "1h", "15m"),
    [switch]$CreateBaseBranch
)

$ErrorActionPreference = "Stop"

$repoRoot = (& git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
    throw "Not inside a git repository."
}

Set-Location $repoRoot

$baseExists = (& git show-ref --verify --quiet "refs/heads/$BaseBranch"; $LASTEXITCODE) -eq 0
if (-not $baseExists) {
    if (-not $CreateBaseBranch) {
        throw "Base branch '$BaseBranch' does not exist. Re-run with -CreateBaseBranch or create it first."
    }
    & git branch $BaseBranch HEAD
}

$laneRoot = Join-Path $repoRoot "lanes"
New-Item -ItemType Directory -Force -Path $laneRoot | Out-Null

$results = @()
foreach ($lane in $Lanes) {
    $branch = "lane/$lane"
    $path = Join-Path $laneRoot $lane
    $branchExists = (& git show-ref --verify --quiet "refs/heads/$branch"; $LASTEXITCODE) -eq 0
    if (-not $branchExists) {
        & git branch $branch $BaseBranch
    }
    if (-not (Test-Path $path)) {
        & git worktree add $path $branch
    }
    $results += [pscustomobject]@{
        Lane   = $lane
        Branch = $branch
        Path   = $path
        Base   = $BaseBranch
    }
}

$results | Format-Table -AutoSize
