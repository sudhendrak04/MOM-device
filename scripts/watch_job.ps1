param(
    [Parameter(Mandatory = $true)]
    [string]$JobId,
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$ApiKey = "mom-dev-key-2026",
    [int]$IntervalSec = 5
)

$headers = @{ "X-API-Key" = $ApiKey }

while ($true) {
    try {
        $job = Invoke-RestMethod -Uri "$BaseUrl/api/jobs/$JobId" -Headers $headers -TimeoutSec 10
    }
    catch {
        Write-Warning "Poll failed: $_"
        Start-Sleep -Seconds $IntervalSec
        continue
    }

    Write-Progress -Activity "MOM job $JobId" `
        -Status "$($job.stage) ($($job.progress)%%) - $($job.stage_message)" `
        -PercentComplete ([math]::Min(100, [int]$job.progress))

    if ($job.status -in @("completed", "failed")) {
        Write-Progress -Activity "MOM job" -Completed
        $job | ConvertTo-Json
        break
    }

    Start-Sleep -Seconds $IntervalSec
}
