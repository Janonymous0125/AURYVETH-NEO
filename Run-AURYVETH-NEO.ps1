param([string]$Prompt)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$client = Get-Command ollama -CommandType Application
$python = Join-Path $root '.venv\Scripts\python.exe'
$address = 'http://127.0.0.1:11435'
$previousHost = $env:OLLAMA_HOST

if (Get-NetTCPConnection -State Listen -LocalPort 11435 -ErrorAction SilentlyContinue) {
    throw 'Port 11435 is in use. Close the other AURYVETH-NEO session or run the service on a different port.'
}

$server = Start-Process -FilePath $python -ArgumentList '-X', 'utf8', '-m', 'scripts.neo_ollama' `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru
try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        if ($server.HasExited) { throw 'AURYVETH-NEO exited before it was ready.' }
        try {
            $ready = (Invoke-RestMethod -Uri $address -TimeoutSec 1).service -eq 'AURYVETH-NEO'
        } catch {
            $ready = $false
        }
        if ($ready) { break }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready) { throw 'AURYVETH-NEO did not start in time.' }
    $env:OLLAMA_HOST = $address
    if ($PSBoundParameters.ContainsKey('Prompt')) {
        & $client.Source run auryveth-neo $Prompt
    } else {
        & $client.Source run auryveth-neo
    }
    $clientExit = $LASTEXITCODE
} finally {
    $env:OLLAMA_HOST = $previousHost
    if (-not $server.HasExited) { Stop-Process -InputObject $server }
}
exit $clientExit
