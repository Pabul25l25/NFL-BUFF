# Corre el ciclo semanal completo de NFL-Predictor sin intervencion manual:
# ingesta, features, reentrena, promueve si corresponde, predice y publica a
# Neon. Pensado para el Programador de Tareas de Windows (martes 22:00).
#
# Historial de por que esta version es asi (dos intentos previos fallaron):
#
# 1. `uv run ... *>> $logFile` con $ErrorActionPreference="Stop": PowerShell
#    5.1 envuelve cada linea de stderr de un comando nativo en un
#    NativeCommandError; con -Stop, el aviso informativo de `uv` sobre
#    VIRTUAL_ENV abortaba el script aunque `uv` fuera a terminar con exito.
# 2. `cmd /c "uv run ... >> log 2>&1"`: funciono, pero bajo el Programador de
#    Tareas (LogonType Interactive) el Programador perdio el rastro del
#    proceso real varios minutos despues de arrancar — lo marco como
#    terminado (y con codigo de error) mientras el proceso seguia vivo y
#    trabajando de verdad en segundo plano, desacoplado de Task Scheduler.
#    Cuantas mas capas de proceso anidadas (powershell -> cmd -> uv ->
#    python), mas facil que el Job Object del Programador pierda el hijo
#    real. Se saca la capa de `cmd` y se usa `Start-Process`, que crea un
#    hijo directo con manejo de E/S nativo de Win32 (sin el problema del
#    punto 1 y con una jerarquia de procesos mas plana).

$env:NFLP_SYNC__ENABLED = [Environment]::GetEnvironmentVariable("NFLP_SYNC__ENABLED", "User")
$env:NFLP_SYNC__POSTGRES_URL = [Environment]::GetEnvironmentVariable("NFLP_SYNC__POSTGRES_URL", "User")

$repoRoot = "D:\NFL"
$uvExe = "C:\Users\joseP\.local\bin\uv.exe"
$logDir = Join-Path $repoRoot "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$stdoutLog = Join-Path $logDir "weekly_$stamp.log"
$stderrLog = Join-Path $logDir "weekly_$stamp.err.log"

$proc = Start-Process -FilePath $uvExe `
    -ArgumentList @("run", "nflp", "pipeline", "weekly") `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -NoNewWindow -Wait -PassThru

$exitCode = $proc.ExitCode
$status = if ($exitCode -eq 0) { "OK" } else { "ERROR" }
"=== $status (exit=$exitCode) $(Get-Date -Format o) ===" | Add-Content $stdoutLog

if ($exitCode -ne 0) {
    exit $exitCode
}
