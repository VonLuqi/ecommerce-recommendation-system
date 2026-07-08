<#
.SYNOPSIS
    Configura o ambiente de desenvolvimento do projeto no Windows.

.DESCRIPTION
    1. Verifica se o Python 3.11+ está instalado.
    2. Localiza ou instala o Poetry.
    3. Adiciona o Poetry ao PATH do usuário (permanente).
    4. Instala todas as dependências do projeto (produção + dev).
    5. Copia .env.example → .env (se .env não existir).
    6. Exibe os próximos passos.

.EXAMPLE
    # Abra o PowerShell como usuário normal (NÃO como Administrador) e rode:
    .\scripts\setup.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers de output
# ---------------------------------------------------------------------------

function Write-Ok($msg)      { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Fail($msg)    { Write-Host "  [ERRO] $msg" -ForegroundColor Red }
function Write-Info($msg)    { Write-Host "  [INFO] $msg" -ForegroundColor Cyan }
function Write-Section($msg) { Write-Host "`n==> $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# 1. Python
# ---------------------------------------------------------------------------

Write-Section "Verificando Python"

$pythonCandidates = @(
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python312\python.exe",
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python311\python.exe",
    "C:\Python312\python.exe",
    "python.exe"
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $pythonExe = $candidate
        break
    }
}

if (-not $pythonExe) {
    Write-Fail "Python 3.11+ não encontrado."
    Write-Info "Baixe em: https://www.python.org/downloads/"
    exit 1
}

$version = & $pythonExe --version 2>&1
Write-Ok "Encontrado: $version ($pythonExe)"

# ---------------------------------------------------------------------------
# 2. Poetry
# ---------------------------------------------------------------------------

Write-Section "Verificando Poetry"

$poetryScriptsDir = "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python312\Scripts"
$poetryExe = Join-Path $poetryScriptsDir "poetry.exe"

if (-not (Test-Path $poetryExe)) {
    Write-Info "Poetry não encontrado. Instalando via pip..."
    & $pythonExe -m pip install poetry --quiet
}

Write-Ok "Poetry encontrado: $poetryExe"

# ---------------------------------------------------------------------------
# 3. Adicionar Poetry ao PATH do usuário (permanente)
# ---------------------------------------------------------------------------

Write-Section "Configurando PATH"

$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$poetryScriptsDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$poetryScriptsDir", "User")
    Write-Ok "Adicionado ao PATH: $poetryScriptsDir"
    Write-Info "Abra um novo terminal para que o PATH tenha efeito."
} else {
    Write-Ok "Já estava no PATH."
}

$env:PATH = "$env:PATH;$poetryScriptsDir"

# ---------------------------------------------------------------------------
# 4. Instalar dependências
# ---------------------------------------------------------------------------

Write-Section "Instalando dependências do projeto"

Push-Location $PSScriptRoot\..
try {
    & $poetryExe install --with dev
    Write-Ok "Dependências instaladas com sucesso."
} finally {
    Pop-Location
}

# ---------------------------------------------------------------------------
# 5. Criar .env a partir do .env.example
# ---------------------------------------------------------------------------

Write-Section "Configurando .env"

$projectRoot = Split-Path $PSScriptRoot -Parent
$envFile     = Join-Path $projectRoot ".env"
$envExample  = Join-Path $projectRoot ".env.example"

if (-not (Test-Path $envFile)) {
    Copy-Item $envExample $envFile
    Write-Ok ".env criado a partir de .env.example"
    Write-Info "Edite o .env se precisar ajustar alguma variável."
} else {
    Write-Ok ".env já existe, mantido sem alterações."
}

# ---------------------------------------------------------------------------
# 6. Próximos passos
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Ambiente configurado com sucesso!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Proximos passos:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. Baixe o dataset Instacart (requer conta Kaggle):"
Write-Host "     https://www.kaggle.com/competitions/instacart-market-basket-analysis/data"
Write-Host ""
Write-Host "  2. Coloque os arquivos em data\raw\:"
Write-Host "       orders.csv"
Write-Host "       order_products__prior.csv"
Write-Host ""
Write-Host "  3. Inicialize o DVC e execute o pipeline:"
Write-Host "     poetry run dvc init"
Write-Host "     poetry run dvc add data\raw\orders.csv"
Write-Host "     poetry run dvc add data\raw\order_products__prior.csv"
Write-Host "     poetry run dvc repro"
Write-Host ""
Write-Host "  4. Veja as metricas:"
Write-Host "     poetry run dvc metrics show"
Write-Host ""
