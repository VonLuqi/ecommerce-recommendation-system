"""Script de validação do ambiente de desenvolvimento.

Verifica se o ambiente está corretamente configurado para executar o
projeto. Útil para detectar problemas antes de rodar treinamento ou
experimentos.

Checagens realizadas:
1. Versão mínima do Python (3.11+).
2. Presença das dependências principais.
3. Carregamento das variáveis de ambiente obrigatórias.
4. Existência dos diretórios essenciais do projeto.

Usage:
    python scripts/validate_env.py

Exit codes:
    0 — Todas as verificações passaram.
    1 — Uma ou mais verificações falharam.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MIN_PYTHON = (3, 11)

# Dependências cuja presença é verificada por importação
REQUIRED_PACKAGES: list[str] = [
    "torch",
    "sklearn",       # scikit-learn
    "mlflow",
    "pydantic",
    "pydantic_settings",
    "dotenv",        # python-dotenv
    "numpy",
    "pandas",
]

# Variáveis de ambiente obrigatórias (lidas via .env)
REQUIRED_ENV_VARS: list[str] = [
    "PROJECT_NAME",
    "ENVIRONMENT",
    "RANDOM_SEED",
    "RAW_DATA_PATH",
    "PROCESSED_DATA_PATH",
    "MODELS_PATH",
    "TOP_K",
]

# Diretórios que devem existir no projeto
REQUIRED_DIRS: list[Path] = [
    Path("src/recsys"),
    Path("tests"),
    Path("data/raw"),
    Path("data/processed"),
    Path("models"),
    Path("configs"),
    Path("scripts"),
]


# ---------------------------------------------------------------------------
# Helpers de exibição
# ---------------------------------------------------------------------------

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}✗{_RESET} {msg}")


def _section(title: str) -> None:
    print(f"\n{_BOLD}{title}{_RESET}")


# ---------------------------------------------------------------------------
# Verificações
# ---------------------------------------------------------------------------


def check_python_version() -> bool:
    """Verifica se a versão do Python atende ao requisito mínimo.

    Returns:
        True se a versão for compatível, False caso contrário.
    """
    _section("Python")
    current = sys.version_info[:2]
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    min_str = ".".join(str(v) for v in MIN_PYTHON)

    if current >= MIN_PYTHON:
        _ok(f"Python {version_str} >= {min_str}")
        return True

    _fail(f"Python {version_str} < {min_str} (mínimo exigido)")
    return False


def check_packages() -> bool:
    """Verifica se as dependências principais estão instaladas.

    Returns:
        True se todas as dependências estiverem disponíveis.
    """
    _section("Dependências")
    all_ok = True

    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            _ok(pkg)
        except ImportError:
            _fail(f"{pkg} — NÃO encontrado")
            all_ok = False

    return all_ok


def check_env_vars() -> bool:
    """Verifica se as variáveis de ambiente obrigatórias estão definidas.

    Carrega o arquivo `.env` antes de validar.

    Returns:
        True se todas as variáveis estiverem definidas.
    """
    _section("Variáveis de ambiente")

    # Carrega .env se existir
    env_file = Path(".env")
    if env_file.exists():
        try:
            from dotenv import load_dotenv  # noqa: PLC0415

            load_dotenv(env_file)
            _ok(f".env carregado ({env_file.resolve()})")
        except ImportError:
            _warn = "python-dotenv indisponível — pulando carga do .env"
            print(f"  {_YELLOW}⚠{_RESET}  {_warn}")
    else:
        _not_found = ".env não encontrado — usando variáveis do sistema"
        print(f"  {_YELLOW}⚠{_RESET}  {_not_found}")

    all_ok = True
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if value is not None:
            _ok(f"{var}={value!r}")
        else:
            _fail(f"{var} — não definida")
            all_ok = False

    return all_ok


def check_directories() -> bool:
    """Verifica se os diretórios essenciais existem no repositório.

    Returns:
        True se todos os diretórios existirem.
    """
    _section("Diretórios do projeto")
    all_ok = True
    root = Path(__file__).resolve().parent.parent  # raiz do repo

    for rel_dir in REQUIRED_DIRS:
        full_path = root / rel_dir
        if full_path.exists():
            _ok(str(rel_dir))
        else:
            _fail(f"{rel_dir} — diretório ausente")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    """Executa todas as verificações e retorna o exit code.

    Returns:
        0 se todas as verificações passarem, 1 caso contrário.
    """
    print(f"\n{_BOLD}{'=' * 50}{_RESET}")
    print(f"{_BOLD}  Validação do Ambiente — ecommerce-recsys{_RESET}")
    print(f"{_BOLD}{'=' * 50}{_RESET}")

    results = [
        check_python_version(),
        check_packages(),
        check_env_vars(),
        check_directories(),
    ]

    print(f"\n{_BOLD}{'=' * 50}{_RESET}")
    total = len(results)
    passed = sum(results)
    failed = total - passed

    if failed == 0:
        _msg = f"  Resultado: {passed}/{total} verificações passaram ✓"
        print(f"{_GREEN}{_BOLD}{_msg}{_RESET}")
        print(f"{_BOLD}{'=' * 50}{_RESET}\n")
        return 0

    print(f"{_RED}{_BOLD}  Resultado: {failed}/{total} verificações falharam ✗{_RESET}")
    print(f"{_BOLD}{'=' * 50}{_RESET}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
