"""Subpacote de configuração centralizada.

Expõe a instância singleton ``settings`` para consumo direto por
qualquer módulo do projeto.

Usage:
    >>> from recsys.config import settings
    >>> settings.model.top_k
    10
"""

from recsys.config.settings import AppSettings

# Instância singleton — carregada uma única vez na importação do pacote.
# Todos os módulos devem importar `settings` daqui, nunca instanciar
# AppSettings diretamente.
settings: AppSettings = AppSettings()

__all__ = ["AppSettings", "settings"]
