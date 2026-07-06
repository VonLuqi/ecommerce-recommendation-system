"""Utilitários para integração e unificação do MLflow."""

from __future__ import annotations

from pathlib import Path
import mlflow

def get_friendly_tracking_uri(current_uri: str) -> str:
    """Resolve a URI de tracking amigável para o host.
    
    Se estiver rodando dentro do container (mlflow:5000), tenta ler 
    a URI real configurada no arquivo .env do host.
    """
    if "mlflow:5000" in current_uri:
        env_path = Path(".env")
        if env_path.exists():
            try:
                for line in env_path.read_text().splitlines():
                    if line.strip().startswith("MLFLOW_TRACKING_URI="):
                        val = line.split("=", 1)[1].strip().strip("'\"")
                        if val:
                            return val
            except Exception:
                pass
        return "http://localhost:5001"
    return current_uri


def get_latest_run_id(experiment_name: str) -> str | None:
    """Busca o ID da última execução registrada no experimento atual."""
    try:
        mlflow.set_experiment(experiment_name)
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs.empty:
            return str(runs.iloc[0].run_id)
    except Exception:
        pass
    return None
