"""Verifica se o modelo neural está funcional."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from recsys.models.neural_net import NeuMF

# 1. Instanciar
model = NeuMF(num_users=10, num_items=10, embedding_dim=8, mlp_hidden_dims=[16, 8])
print(f"[OK] NeuMF instanciado - parametros: {sum(p.numel() for p in model.parameters())}")

# 2. Forward pass
user_ids = torch.tensor([0, 1, 2])
item_ids = torch.tensor([3, 4, 5])
out = model(user_ids, item_ids)
print(f"[OK] Forward pass - shape: {out.shape}, valores: {out.detach().tolist()}")

# 3. Verificar heranca
from recsys.recommenders.base import BaseRecommender
from recsys.recommenders.neural import NeuralRecommender
assert issubclass(NeuralRecommender, BaseRecommender)
print("[OK] NeuralRecommender herda de BaseRecommender")

# 4. Verificar arquivos existem
files = [
    "src/recsys/models/neural_net.py",
    "src/recsys/models/__init__.py",
    "src/recsys/recommenders/neural.py",
    "src/recsys/recommenders/__init__.py",
    "src/recsys/pipeline/train.py",
]
base = Path(__file__).resolve().parent.parent
for f in files:
    assert (base / f).exists(), f"Arquivo faltando: {f}"
    print(f"[OK] {f} existe")

print("\n=== MODELO NEURAL FUNCIONAL ===")