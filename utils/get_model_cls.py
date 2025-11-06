# utils/get_model_cls.py
import difflib

from models import (
    TCFormer, ATCNet, BaseNet, EEGConformer, EEGNet, EEGTCNet,
    ShallowNet, TSSEFFNet, CTNet, MSCFormer, EEGDeformer, RatioWaveNet, 
    #RatioWaveNet_CrossAtt
)

model_dict = dict(
    TCFormer=TCFormer,
    ATCNet=ATCNet,
    BaseNet=BaseNet,
    EEGConformer=EEGConformer,
    EEGNet=EEGNet,
    EEGTCNet=EEGTCNet,
    ShallowNet=ShallowNet,
    TSSEFFNet=TSSEFFNet,
    CTNet=CTNet,
    MSCFormer=MSCFormer,
    EEGDeformer=EEGDeformer,
    RatioWaveNet=RatioWaveNet,
    #RatioWaveNet_CrossAtt=RatioWaveNet_CrossAtt,
)

def _normalize(s: str) -> str:
    # lower + rimuove tutto tranne [a-z0-9]
    return ''.join(ch for ch in s.lower() if ch.isalnum())

# mappa normalizzata -> classe (consente varianti come ratiowavenet / ratio-wavenet / ratio_wavenet)
_NORMALIZED = { _normalize(k): v for k, v in model_dict.items() }

def get_model_cls(model_name: str):
    key = _normalize(model_name)
    if key in _NORMALIZED:
        return _NORMALIZED[key]
    # suggerimenti utili in errore
    suggestions = difflib.get_close_matches(model_name, list(model_dict.keys()), n=3, cutoff=0.0)
    valid = ", ".join(sorted(model_dict.keys()))
    hint = f" Forse intendevi: {', '.join(suggestions)}." if suggestions else ""
    raise KeyError(f"Modello '{model_name}' non trovato. Opzioni valide: {valid}.{hint}")
