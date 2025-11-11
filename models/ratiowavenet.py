########
# RatioWaveNet: Subject-Adaptive RDWT Frontend for EEG Classification
########

# Core Libraries
import torch
from torch import nn, Tensor

# Utility Libraries
from einops import rearrange
from einops.layers.torch import Rearrange

# Local application-specific imports
from .classification_module import ClassificationModule
from .modules import CausalConv1d, Conv1dWithConstraint
from .channel_group_attention import ChannelGroupAttention
from utils.weight_initialization import glorot_weight_zero_bias
from utils.latency  import measure_latency

# --- RDWT front-end (PyTorch) -----------------------------------------------
import torch.nn.functional as F


class _SpectralBandAttention(nn.Module):
    """
    Attention sui livelli RDWT.
    Input: energia per-banda (B, L). Output: pesi softmax (B, L).
    Inizializzata per tornare pesi uniformi (W=0, b=0).
    """
    def __init__(self, levels: int, temperature: float = 1.0):
        super().__init__()
        self.levels = int(levels)
        self.temperature = float(temperature)
        self.fc = nn.Linear(self.levels, self.levels, bias=True)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, band_energy: Tensor) -> Tensor:
        logits = self.fc(band_energy) / self.temperature
        return torch.softmax(logits, dim=-1)


class RDWTFrontEnd(nn.Module):
    """
    RDWT front-end con scale reparametrizzate (sigmoide) + soft-threshold opzionale.
    Lavora su tensori (B, C, T). Supporta level-dropout per spegnere intere bande.
    """
    
    def __init__(self,
                 levels: int = 4,
                 base_kernel_len: int = 16,
                 init_wavelet: str = 'db4',
                 init_dilations=(1.5, 5/3, 7/4, 9/5),
                 use_soft_threshold: bool = True,
                 threshold_init: float = 0.0,
                 max_scale: float = 4.0,
                 l2_on_logscale: float = 0.0,     # >0 per reg. su z
                 use_spread_loss: bool = True,
                 spread_lambda: float = 5e-3,
                 spread_gamma: float = 4.0,
                 temp_init: float = 0.5,
                 # NEW: level-dropout
                 level_dropout_p: float = 0.30,
                 level_dropout_mode: str = "per_sample",
                 # SAFE knobs (ex Modifica A)
                 alpha_scale: float = 1.10,
                 energy_guard: bool = False,
                 energy_guard_max_gain: float = 2.0,
                 # NEW: attention per-banda
                 use_spectral_attention: bool = True,
                 attn_temperature: float = 1.0):
        super().__init__()
        assert levels >= 1
        self.levels = int(levels)
        self.K = int(base_kernel_len)
        self.use_soft_threshold = bool(use_soft_threshold)
        self.max_scale = float(max_scale)
        self.l2_on_logscale = float(l2_on_logscale)
        self.use_spread_loss = bool(use_spread_loss)
        self.spread_lambda = float(spread_lambda)
        self.spread_gamma  = float(spread_gamma)
        self.level_dropout_p = float(level_dropout_p)
        self.level_dropout_mode = str(level_dropout_mode)

        # Spettrale: attenzione per-banda
        self.use_spectral_attention = bool(use_spectral_attention)
        self.attn_temperature = float(attn_temperature)

        # SAFE knobs
        self.alpha_scale = float(alpha_scale)
        self.energy_guard = bool(energy_guard)
        self.energy_guard_max_gain = float(energy_guard_max_gain)

        # db4 prototipi
        if init_wavelet != 'db4':
            raise NotImplementedError("Only 'db4' supported here.")
        db4_lp = torch.tensor([
            -0.0105974,  0.0328830,  0.0308414, -0.1870348,
            -0.0279838,  0.6308808,  0.7148466,  0.2303778
        ], dtype=torch.float32)
        db4_hp = torch.tensor([
            -0.2303778,  0.7148466, -0.6308808, -0.0279838,
             0.1870348,  0.0308414, -0.0328830, -0.0105974
        ], dtype=torch.float32)

        # resample iniziale a K (scale=1.0)
        lo0 = self._resample(db4_lp, scale=1.0, out_len=self.K)
        hi0 = self._resample(db4_hp, scale=1.0, out_len=self.K)

        # Filtri base trainabili
        self.lo_proto = nn.Parameter(lo0.clone())
        self.hi_proto = nn.Parameter(hi0.clone())

        # z (logit delle scale) + centro z_mu (non trainabile)
        init_dils = torch.tensor(init_dilations, dtype=torch.float32)[:self.levels]
        z_init = self._z_from_s_init(init_dils, smax=self.max_scale)  # logit in (0,1)
        self.z     = nn.Parameter(z_init.clone())
        self.register_buffer('z_mu', z_init.clone(), persistent=False)

        # temperatura (buffer)
        self.register_buffer('temp', torch.tensor(float(temp_init), dtype=torch.float32), persistent=False)

        # soglie e alpha per livello
        if self.use_soft_threshold:
            self.tau_raw = nn.Parameter(torch.full((self.levels,), float(threshold_init)))
        else:
            self.tau_raw = None
        self.alpha = nn.Parameter(torch.ones(self.levels))
        if self.use_spectral_attention:
            self.band_attn = _SpectralBandAttention(self.levels, self.attn_temperature)
        else:
            self.band_attn = None

        # reg-loss (solo se abilitata)
        self._last_reg_loss = torch.tensor(0.0)

    @staticmethod
    def _resample(kernel: Tensor, scale: float, out_len: int) -> Tensor:
        """Interpolazione lineare 1D (stretch/compress). kernel: (K,) → (out_len,)"""
        K = kernel.numel()
        c = 0.5 * (K - 1)
        t = torch.arange(out_len, dtype=kernel.dtype, device=kernel.device)
        u = c + (t - c) / float(scale)                       # posizioni in [0, K-1]
        u0 = torch.clamp(torch.floor(u), 0, K - 1).long()
        u1 = torch.clamp(u0 + 1, 0, K - 1)
        w1 = (u - u0.to(u.dtype))
        w0 = 1.0 - w1
        v0 = kernel[u0]
        v1 = kernel[u1]
        return w0 * v0 + w1 * v1

    @staticmethod
    def _l1norm(vec: Tensor, eps: float = 1e-6) -> Tensor:
        return vec / (vec.abs().sum() + eps)

    @staticmethod
    def _soft_thresh(x: Tensor, tau: Tensor) -> Tensor:
        return torch.sign(x) * torch.relu(x.abs() - tau)

    @staticmethod
    def _z_from_s_init(s_init: Tensor, smax: float) -> Tensor:
        """logit di p = (s-1)/(smax-1) con clipping per stabilità."""
        p = (s_init - 1.0) / (smax - 1.0)
        p = torch.clamp(p, 1e-4, 1 - 1e-4)
        return torch.log(p) - torch.log(1.0 - p)

    def set_temperature(self, new_temp: float):
        self.temp = torch.tensor(float(new_temp), dtype=self.temp.dtype, device=self.temp.device)

    def regularization_loss(self) -> Tensor:
        """Ritorna l'ultima reg-loss calcolata in forward (0 se disattivata)."""
        return self._last_reg_loss

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, T) -> y: (B, C, T)
        Applica per-livello depthwise conv (per canale) con low/high db4 riscalati.
        """
        B, C, T = x.shape
        device = x.device
        dtype  = x.dtype

        # s = 1 + (Smax - 1) * sigmoid(z / temp)
        s = 1.0 + (self.max_scale - 1.0) * torch.sigmoid(self.z / self.temp)  # (L,)

        # regularizzazioni (disattive se non richieste)
        reg = torch.tensor(0.0, device=device, dtype=dtype)
        if self.use_spread_loss or self.l2_on_logscale > 0:
            k = 4.0
            left  = torch.exp(-k * (s - 1.0))
            right = torch.exp(-k * (self.max_scale - s))
            reg = reg + 2e-3 * (left.mean() + right.mean())
        if self.l2_on_logscale > 0:
            reg = reg + self.l2_on_logscale * torch.mean((self.z - self.z_mu) ** 2)
        if self.use_spread_loss and self.spread_lambda > 0:
            diffs = (s.view(-1,1) - s.view(1,-1)).abs() + 1e-6
            spread = torch.exp(-self.spread_gamma * diffs).sum() - float(self.levels)
            reg = reg + self.spread_lambda * spread
        self._last_reg_loss = reg

        # pipeline RDWT
        a_prev = x  # (B,C,T)
        details_sum = torch.zeros_like(x)
        detail_list = []

        # padding "same"
        pad_left  = self.K // 2
        pad_right = self.K - 1 - pad_left

        for i in range(self.levels):
            # filtri low/high per la scala corrente
            lo = self._l1norm(self._resample(self.lo_proto, s[i], self.K))
            hi = self._l1norm(self._resample(self.hi_proto, s[i], self.K))

            # (C,1,K) — stesso filtro per tutti i canali, conv depthwise
            lo_w = lo.view(1, 1, self.K).repeat(C, 1, 1)
            hi_w = hi.view(1, 1, self.K).repeat(C, 1, 1)

            # padding same + conv1d depthwise (groups=C)
            a_pad = F.pad(a_prev, (pad_left, pad_right))
            a_low  = F.conv1d(a_pad, lo_w, groups=C)      # (B,C,T)
            detail = F.conv1d(a_pad, hi_w, groups=C)      # (B,C,T)

            # soft-threshold opzionale (per-livello)
            if self.use_soft_threshold:
                tau_i = F.softplus(self.tau_raw[i]).to(detail.device, detail.dtype).view(1, 1, 1)
                detail = self._soft_thresh(detail, tau_i)

            # === Level-Dropout: spegni l'intero livello i ===
            if self.training and self.level_dropout_p > 0:
                if self.level_dropout_mode == "per_sample":
                    keep = torch.bernoulli(
                        torch.full((B, 1, 1), 1.0 - self.level_dropout_p, device=detail.device, dtype=detail.dtype)
                    )
                else:  # una sola maschera per tutto il batch
                    keep = torch.bernoulli(
                        torch.tensor(1.0 - self.level_dropout_p, device=detail.device, dtype=detail.dtype)
                    ).view(1, 1, 1)
                detail = detail * keep

            # gain per livello (sigmoide + scala morbida)
            alpha_i = torch.sigmoid(self.alpha[i]) * self.alpha_scale
            detail = alpha_i * detail
            details_sum = details_sum + detail
            detail_list.append(detail)
            a_prev = a_low

        # Fusione spettrale attentiva: pesa dinamicamente i livelli RDWT
        if self.use_spectral_attention and self.band_attn is not None and len(detail_list) == self.levels:
            band_energy = torch.stack([d.pow(2).mean(dim=(1,2)) for d in detail_list], dim=1)  # (B, L)
            weights = self.band_attn(band_energy)  # (B, L)
            fused = 0.0
            for i, d in enumerate(detail_list):
                fused = fused + d * weights[:, i].view(-1, 1, 1)
            details_sum = fused

        y = a_prev + details_sum  # (B,C,T)

        # energy guard (OFF per default)
        if self.energy_guard and self.training:
            eps = 1e-6
            rms_in  = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps   # (B,C,1)
            rms_out = y.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps
            gain = torch.clamp(rms_in / rms_out, max=self.energy_guard_max_gain)
            y = y * gain
        return y


class ParallelRDWTFrontEnd(nn.Module):
    """Parallel ensemble of :class:`RDWTFrontEnd` blocks with learnable fusion.
    Supports branch-dropout and initial jitter of scales to break symmetry.

    NEW: residual_mode=True => restituisce il *residuo* (y - x), per sommarlo al RAW.
    """
    def __init__(
        self,
        level_choices,
        base_kernel_len: int = 16,
        init_wavelet: str = "db4",
        init_dilations=(1.5, 5 / 3, 7 / 4, 9 / 5, 11 / 6, 13 / 7, 15 / 8, 17 / 9, 19 / 10, 21 / 11),
        use_soft_threshold: bool = True,
        threshold_init: float = 0.0,
        max_scale: float = 4.0,
        l2_on_logscale: float = 0.0,
        use_spread_loss: bool = True,
        spread_lambda: float = 5e-3,
        spread_gamma: float = 4.0,
        temp_init: float = 0.5,
        # NEW: dropouts & jitter
        level_dropout_p: float = 0.30,
        level_dropout_mode: str = "per_sample",
        branch_drop_prob: float = 0.15,
        init_jitter_std: float = 0.05,
        # NEW: residual mode
        residual_mode: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(level_choices, int):
            level_choices = [level_choices]
        self.level_choices = tuple(int(l) for l in level_choices)
        if len(self.level_choices) == 0:
            raise ValueError("`level_choices` must contain at least one level.")

        max_requested_level = max(self.level_choices)
        init_dilations = tuple(init_dilations)
        if len(init_dilations) < max_requested_level:
            # Autopad: ripete l'ultima dilatazione finché basta
            init_dilations = init_dilations + (init_dilations[-1],) * (max_requested_level - len(init_dilations))

        self.front_ends = nn.ModuleList(
            [
                RDWTFrontEnd(
                    levels=levels,
                    base_kernel_len=base_kernel_len,
                    init_wavelet=init_wavelet,
                    init_dilations=init_dilations,
                    use_soft_threshold=use_soft_threshold,
                    threshold_init=threshold_init,
                    max_scale=max_scale,
                    l2_on_logscale=l2_on_logscale,
                    use_spread_loss=use_spread_loss,
                    spread_lambda=spread_lambda,
                    spread_gamma=spread_gamma,
                    temp_init=temp_init,
                    level_dropout_p=level_dropout_p,
                    level_dropout_mode=level_dropout_mode,
                )
                for levels in self.level_choices
            ]
        )

        # Branch-dropout prob
        self.branch_drop_prob = float(branch_drop_prob)

        # Jitter iniziale delle scale per ramo (rompe la simmetria)
        if init_jitter_std and init_jitter_std > 0:
            for fe in self.front_ends:
                with torch.no_grad():
                    fe.z.add_(torch.randn_like(fe.z) * float(init_jitter_std))

        self.fusion_logits = nn.Parameter(torch.zeros(len(self.front_ends)))
        self.register_buffer(
            "_level_choices_tensor",
            torch.tensor(self.level_choices, dtype=torch.int32),
            persistent=False,
        )

        self.residual_mode = bool(residual_mode)

    @property
    def fusion_weights(self) -> Tensor:
        """Return the current softmax-normalised fusion weights."""
        return torch.softmax(self.fusion_logits, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        if len(self.front_ends) == 1:
            y = self.front_ends[0](x)
            return (y - x) if self.residual_mode else y

        branch_outputs = [frontend(x) for frontend in self.front_ends]  # [R, B, C, T]
        stacked = torch.stack(branch_outputs, dim=0)
        weights = self.fusion_weights.view(-1, 1, 1, 1)

        # === Branch-Dropout in training ===
        if self.training and self.branch_drop_prob > 0:
            keep = torch.bernoulli(
                torch.full((weights.shape[0],), 1.0 - self.branch_drop_prob, device=x.device)
            )
            if keep.sum() == 0:
                idx = torch.randint(0, keep.numel(), (1,), device=x.device)
                keep[idx] = 1.0
            weights = weights * keep.view(-1, 1, 1, 1)
            weights = weights / (weights.sum() + 1e-6)

        y = (weights * stacked).sum(dim=0)  # (B,C,T)

        # restituisci residuo se richiesto
        if self.residual_mode:
            return y - x
        return y


# ========================================================
# 1) Gate ADATTIVO PER SOGGETTO (Molto più semplice)
#    SOSTITUISCE IL VECCHIO AdaptiveRDWTGate
# ========================================================
class SubjectAdaptiveGate(nn.Module):
    """
    Un gate "soft" con un singolo logit trainabile.
    Impara un singolo valore di mix (g) per l'intero soggetto,
    invece di calcolarlo dinamicamente per ogni batch.
    
    out = raw + g * delta
    """
    def __init__(
        self,
        tau: float = 1.0,           # Temperatura fissa
        initial_bias: float = -2.0  # Inizia "spento" (sigmoid(-2) ≈ 0.12)
    ):
        super().__init__()
        self.tau = torch.tensor(float(tau))
        # UNICO parametro trainabile
        #self.logit = nn.Parameter(torch.tensor(float(initial_bias)))
        self.initial_bias = float(initial_bias)
        self.logit = nn.Parameter(torch.tensor(self.initial_bias))

        
        # buffer per logging (per compatibilità con il tuo script)
        self.register_buffer("_last_g_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("_l0_reg", torch.tensor(0.0), persistent=False) # Non usato

    def extra_loss(self) -> Tensor:
        return self._l0_reg # Sarà sempre 0

    def get_gate_value(self) -> Tensor:
        """ Ritorna il valore soft corrente del gate """
        # Usiamo tau solo in training per "ammorbidire" il gradiente
        temp = self.tau if self.training else 1.0
        g = torch.sigmoid(self.logit / temp)
        return g

    def forward(self, raw: Tensor, delta: Tensor) -> Tensor:
        """
        raw:   (B,C,T)
        delta: (B,C,T)  ← residuo RDWT
        """
        g = self.get_gate_value()
        
        # Salva il valore per il logging
        self._last_g_mean = g.detach()

        # Applica il gate (si espande automaticamente a B,C,T)
        return raw + g * delta, g


# ===========================================
# 2) HybridParallelFrontEnd ADATTIVO (drop-in)
# ===========================================
class HybridParallelFrontEnd(nn.Module):
    """
    Front-end ibrido RAW + RDWT-residuo con gate adattivo:
         out = RAW + g * (RDWT(x) - x)     [se residual_mode=True nel ramo RDWT]
    Il gate g è:
      - binario con straight-through Gumbel (training)
      - deterministico (eval)
    Granularità: per-batch (default) o per-canale.
    """
    def __init__(
        self,
        rdwt_config,
        fusion_method="learned",     # "learned" usa il gate; "attention"/"concat" restano disponibili
        # --- Parametri del vecchio gate rimossi ---
        # gate_granularity="batch",
        # gate_tau_init=0.5,
        # gate_l0_lambda=1e-3,
        # gate_clamp=(0.0, 1.0),
        # --- Nuovo parametro per SubjectAdaptiveGate ---
        gate_initial_bias: float = -2.0,
    ):
        super().__init__()
        # Ramo RAW
        self.raw_branch = nn.Identity()

        # Ramo RDWT (usiamo il residuo)
        rdwt_config = dict(rdwt_config)
        rdwt_config.setdefault("residual_mode", True)
        self.rdwt_branch = ParallelRDWTFrontEnd(**rdwt_config)

        # Gate adattivo (solo per "learned")
        self.fusion_method = fusion_method
        if fusion_method == "learned":
            # --- VECCHIO CODICE ---
            # self.gate = AdaptiveRDWTGate(
            #     granularity=gate_granularity,
            #     tau_init=gate_tau_init,
            #     l0_lambda=gate_l0_lambda,
            #     clamp_range=gate_clamp,
            # )
            # --- NUOVO CODICE ---
            self.gate = SubjectAdaptiveGate(
                initial_bias=gate_initial_bias
            )
        elif fusion_method == "attention":
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(2, 2, kernel_size=1),
                nn.Softmax(dim=1)
            )
        elif fusion_method == "concat":
            pass
        else:
            raise ValueError(f"Metodo di fusione non supportato: {fusion_method}")

        # per diagnostica/reset
        self.register_buffer("_last_gate_mean", torch.tensor(0.0), persistent=False)
        self.force_raw = False  # bypass totale opzionale

    @torch.no_grad()
    def reset_subject(self):
        """Reset consigliato quando cambi soggetto: riporta il gate a una policy neutra."""
        if self.fusion_method == "learned":
            # --- NUOVO RESET PER SubjectAdaptiveGate ---
            self.gate.logit.data.fill_(self.gate.initial_bias) # Resetta al bias iniziale
            # --- VECCHIO RESET ---
            # self.gate.w.data.copy_(torch.tensor([1.0, 1.0, -2.0], dtype=self.gate.w.dtype, device=self.gate.w.device))
            # self.gate.set_temperature(0.5)

        # resetta anche i rami RDWT (scale/gain/soglie)
        for fe in self.rdwt_branch.front_ends:
            fe.alpha.data.fill_(1.0)
            if fe.use_soft_threshold:
                fe.tau_raw.data.fill_(0.0)
            fe.z.data.copy_(fe.z_mu)

        # opzionale: azzera i pesi di fusione tra rami RDWT paralleli
        self.rdwt_branch.fusion_logits.data.zero_()

    def forward(self, x):
        raw = self.raw_branch(x)          # (B,C,T)
        delta = self.rdwt_branch(x)       # (B,C,T)  residuo RDWT

        if self.force_raw:
            return raw

        if self.fusion_method == "learned":
            out, g = self.gate(raw, delta)           # out = raw + g*delta
            self._last_gate_mean = self.gate._last_g_mean
            # Esponi una "extra loss" opzionale per chi vuole sommarla alla CE
            self._gate_reg = self.gate.extra_loss()
            return out

        elif self.fusion_method == "attention":
            # Fusione alternativa: attenzione tra RAW e RAW+Δ
            stacked = torch.stack([raw, raw + delta], dim=1)  # (B,2,C,T)
            desc = stacked.mean(dim=-1)                       # (B,2,C)
            w = self.channel_attention(desc)                  # (B,2,C)
            return (stacked * w.unsqueeze(-1)).sum(dim=1)

        else:  # "concat": concatena RAW e RAW+Δ
            return torch.cat([raw, raw + delta], dim=1)


class MultiKernelConvBlock(nn.Module):
    """
    Multi-Kernel Convolution Block for EEG Feature Extraction.
    This block applies multiple temporal convolutions with different kernel sizes,
    followed by channel-wise and temporal processing with optional group attention.
    """
    def __init__(
        self,
        n_channels: int,
        temp_kernel_lengths: tuple = (20, 32, 64),
        F1: int = 32,
        D: int = 2,
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        dropout: float = 0.4,
        d_group: int = 16,
        use_group_attn: bool = True,
    ):
        super().__init__()
        # --- 1. one temporal conv per kernel -----------------
        self.rearrange = Rearrange("b c seq -> b 1 c seq")
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad2d((k//2-1, k//2, 0, 0) if k % 2 == 0 else (k//2, k//2, 0, 0), 0),
                nn.Conv2d(1, F1, (1, k), bias=False),
                nn.BatchNorm2d(F1),
            )
            for k in temp_kernel_lengths
        ])
        # --- 2. shared processing after concatenation --------
        n_groups = len(temp_kernel_lengths)
        self.d_model = d_group * n_groups

        # Channel Reduction Stage 1 (disattivata per default)
        self.use_channel_reduction_1  = False
        if self.use_channel_reduction_1:
            self.channel_reduction_1 = nn.Sequential(
                nn.Conv2d(F1 * n_groups, self.d_model, (1, 1), bias=False, groups=n_groups),
                nn.BatchNorm2d(self.d_model),
            )

        # Depth-wise convolution across EEG channels
        F2 = self.d_model * D if self.use_channel_reduction_1 else F1 * n_groups * D
        self.channel_DW_conv = nn.Sequential(
            nn.Conv2d(F1 * n_groups, F2, (n_channels, 1), bias=False, groups=F1 * n_groups),
            nn.BatchNorm2d(F2),
            nn.ELU(),
        )
        self.pool1 = nn.AvgPool2d((1, pool_length_1))
        self.drop1 = nn.Dropout(dropout)

        # Channel Reduction Stage 2: Grouped Pointwise (1×1) Conv (F2 → d_model)
        self.use_channel_reduction_2 = (self.d_model != F2)
        if self.use_channel_reduction_2:
            self.channel_reduction_2 = nn.Sequential(
                nn.Conv2d(F2, self.d_model, (1, 1), bias=False, groups=n_groups),
                nn.BatchNorm2d(self.d_model),
            )

        # Grouped temporal convolution (1 × 16) per group
        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, (1, 16), padding='same', bias=False, groups=n_groups),
            nn.BatchNorm2d(self.d_model),
            nn.ELU(),
        )

        # Grouped attention opzionale
        self.use_group_attn = False if n_groups == 1 else use_group_attn
        if self.use_group_attn:
            self.group_attn = ChannelGroupAttention(
                in_channels=self.d_model,
                num_groups=n_groups,
            )
        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.drop2 = nn.Dropout(dropout)

        # Initialize weights
        glorot_weight_zero_bias(self)

    def forward(self, x):
        # --- 1. one temporal conv per kernel -----------------
        x = self.rearrange(x)         # (B, 1, C, T)
        feats = [conv(x) for conv in self.temporal_convs]  # list of (B, F1, C, T')
        x = torch.cat(feats, dim=1)   # [B, F1 * n_groups, C, T]

        # --- 2. shared processing after concatenation --------
        if self.use_channel_reduction_1:
            x = self.channel_reduction_1(x)
        x = self.channel_DW_conv(x)
        x = self.pool1(x)
        x = self.drop1(x)

        if self.use_channel_reduction_2:
            x = self.channel_reduction_2(x)
        x = self.temporal_conv_2(x)

        if self.use_group_attn:
            x = x + self.group_attn(x)  # Residual
        x = self.pool2(x)
        x = self.drop2(x)
        return x.squeeze(2)


class TCNBlock(nn.Module):
    def __init__(self, kernel_length: int = 4, n_filters: int = 32, dilation: int = 1,
                 n_groups: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv1 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation, groups=n_groups)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.nonlinearity1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation, groups=n_groups)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.nonlinearity2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)
        self.nonlinearity3 = nn.ELU()
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, input):
        x = self.drop1(self.nonlinearity1(self.bn1(self.conv1(input))))
        x = self.drop2(self.nonlinearity2(self.bn2(self.conv2(x))))
        x = self.nonlinearity3(input + x)
        return x


class TCN(nn.Module):
    def __init__(self, depth: int = 2, kernel_length: int = 4, n_filters: int = 32,
                 n_groups: int = 1, dropout: float = 0.3):
        super(TCN, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            self.blocks.append(TCNBlock(kernel_length, n_filters, dilation, n_groups, dropout))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ClassificationHead(nn.Module):
    """
    Maps TCN features to class logits and optionally averages across groups.
    Expected input shape: (batch, d_model, 1)   ← after time-step selection.
    Output shape:        (batch, n_classes)
    """
    def __init__(
        self,
        d_features: int,
        n_groups: int,
        n_classes: int,
        kernel_size: int = 1,
        max_norm: float = 0.25,
    ):
        super().__init__()
        self.n_groups   = n_groups
        self.n_classes  = n_classes
        # point-wise (1 × 1) grouped conv = class projection per group
        self.linear = Conv1dWithConstraint(
            in_channels=d_features,
            out_channels=n_classes * n_groups,
            kernel_size=kernel_size,
            groups=n_groups,
            max_norm=max_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_model, Tc)  →  logits: (B, n_classes)
        """
        x = self.linear(x).squeeze(-1)                     # (B, n_classes*n_groups)
        x = x.view(x.size(0), self.n_groups, self.n_classes).mean(dim=1)
        return x


class TCNHead(nn.Module):
    def __init__(self, d_features: int = 64, n_groups: int = 1, tcn_depth: int = 2,
                 kernel_length: int = 4,  dropout_tcn: float = 0.3, n_classes: int = 4):
        super().__init__()
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.tcn = TCN(tcn_depth, kernel_length, d_features, n_groups, dropout_tcn)
        self.classifier = ClassificationHead(
            d_features=d_features,
            n_groups=n_groups,
            n_classes=n_classes,
        )

    def forward(self, x):
        x = self.tcn(x)
        x = x[:, :, -1:]
        x = self.classifier(x)
        return x


class DropPath(nn.Module):
    """Stochastic Depth / DropPath (per-sample)"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class _GQAttention(nn.Module):
    """Grouped-Query Attention (num_q_heads >= num_kv_heads) with RoPE."""
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int, dropout: float = 0.3):
        super().__init__()
        assert d_model % num_q_heads == 0, "d_model must divide num_q_heads"
        assert num_q_heads % num_kv_heads == 0, "q_heads must be multiple of kv_heads"
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        _xavier_zero_bias(self)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:  # x (B,T,C)
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)  # (B, hq, T, d)
        kv = self.kv_proj(x)
        kv = kv.view(B, T, self.num_kv_heads, 2, self.head_dim)
        k, v = kv[..., 0, :].transpose(1, 2), kv[..., 1, :].transpose(1, 2)  # (B, hk, T, d)
        repeat_factor = self.num_q_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        q, k = _rope(q, k, cos[:T, :], sin[:T, :])
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B,h,T,T)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = attn @ v                                  # (B,h,T,d)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


def _build_rotary_cache(head_dim: int, seq_len: int, device: torch.device):
    """Return cos & sin tensors of shape (seq_len, head_dim)."""
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    seq_idx = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(seq_idx, theta)                 # (seq, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)             # duplicate for even/odd
    cos, sin = emb.cos(), emb.sin()
    return cos, sin                                     # each: (seq, head_dim)


def _rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):  # q/k: (B, h, T, d)
    def _rotate(x):                                        # half rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    q_out = (q * cos) + (_rotate(q) * sin)
    k_out = (k * cos) + (_rotate(k) * sin)
    return q_out, k_out


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, q_heads: int, kv_heads: int, mlp_ratio: int = 2, dropout=0.4, drop_path_rate=0.25):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _GQAttention(d_model, q_heads, kv_heads, dropout)
        self.drop_path   = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), cos, sin))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# === MODELLO PRINCIPALE CON FUSIONE IBRIDA ===
class TCFormerModule(nn.Module):
    def __init__(self,
            n_channels: int,
            n_classes: int,
            F1: int = 16,
            temp_kernel_lengths=(16, 32, 64),
            pool_length_1: int = 8,
            pool_length_2: int = 7,
            D: int = 2,
            dropout_conv: float = 0.3,
            d_group: int = 16,
            tcn_depth: int = 2,
            kernel_length_tcn: int = 4,
            dropout_tcn: float = 0.3,
            use_group_attn: bool = True,
            kv_heads: int = 4,
            q_heads: int = 8,
            trans_dropout: float = 0.4,
            drop_path_max: float = 0.25,
            trans_depth: int = 5,
            # -------- FRONTEND IBRIDO ----------
            use_hybrid: bool = True,
            fusion_method: str = "learned",  # "learned", "attention", "concat"
            # --- Nuovo parametro per SubjectAdaptiveGate ---
            gate_initial_bias: float = -2.0,
            # -------- RDWT ----------
            #rdwt_levels: int = 4,
            rdwt_level_choices=tuple(range(2, 11)),
            rdwt_base_kernel_len: int = 16,
            rdwt_init_dilations=(1.5, 5 / 3, 7 / 4, 9 / 5, 11 / 6, 13 / 7, 15 / 8, 17 / 9, 19 / 10, 21 / 11),
            rdwt_soft_threshold: bool = True,
            rdwt_threshold_init: float = 0.0,
            rdwt_max_scale: float = 4.0,
            rdwt_l2_on_logscale: float = 0.0,
            rdwt_use_spread_loss: bool = True,
            rdwt_spread_lambda: float = 5e-3,
            rdwt_spread_gamma: float = 4.0,
            rdwt_temp_init: float = 0.5,
            # RDWT regularization/exploration
            rdwt_level_dropout_p: float = 0.30,
            rdwt_level_dropout_mode: str = "per_sample",
            rdwt_branch_drop_prob: float = 0.15,
            rdwt_init_jitter_std: float = 0.05,
            # -------- Policy di training ----------
            hybrid_auto_policy: str = "off",  # "off" | "auto_conf"
            policy_eval_prob: float = 0.02, 
            policy_margin: float = 0.02,
            policy_lock_after: int = 5
            
        ):
        super().__init__()
        self.n_classes = n_classes
        self.n_groups = len(temp_kernel_lengths)
        self.d_model = d_group * self.n_groups

        # ---------- FRONTEND IBRIDO ----------
        self.use_hybrid = use_hybrid
        self.fusion_method = fusion_method
        self._gate_reg = torch.tensor(0.0)  # placeholder per extra loss opzionale
        
        # Policy di training ibrido 
        self.hybrid_auto_policy = hybrid_auto_policy
        self.policy_eval_prob   = float(policy_eval_prob)
        self.policy_margin      = float(policy_margin)
        self.policy_lock_after  = int(policy_lock_after)
        self._policy_votes = 0
    
        if self.use_hybrid:
            rdwt_config = {
                'level_choices': rdwt_level_choices,
                'base_kernel_len': rdwt_base_kernel_len,
                'init_dilations': rdwt_init_dilations,
                'use_soft_threshold': rdwt_soft_threshold,
                'threshold_init': rdwt_threshold_init,
                'max_scale': rdwt_max_scale,
                'l2_on_logscale': rdwt_l2_on_logscale,
                'use_spread_loss': rdwt_use_spread_loss,
                'spread_lambda': rdwt_spread_lambda,
                'spread_gamma': rdwt_spread_gamma,
                'temp_init': rdwt_temp_init,
                'level_dropout_p': rdwt_level_dropout_p,
                'level_dropout_mode': rdwt_level_dropout_mode,
                'branch_drop_prob': rdwt_branch_drop_prob,
                'init_jitter_std': rdwt_init_jitter_std,
                # fondamentale: usa residuo
                'residual_mode': True,
            }
            self.hybrid_frontend = HybridParallelFrontEnd(
                rdwt_config=rdwt_config,
                fusion_method=fusion_method,
                gate_initial_bias=gate_initial_bias # Passa il nuovo parametro
            )

            # canali effettivi per il backbone
            if fusion_method == "concat":
                effective_channels = n_channels * 2
            else:
                effective_channels = n_channels
        else:
            effective_channels = n_channels
            self.rdwt_branch = None

        self.rearrange = Rearrange("b c seq -> b seq c")
        self.conv_block = MultiKernelConvBlock(effective_channels, temp_kernel_lengths, F1, D,
                                               pool_length_1, pool_length_2, dropout_conv,
                                               d_group, use_group_attn)
        self.mix = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.SiLU()
        )

        drop_rates = torch.linspace(0, 1, trans_depth) ** 2 * drop_path_max
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)
        self.transformer = nn.ModuleList([
            _TransformerBlock(self.d_model, q_heads, kv_heads, dropout=trans_dropout,
                              drop_path_rate=drop_rates[i].item())
            for i in range(trans_depth)
        ])

        # Adatta la dimensionalità per la concatenazione finale
        if self.use_hybrid and fusion_method == "concat":
            final_d_group = d_group * 2
        else:
            final_d_group = d_group

        self.reduce = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(self.d_model, final_d_group, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm1d(final_d_group),
            nn.SiLU(),
        )

        # Calcola i gruppi finali per la testa TCN
        if self.use_hybrid and fusion_method == "concat":
            final_n_groups = self.n_groups + 2
        else:
            final_n_groups = self.n_groups + 1

        self.tcn_head = TCNHead(final_d_group * final_n_groups, final_n_groups,
                                tcn_depth, kernel_length_tcn, dropout_tcn, n_classes)
        
        # --- AGGIUNTA PER LOGGING ---
        self.register_buffer("_last_gate_mean", torch.tensor(0.0), persistent=False)


    def set_force_raw(self, flag: bool):
        """Opzionale: forza il bypass RAW (utile per distillazione/diagnostica)."""
        if self.use_hybrid:
            self.hybrid_frontend.force_raw = bool(flag)

    def forward(self, x_in):      # x_in: [B, C, T]  <-- rinomina: è l'input GREZZO
        # --- Frontend processing (FUSED path) ---
        if self.use_hybrid:
            x_fused = self.hybrid_frontend(x_in)   # (B,C,T) o (B,2*C,T)
            if hasattr(self.hybrid_frontend, "_gate_reg"):
                self._gate_reg = self.hybrid_frontend._gate_reg
            if hasattr(self.hybrid_frontend, "_last_gate_mean"):
                self._last_gate_mean = self.hybrid_frontend._last_gate_mean
        elif hasattr(self, 'rdwt_branch') and self.rdwt_branch is not None:
            x_fused = self.rdwt_branch(x_in)
        else:
            x_fused = x_in

        # --- Backbone (FUSED) ---
        conv_features = self.conv_block(x_fused)                     # (B, d_model, T1)
        tokens = self.rearrange(self.mix(conv_features))             # (B, T1, d_model)
        cos, sin = self._rotary_cache(tokens.shape[1], tokens.device)
        for blk in self.transformer:
            tokens = blk(tokens, cos, sin)
        tran_features = self.reduce(tokens)                          # (B, d_group, T1)
        features = torch.cat((conv_features, tran_features), dim=1)  # (B, d_model+d_group, T1)
        out = self.tcn_head(features)                                 # (B, n_classes)

        # --- POLICY: auto_conf per spegnere RDWT sui "forti" ---
        if (self.training and self.use_hybrid and self.fusion_method == "learned"
            and self.hybrid_auto_policy == "auto_conf"):
            if torch.rand(()) < self.policy_eval_prob:
                with torch.no_grad():
                    # 1) conf (FUSED)
                    p_fused = torch.softmax(out, dim=-1).amax(dim=-1).mean()

                    # 2) conf (RAW) -> ricalcola TUTTO partendo dall'input GREZZO
                    self.set_force_raw(True)            # bypass RDWT/gate
                    x_raw = self.hybrid_frontend(x_in)  # con force_raw=True ritorna il RAW
                    self.set_force_raw(False)

                    conv_r = self.conv_block(x_raw)
                    tokens_r = self.rearrange(self.mix(conv_r))
                    # usa la stessa cache RoPE (se T coincide) o ricostruiscila
                    if tokens_r.shape[1] != tokens.shape[1]:
                        cos_r, sin_r = self._rotary_cache(tokens_r.shape[1], tokens_r.device)
                    else:
                        cos_r, sin_r = cos, sin
                    for blk in self.transformer:
                        tokens_r = blk(tokens_r, cos_r, sin_r)
                    tran_r = self.reduce(tokens_r)
                    feat_r = torch.cat((conv_r, tran_r), dim=1)
                    out_raw = self.tcn_head(feat_r)

                    p_raw = torch.softmax(out_raw, dim=-1).amax(dim=-1).mean()

                    # 3) voto: se RAW è più confidente del FUSED di un margine
                    if (p_raw - p_fused) > self.policy_margin:
                        self._policy_votes += 1
                        if self._policy_votes >= self.policy_lock_after:
                            # da qui in poi il soggetto usa sempre RAW (bypass RDWT)
                            self.set_force_raw(True)

        return out


    def _rotary_cache(self, seq_len: int, device: torch.device):
        """Build (or reuse) RoPE caches for the current sequence length."""
        head_dim = self.transformer[0].attn.head_dim
        if (self._cos is None) or (self._cos.shape[0] < seq_len):
            cos, sin = _build_rotary_cache(head_dim, seq_len, device)
            self._cos, self._sin = cos.to(device), sin.to(device)
        return self._cos, self._sin


# -----------------------------------------------------------------------------
# 0.  helpers
# -----------------------------------------------------------------------------
def _xavier_zero_bias(module: nn.Module) -> None:
    """Apply Xavier-uniform + zero bias to every conv/linear inside *module*."""
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# -----------------------------------------------------------------------------
# 1.  Rotary positional embedding utilities (defined above)
# -----------------------------------------------------------------------------

class RatioWaveNet(ClassificationModule):
    def __init__(self,
        n_channels: int,
        n_classes: int,
        F1: int = 16,
        temp_kernel_lengths=(16, 32, 64),
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        D: int = 2,
        dropout_conv: float = 0.3,
        d_group: int = 16,
        tcn_depth: int = 2,
        kernel_length_tcn: int = 4,
        dropout_tcn: float = 0.3,
        use_group_attn: bool = True,
        q_heads: int = 8,
        kv_heads: int = 4,
        trans_depth: int = 5,
        trans_dropout: float = 0.4,
        # --- FRONTEND IBRIDO ---
        use_hybrid: bool = True,
        fusion_method: str = "learned",
        # --- Nuovo parametro per SubjectAdaptiveGate ---
        gate_initial_bias: float = -2.0,
        # --- RDWT params ---
        #rdwt_levels: int = 4,
        rdwt_level_choices=tuple(range(2, 11)),
        rdwt_base_kernel_len: int = 16,
        rdwt_init_dilations=(1.5, 5 / 3, 7 / 4, 9 / 5, 11 / 6, 13 / 7, 15 / 8, 17 / 9, 19 / 10, 21 / 11),
        rdwt_soft_threshold: bool = True,
        rdwt_threshold_init: float = 0.0,
        rdwt_max_scale: float = 4.0,
        rdwt_l2_on_logscale: float = 0.0,
        rdwt_use_spread_loss: bool = True,
        rdwt_spread_lambda: float = 5e-3,
        rdwt_spread_gamma: float = 4.0,
        rdwt_temp_init: float = 0.5,
        # exploration / regularization knobs
        rdwt_level_dropout_p: float = 0.30,
        rdwt_level_dropout_mode: str = "per_sample",
        rdwt_branch_drop_prob: float = 0.15,
        rdwt_init_jitter_std: float = 0.05,
        **kwargs
    ):
        model = TCFormerModule(
            n_channels=n_channels,
            n_classes=n_classes,
            F1=F1,
            temp_kernel_lengths=temp_kernel_lengths,
            pool_length_1=pool_length_1,
            pool_length_2=pool_length_2,
            D=D,
            dropout_conv=dropout_conv,
            d_group=d_group,
            tcn_depth=tcn_depth,
            kernel_length_tcn=kernel_length_tcn,
            dropout_tcn=dropout_tcn,
            use_group_attn=use_group_attn,
            q_heads=q_heads, kv_heads=kv_heads,
            trans_depth=trans_depth, trans_dropout=trans_dropout,
            # FRONTEND IBRIDO
            use_hybrid=use_hybrid,
            fusion_method=fusion_method,
            gate_initial_bias=gate_initial_bias, # Passa il nuovo parametro
            # RDWT
            #rdwt_levels=rdwt_levels,
            rdwt_level_choices=rdwt_level_choices,
            rdwt_base_kernel_len=rdwt_base_kernel_len,
            rdwt_init_dilations=rdwt_init_dilations,
            rdwt_soft_threshold=rdwt_soft_threshold,
            rdwt_threshold_init=rdwt_threshold_init,
            rdwt_max_scale=rdwt_max_scale,
            rdwt_l2_on_logscale=rdwt_l2_on_logscale,
            rdwt_use_spread_loss=rdwt_use_spread_loss,
            rdwt_spread_lambda=rdwt_spread_lambda,
            rdwt_spread_gamma=rdwt_spread_gamma,
            rdwt_temp_init=rdwt_temp_init,
            rdwt_level_dropout_p=rdwt_level_dropout_p,
            rdwt_level_dropout_mode=rdwt_level_dropout_mode,
            rdwt_branch_drop_prob=rdwt_branch_drop_prob,
            rdwt_init_jitter_std=rdwt_init_jitter_std,
        )
        super().__init__(model, n_classes, **kwargs)
        
        # --- AGGIUNTA PER LOGGING ---
        # Buffer per accumulare i valori del gate durante il test
        self.register_buffer("test_gate_values", torch.empty(0), persistent=False)
        # Questo attributo è cercato da train_pipeline.py
        self.rdwt_summary = [0.0] # Inizializza come lista

    # Opzionale: fornisce extra-loss (se vuoi sommarla nel tuo training_step)
    def get_extra_losses(self):
        device = next(self.parameters()).device
        m = self.model
        return {
            "gate_l1": getattr(m, "_gate_reg", torch.tensor(0.0, device=device))
        }

    @staticmethod
    def benchmark(input_shape, device="cuda:0", warmup=100, runs=500):
        return measure_latency(RatioWaveNet(22, 4), input_shape, device, warmup, runs)


if __name__ == "__main__":
    # Example usage: run benchmark with dummy input shape (batch, channels, time)
    C, T = 22, 1000  # adjust as needed
    print(RatioWaveNet.benchmark((1, C, T)))
