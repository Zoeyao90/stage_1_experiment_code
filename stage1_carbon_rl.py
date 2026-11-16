import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)


# ==============================================================================
# 1. HARDWARE, MODEL & LITERATURE PARAMETERS (Chapter 2, Sections 2.3 & 2.4)
# ==============================================================================
class SimulationConfig:
    # Control horizon: 24 hours at 10-second intervals = 8,640 steps per episode
    STEP_SEC = 10.0
    STEPS_PER_DAY = int(24 * 3600 / STEP_SEC)

    # Hardware: NVIDIA A100-80GB SXM4 (Vidur-Energy & LLMCarbon Table 3)
    GPU_VRAM_GB = 80.0
    P_IDLE_W = 100.0           # Vidur-Energy Section 4.1
    P_MAX_INST_W = 400.0       # Vidur-Energy Section 4.1
    MFU_SAT = 0.45             # Vidur-Energy Section 4.1
    POWER_GAMMA = 0.70         # Vidur-Energy Section 4.1
    PUE = 1.30                 # Cluster PUE (set to 1.0 for local workstation)

    # Embodied Carbon (LLMCarbon Section 4.7 & Table 3)
    # A100/H100 class ~814-826 mm2 (~8.2 cm2) at ~1.5 kgCO2/cm2 + host share ~ 15 kgCO2/GPU
    GPU_EMBODIED_KG_CO2 = 15.0
    LIFETIME_SEC = 5.0 * 365.25 * 24 * 3600  # 5-year hardware lifespan

    # Model Specs (Llama-3.1-70B-Instruct default; toggle to "8B" for local single-GPU)
    MODEL_NAME = "Llama-3.1-70B"
    MODEL_PARAMS_B = 70.0      # Change to 8.0 for Llama-3.1-8B

    # Interactive Latency SLOs (Vidur Section 7.3)
    SLO_TTFT_SEC = 2.0         # P90 TTFT < 2.0 s
    SLO_TBT_SEC = 0.200        # P99 TBT < 200 ms

    # Workload Profile: "LMSys-Chat-1M" (P:D = 2.3) or "Arxiv-Sum" (P:D = 15.7)
    WORKLOAD_TYPE = "LMSys-Chat-1M"

    # Reward Hyperparameters (Eq. 3)
    ALPHA_OP = 0.45            # Operational carbon weight
    BETA_EMB = 0.25            # Embodied carbon weight
    CARBON_BUDGET_G = 12.0     # Reference step carbon budget B_carb (gCO2eq / 10s)


# ==============================================================================
# 2. ZERO-SHOT SURROGATE ENGINE (Vidur-Energy + TokenPowerBench + LLMCarbon)
# ==============================================================================
class ZeroShotVidurEnergySurrogate:
    """
    Analytical + empirical surrogate replicating Vidur-Energy (Ozcan et al., 2025),
    TokenPowerBench (Niu et al., 2025), and LLMCarbon (Faiz et al., 2024).
    Optionally loads pre-computed CSV traces from a physical Vidur-Energy sweep.
    """
    def __init__(self, cfg: SimulationConfig, vidur_csv_path: str = None):
        self.cfg = cfg
        self.vidur_lookup = None
        if vidur_csv_path and os.path.exists(vidur_csv_path):
            self.vidur_lookup = pd.read_csv(vidur_csv_path)

        # Build the 36 discrete actions: (TP, PP, BS_max, Quant)
        tp_vals = [1, 2, 4]
        pp_vals = [1, 2]
        bs_vals = [8, 16, 32, 64, 128, 256]
        quant_vals = ["FP16", "FP8", "INT4"]

        self.actions = list(itertools.product(tp_vals, pp_vals, bs_vals, quant_vals))

        # Workload token characteristics (Vidur Table 1)
        if cfg.WORKLOAD_TYPE == "LMSys-Chat-1M":
            self.in_tokens = 786
            self.out_tokens = 215
        else:  # Arxiv-Summarization
            self.in_tokens = 2588
            self.out_tokens = 291

        # TokenPowerBench (Fig. 6) & Poddar et al. (Sec. 3.5) quantisation priors
        self.quant_priors = {
            "FP16": {"bytes": 2.0, "energy_mult": 1.00, "speed_mult": 1.00},
            "FP8":  {"bytes": 1.0, "energy_mult": 0.70, "speed_mult": 1.30},
            "INT4": {"bytes": 0.5, "energy_mult": 0.65, "speed_mult": 1.35},
        }

    def is_action_feasible(self, action_idx: int) -> bool:
        """Checks GPU VRAM capacity for model weights + worst-case KV cache."""
        tp, pp, bs_max, quant = self.actions[action_idx]
        num_gpus = tp * pp
        if num_gpus > 4:  # Cap cluster replica at 4 GPUs (matching Ch. 2 Sec. 2.5)
            return False

        total_vram_gb = num_gpus * self.cfg.GPU_VRAM_GB
        bytes_per_param = self.quant_priors[quant]["bytes"]

        # Weight memory + KV cache memory footprint (~0.5 MB per token per sequence for 70B)
        weight_mem_gb = self.cfg.MODEL_PARAMS_B * bytes_per_param
        kv_per_seq_gb = 0.0005 * (self.cfg.MODEL_PARAMS_B / 70.0) * (self.in_tokens + self.out_tokens)
        total_mem_gb = weight_mem_gb + (bs_max * kv_per_seq_gb)

        return total_mem_gb <= (0.90 * total_vram_gb)  # Leave 10% VRAM headroom

    def evaluate_step(self, action_idx: int, qps: float, queue_len: int, carbon_intensity: float):
        """
        Simulates a 10-second serving window under action_idx.
        Returns latency, throughput, energy, operational/embodied carbon, and new queue length.
        """
        tp, pp, bs_max, quant = self.actions[action_idx]
        num_gpus = tp * pp
        prior = self.quant_priors[quant]

        # Incoming requests in this 10s interval
        arrivals = np.random.poisson(qps * self.cfg.STEP_SEC)
        total_demand = queue_len + arrivals

        # Effective parallel scaling (TokenPowerBench Fig. 5: TP beats PP due to pipeline bubbles)
        pp_bubble_penalty = 1.0 if pp == 1 else 0.86
        parallel_efficiency = (num_gpus ** 0.85) * pp_bubble_penalty

        # Base single-GPU service capacity (req/sec) scaled by batch concurrency & quantisation
        model_scale = 70.0 / self.cfg.MODEL_PARAMS_B
        batch_throughput_gain = np.log2(bs_max + 1) / np.log2(33)  # Sublinear batching gain
        max_capacity_qps = (
            1.45 * model_scale * parallel_efficiency * batch_throughput_gain * prior["speed_mult"]
        )

        # Requests processed in this 10s window
        max_possible_reqs = int(max_capacity_qps * self.cfg.STEP_SEC)
        completed_reqs = min(total_demand, max_possible_reqs)
        next_queue_len = max(0, total_demand - completed_reqs)
        active_batch = min(bs_max, max(1, int(completed_reqs / max(1.0, self.cfg.STEP_SEC / 1.5))))

        # 1. Latency Estimation (TTFT & TBT)
        queue_wait_sec = (next_queue_len / max(0.1, max_capacity_qps)) * 0.5
        prefill_sec = (0.35 / (model_scale * parallel_efficiency * prior["speed_mult"])) * (self.in_tokens / 786.0)
        ttft_sec = prefill_sec + queue_wait_sec

        # TBT grows with active batch concurrency (memory-bandwidth contention)
        batch_contention = 1.0 + 0.014 * active_batch
        tbt_sec = (0.042 * batch_contention) / ((num_gpus ** 0.65) * pp_bubble_penalty * prior["speed_mult"])

        # SLO Compliance Ratio (Eq. 4)
        if total_demand == 0:
            slo_ratio = 1.0
        elif completed_reqs == 0:
            slo_ratio = 0.0
        else:
            ttft_ok = 1.0 if ttft_sec <= self.cfg.SLO_TTFT_SEC else max(0.0, 1.0 - (ttft_sec - self.cfg.SLO_TTFT_SEC) / 2.0)
            tbt_ok = 1.0 if tbt_sec <= self.cfg.SLO_TBT_SEC else max(0.0, 1.0 - (tbt_sec - self.cfg.SLO_TBT_SEC) / 0.15)
            slo_ratio = float(np.clip(ttft_ok * tbt_ok, 0.0, 1.0))

        # 2. Model FLOPs Utilisation (MFU) & Vidur-Energy Sublinear Power Model (Eq. 7 & 8)
        util_fraction = np.clip(completed_reqs / max(1.0, max_possible_reqs), 0.0, 1.0)
        mfu = self.cfg.MFU_SAT * (util_fraction ** 0.65)

        if mfu <= 1e-4:
            gpu_power_w = self.cfg.P_IDLE_W
        else:
            gpu_power_w = self.cfg.P_IDLE_W + (self.cfg.P_MAX_INST_W - self.cfg.P_IDLE_W) * (
                (mfu / self.cfg.MFU_SAT) ** self.cfg.POWER_GAMMA
            )

        # Apply TokenPowerBench quantisation energy scaling (Poddar et al.: INT4 overhead at tiny batches)
        quant_energy_mult = prior["energy_mult"]
        if quant == "INT4" and bs_max <= 16:
            quant_energy_mult = 0.95  # Dequantisation overhead at small batch sizes

        # 3. Operational & Embodied Carbon Accounting (Eq. 6 & 9)
        step_hours = self.cfg.STEP_SEC / 3600.0
        energy_op_kwh = (gpu_power_w * quant_energy_mult * num_gpus * step_hours * self.cfg.PUE) / 1000.0
        co2_op_g = energy_op_kwh * carbon_intensity

        # Embodied carbon (LLMCarbon Eq. 12): accrues per allocated GPU over step duration
        co2_emb_g = (self.cfg.STEP_SEC / self.cfg.LIFETIME_SEC) * (self.cfg.GPU_EMBODIED_KG_CO2 * 1000.0) * num_gpus

        return {
            "active_batch": active_batch,
            "next_queue": next_queue_len,
            "completed_reqs": completed_reqs,
            "ttft_sec": ttft_sec,
            "tbt_sec": tbt_sec,
            "slo_ratio": slo_ratio,
            "mfu": mfu,
            "power_w": gpu_power_w * quant_energy_mult * num_gpus,
            "energy_kwh": energy_op_kwh,
            "co2_op_g": co2_op_g,
            "co2_emb_g": co2_emb_g,
            "co2_total_g": co2_op_g + co2_emb_g,
        }


# ==============================================================================
# 3. SYNTHETIC TRACE GENERATOR (Azure Diurnal QPS + Grid Carbon Intensity)
# ==============================================================================
def generate_diurnal_traces(steps: int, seed: int = 42):
    """
    Generates realistic 24-hour traces at 10s resolution:
    - Request arrival rate lambda(t) following Azure diurnal peaks (DynamoLLM Fig. 2)
    - Grid carbon intensity c(t) with midday solar trough & evening peak (120 - 440 gCO2/kWh)
    """
    rng = np.random.default_rng(seed)
    t_hours = np.linspace(0, 24, steps, endpoint=False)

    # Diurnal QPS: low overnight (~0.8 QPS), peaking midday/afternoon (~4.5 QPS) + bursts
    base_qps = 0.8 + 3.4 * np.exp(-0.5 * ((t_hours - 14.0) / 4.0) ** 2)
    bursts = rng.gamma(shape=2.0, scale=0.25, size=steps)
    qps_trace = np.clip(base_qps + bursts, 0.2, 8.0)

    # Diurnal Grid Carbon Intensity (gCO2eq/kWh): solar dip around 12:30, fossil peak at 19:30
