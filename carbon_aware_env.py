import os
import itertools
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


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
    GPU_EMBODIED_KG_CO2 = 15.0
    LIFETIME_SEC = 5.0 * 365.25 * 24 * 3600  # 5-year hardware lifespan

    # Model Specs (Llama-3.1-70B-Instruct default; toggle to 8.0 for single-GPU)
    MODEL_NAME = "Llama-3.1-70B"
    MODEL_PARAMS_B = 70.0

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
        if num_gpus > 4:
            return False

        total_vram_gb = num_gpus * self.cfg.GPU_VRAM_GB
        bytes_per_param = self.quant_priors[quant]["bytes"]

        weight_mem_gb = self.cfg.MODEL_PARAMS_B * bytes_per_param
        kv_per_seq_gb = 0.0005 * (self.cfg.MODEL_PARAMS_B / 70.0) * (self.in_tokens + self.out_tokens)
        total_mem_gb = weight_mem_gb + (bs_max * kv_per_seq_gb)

        return total_mem_gb <= (0.90 * total_vram_gb)

    def evaluate_step(self, action_idx: int, qps: float, queue_len: int, carbon_intensity: float):
        """
        Simulates a 10-second serving window under action_idx.
        Returns latency, throughput, energy, operational/embodied carbon, and new queue length.
        """
        tp, pp, bs_max, quant = self.actions[action_idx]
        num_gpus = tp * pp
        prior = self.quant_priors[quant]

        arrivals = np.random.poisson(qps * self.cfg.STEP_SEC)
        total_demand = queue_len + arrivals

        pp_bubble_penalty = 1.0 if pp == 1 else 0.86
        parallel_efficiency = (num_gpus ** 0.85) * pp_bubble_penalty

        model_scale = 70.0 / self.cfg.MODEL_PARAMS_B
        batch_throughput_gain = np.log2(bs_max + 1) / np.log2(33)
        max_capacity_qps = (
            1.45 * model_scale * parallel_efficiency * batch_throughput_gain * prior["speed_mult"]
        )

        max_possible_reqs = int(max_capacity_qps * self.cfg.STEP_SEC)
        completed_reqs = min(total_demand, max_possible_reqs)
        next_queue_len = max(0, total_demand - completed_reqs)
        active_batch = min(bs_max, max(1, int(completed_reqs / max(1.0, self.cfg.STEP_SEC / 1.5))))

        # 1. Latency Estimation (TTFT & TBT)
        queue_wait_sec = (next_queue_len / max(0.1, max_capacity_qps)) * 0.5
        prefill_sec = (0.35 / (model_scale * parallel_efficiency * prior["speed_mult"])) * (self.in_tokens / 786.0)
        ttft_sec = prefill_sec + queue_wait_sec

        batch_contention = 1.0 + 0.014 * active_batch
        tbt_sec = (0.042 * batch_contention) / ((num_gpus ** 0.65) * pp_bubble_penalty * prior["speed_mult"])

        if total_demand == 0:
            slo_ratio = 1.0
        elif completed_reqs == 0:
            slo_ratio = 0.0
        else:
            ttft_ok = 1.0 if ttft_sec <= self.cfg.SLO_TTFT_SEC else max(0.0, 1.0 - (ttft_sec - self.cfg.SLO_TTFT_SEC) / 2.0)
            tbt_ok = 1.0 if tbt_sec <= self.cfg.SLO_TBT_SEC else max(0.0, 1.0 - (tbt_sec - self.cfg.SLO_TBT_SEC) / 0.15)
            slo_ratio = float(np.clip(ttft_ok * tbt_ok, 0.0, 1.0))

        # 2. Model FLOPs Utilisation (MFU) & Vidur-Energy Sublinear Power Model
        util_fraction = np.clip(completed_reqs / max(1.0, max_possible_reqs), 0.0, 1.0)
        mfu = self.cfg.MFU_SAT * (util_fraction ** 0.65)

        if mfu <= 1e-4:
            gpu_power_w = self.cfg.P_IDLE_W
        else:
            gpu_power_w = self.cfg.P_IDLE_W + (self.cfg.P_MAX_INST_W - self.cfg.P_IDLE_W) * (
                (mfu / self.cfg.MFU_SAT) ** self.cfg.POWER_GAMMA
            )

        quant_energy_mult = prior["energy_mult"]
        if quant == "INT4" and bs_max <= 16:
            quant_energy_mult = 0.95

        # 3. Operational & Embodied Carbon Accounting
        step_hours = self.cfg.STEP_SEC / 3600.0
        energy_op_kwh = (gpu_power_w * quant_energy_mult * num_gpus * step_hours * self.cfg.PUE) / 1000.0
        co2_op_g = energy_op_kwh * carbon_intensity
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
    """Generates 24-hour QPS arrival and grid carbon intensity traces at 10s resolution."""
    rng = np.random.default_rng(seed)
    t_hours = np.linspace(0, 24, steps, endpoint=False)

    base_qps = 0.8 + 3.4 * np.exp(-0.5 * ((t_hours - 14.0) / 4.0) ** 2)
    bursts = rng.gamma(shape=2.0, scale=0.25, size=steps)
    qps_trace = np.clip(base_qps + bursts, 0.2, 8.0)

    solar_dip = -160.0 * np.exp(-0.5 * ((t_hours - 12.5) / 2.5) ** 2)
    evening_peak = 130.0 * np.exp(-0.5 * ((t_hours - 19.5) / 2.5) ** 2)
    ci_trace = np.clip(300.0 + solar_dip + evening_peak + rng.normal(0, 8.0, size=steps), 100.0, 460.0)

    return t_hours, qps_trace, ci_trace


# ==============================================================================
# 4. GYMNASIUM RL ENVIRONMENT (Chapter 2, Section 2.3)
# ==============================================================================
class CarbonAwareLLMEnv(gym.Env):
    """
    Custom Gymnasium Environment for Stage 1 Intra-Cluster Carbon-Aware LLM Inference Control.
    State Space:  [n_p(t), n_q(t), lambda(t), c(t), h(t)]
    Action Space: Discrete feasible configurations (TP, PP, BS_max, Quant) under vLLM.
    Reward:       R_SLO(t) - alpha * R_carbon_op(t) - beta * R_carbon_emb(t)
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: SimulationConfig = None, seed: int = 42, vidur_csv_path: str = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else SimulationConfig()
        self.surrogate = ZeroShotVidurEnergySurrogate(self.cfg, vidur_csv_path=vidur_csv_path)
        self.seed_val = seed

        # Filter VRAM-feasible actions for the target LLM
        self.valid_action_indices = [
            idx for idx in range(len(self.surrogate.actions)) if self.surrogate.is_action_feasible(idx)
        ]
        self.action_space = spaces.Discrete(len(self.valid_action_indices))

        # 5D Continuous State Space: [n_p(t), n_q(t), lambda(t), c(t), h(t)] (Eq. 1)
   self.observation_space = spaces.Box(
   low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
   high=np.array([512.0, 5000.0, 50.0, 1000.0, 1.0], dtype=np.float32),
   dtype=np.float32,
   )
   self.reset(seed=seed)
   def _get_obs(self) -> np.ndarray:
   return np.array(
   [
   self.active_batch / 256.0,
   self.queue_len / 500.0,
   self.qps_trace[self.step_idx] / 10.0,
   self.ci_trace[self.step_idx] / 500.0,
   self.t_hours[self.step_idx] / 24.0,
   ],
   dtype=np.float32,
   )     
   def reset(self, seed=None, options=None):
       super().reset(seed=seed)
       self.step_idx = 0
       self.queue_len = 0
       self.active_batch = 0
       self.t_hours, self.qps_trace, self.ci_trace = generate_diurnal_traces(
       self.cfg.STEPS_PER_DAY, seed=(seed if seed is not None else self.seed_val)
       )
      return self._get_obs(), {} 
   def step(self, mapped_action: int):
       real_action_idx = self.valid_action_indices[mapped_action]
       qps = self.qps_trace[self.step_idx]
       ci = self.ci_trace[self.step_idx]
       metrics = self.surrogate.evaluate_step(real_action_idx, qps, self.queue_len, ci)
       self.queue_len = metrics["next_queue"]
       self.active_batch = metrics["active_batch"]
      # Multi-Objective Reward Function (Chapter 2, Eq. 3 - Eq. 6)
       r_slo = metrics["slo_ratio"]
       r_carbon_op = (metrics["co2_op_g"] - self.cfg.CARBON_BUDGET_G) / self.cfg.CARBON_BUDGET_G
       r_carbon_emb = metrics["co2_emb_g"] / self.cfg.CARBON_BUDGET_G

       reward = r_slo - (self.cfg.ALPHA_OP * r_carbon_op) - (self.cfg.BETA_EMB * r_carbon_emb)
# Queueing stability penalty to prevent unbounded backlog
      if self.queue_len > 150:
      reward -= 0.5 * (self.queue_len / 150.0)
      info = {
      **metrics,
     "hour": self.t_hours[self.step_idx],
"qps": qps,
"ci": ci,
"config": self.surrogate.actions[real_action_idx],
"reward": reward,
}
self.step_idx += 1
terminated = self.step_idx >= self.cfg.STEPS_PER_DAY
truncated = False 

obs = self._get_obs() if not terminated else np.zeros(5, dtype=np.float32)
return obs, float(reward), terminated, truncated, info
