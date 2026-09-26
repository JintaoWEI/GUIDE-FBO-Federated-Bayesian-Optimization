import torch
import numpy as np
import pandas as pd
import random
import warnings
import time
from GUIDE_Config import *
from GUIDE_Utils import equal_CLHS, get_full_covariance_jitter_events
from GUIDE_Agent import Agent
from GUIDE_Server import Server

warnings.filterwarnings('ignore')

# Sensitivity/ablation runner reads this after each fresh experiment instance.
LAST_EXPERIMENT_TELEMETRY = []

# Run one seeded experimental repetition.
def run_single_experiment(exp_id):
    global LAST_EXPERIMENT_TELEMETRY
    print(f"\n=== Running Experiment {exp_id + 1}/{NUM_EXPERIMENTS} ===")
    start_time = time.time()
    
    seed = SEED_BASE + exp_id
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    # Pass the experiment ID to common Latin hypercube sampling.
    cubes = equal_CLHS(N_AGENTS, INITIAL_SAMPLES, exp_id)
    
    agents = []
    for i in range(N_AGENTS):
        # Pass the experiment ID to generate fixed heterogeneous agent settings.
        agent = Agent(i, cubes[i][0], cubes[i][1], exp_id)
        # Pass the experiment ID to fix initial points and observation noise.
        agent.generate_initial_data(exp_id)
        agents.append(agent)
        
    server = Server(N_AGENTS)
    
    avg_btv_hist = []
    avg_inst_hist = []
    round_telemetry = []
    
    btv_vals = [a.get_btv_value() for a in agents]
    inst_vals = [a.get_instantaneous_value() for a in agents]
    avg_btv_hist.append(np.mean(btv_vals))
    avg_inst_hist.append(np.mean(inst_vals))
    
    print(f"Iter 0 | Avg Best-so-far True Value: {np.mean(btv_vals):.4f}")
    
    for itr in range(1, MAX_ITERATIONS + 1):
        # --- Stage 1: agents extract and score promising local candidates. ---
        agent_dists_to_server = []
        for agent in agents:
            agent.fit_gmm_and_extract_distributions()
            if len(agent.current_distributions) > 0:
                # The selected component already contains its strategy score.
                agent_dists_to_server.append(agent.current_distributions[0])
                
        # --- Stage 2: the server aggregates the submitted components. ---
        # Wrap the components in the server's expected input format.
        cv_agent_dists = {d['agent_id']: [d] for d in agent_dists_to_server}
        server.collect_distributions(cv_agent_dists)
            
        # --- Optional diagnostic output ---
        if VISUALIZE_PROCESS:
            server.visualize_top_weights()
        
        # --- Stage 3: agents receive global guidance and optimize continuously. ---
        for agent in agents:
            recv_dists = server.distribute_distributions_to_agent(agent.agent_id)
            agent.receive_distributions_from_server(recv_dists)
            agent.select_next_observation_point()

        telemetry = server.get_round_telemetry()
        telemetry['full_covariance_jitter_events'] = (
            get_full_covariance_jitter_events()
        )
        telemetry['iteration'] = itr
        telemetry['direct_posterior_mean_difference'] = float(np.mean([
            agent.last_posterior_mean_difference for agent in agents
        ]))
        telemetry['direct_posterior_variance_difference'] = float(np.mean([
            agent.last_posterior_variance_difference for agent in agents
        ]))
        round_telemetry.append(telemetry)

        # Logging
        btv_vals = [a.get_btv_value() for a in agents]
        inst_vals = [a.get_instantaneous_value() for a in agents]
        avg_btv_hist.append(np.mean(btv_vals))
        avg_inst_hist.append(np.mean(inst_vals))
        
        print(f"Iter {itr} | Avg Best-so-far True Value: {np.mean(btv_vals):.4f} | Avg Instantaneous Value: {np.mean(inst_vals):.4f}")
            
    LAST_EXPERIMENT_TELEMETRY = round_telemetry
    return avg_btv_hist, avg_inst_hist, time.time() - start_time

def main():
    print(f"=== GUIDE-FBO {ACQ_FUNCTION} Experiment: {FUNCTION_NAME.upper()} ===")
    print(f"Config: Agents={N_AGENTS} Cov_Type={DPGMM_COVARIANCE_TYPE}")
    all_btv, all_inst, all_time = [], [], []
    
    for i in range(NUM_EXPERIMENTS):
        f, ins, t = run_single_experiment(i)
        all_btv.append(f); all_inst.append(ins); all_time.append(t)
        
    iters = list(range(MAX_ITERATIONS + 1))
    
    btv_df = pd.DataFrame({'iteration': iters})
    for i in range(NUM_EXPERIMENTS):
        btv_df[f'exp_{i+1}'] = all_btv[i]
    btv_df.to_csv(BTV_CSV, index=False)

    inst_df = pd.DataFrame({'iteration': iters})
    for i in range(NUM_EXPERIMENTS):
        inst_df[f'exp_{i+1}'] = all_inst[i]
    inst_df.to_csv(INSTANTANEOUS_CSV, index=False)
    
    print(f"\n=== Experiment Finished ===")
    print(f"Final Avg Best-so-far True Value: {np.mean([r[-1] for r in all_btv]):.4f}")
    
    print(f"\n=== Timing Stats ===")
    timing_lines = []
    for exp_id, elapsed_time in enumerate(all_time):
        timing_line = f"Exp {exp_id + 1} Time: {elapsed_time:.2f} s"
        print(timing_line)
        timing_lines.append(timing_line)
    
    avg_time = np.mean(all_time)
    total_time = np.sum(all_time)
    
    print("---")
    avg_line = f"Avg Time: {avg_time:.2f} s"
    total_line = f"Total Time: {total_time:.2f} s"
    print(avg_line)
    print(total_line)
    
    timing_lines.append("---")
    timing_lines.append(avg_line)
    timing_lines.append(total_line)
    
    with open(TIMING_TXT, 'w', encoding='utf-8') as f:
        f.write("=== Timing Stats ===\n")
        for line in timing_lines:
            f.write(line + "\n")
    
    print(f"\nTiming stats saved to: {TIMING_TXT}")

if __name__ == "__main__":
    main()
