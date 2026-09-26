"""Server-side distribution aggregation and GUIDE packet delivery."""

import copy
import numpy as np

from GUIDE_Config import *
from GUIDE_Utils import (
    apply_weight_reset,
    get_full_covariance_jitter_events,
    iterative_merge_distributions,
    reset_full_covariance_jitter_events,
)


# Coordinates agent messages for GUIDE_Server.
class Server:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset_distributions = []
        self.round_telemetry = {}
        self._shared_packet_indices = None

    @staticmethod
    def _covariance_scalar_counts(covariance_mode):
        if covariance_mode == 'diag':
            return DIM, DIM
        if covariance_mode == 'full':
            return DIM * DIM, DIM * (DIM + 1) // 2
        raise ValueError("DPGMM_COVARIANCE_TYPE must be 'diag' or 'full'.")

    @staticmethod
    def _validate_covariance(distribution, covariance_mode):
        covariance = np.asarray(distribution['covariance'], dtype=float)
        expected = (DIM,) if covariance_mode == 'diag' else (DIM, DIM)
        if covariance.shape != expected:
            raise ValueError(
                f"Server expected {covariance_mode} covariance shape {expected}, "
                f"got {covariance.shape}."
            )

    def collect_distributions(self, agent_distributions_dict):
        """Collect, optionally merge, and calibrate uploaded components."""
        # A shared packet, when requested, is sampled afresh once per round.
        self._shared_packet_indices = None
        raw_distributions = []
        for agent_id, distributions in agent_distributions_dict.items():
            for distribution in distributions:
                copied = copy.deepcopy(distribution)
                self._validate_covariance(copied, DPGMM_COVARIANCE_TYPE)
                copied['source_agent_ids'] = sorted(
                    set(copied.get('source_agent_ids', [agent_id])),
                    key=lambda value: str(value),
                )
                raw_distributions.append(copied)

        covariance_scalars, theoretical_covariance_scalars = (
            self._covariance_scalar_counts(DPGMM_COVARIANCE_TYPE)
        )
        uplink_per_component = 1 + DIM + covariance_scalars + 1
        theoretical_uplink_per_component = (
            1 + DIM + theoretical_covariance_scalars + 1
        )
        self.round_telemetry = {
            'covariance_mode': DPGMM_COVARIANCE_TYPE,
            'raw_pool_size': len(raw_distributions),
            'merged_pool_size': 0,
            'uplink_components': len(raw_distributions),
            'uplink_scalars_actual': len(raw_distributions) * uplink_per_component,
            'uplink_scalars_theoretical': (
                len(raw_distributions) * theoretical_uplink_per_component
            ),
            'downlink_components_total': 0,
            'downlink_scalars_actual': 0,
            'downlink_scalars_theoretical': 0,
            'full_covariance_jitter_events': 0,
        }

        if not raw_distributions:
            self.reset_distributions = []
            return

        reset_full_covariance_jitter_events()
        if ENABLE_MERGING:
            merged = iterative_merge_distributions(
                raw_distributions,
                threshold=RMS_EUCLIDEAN_THRESHOLD,
                covariance_mode=DPGMM_COVARIANCE_TYPE,
            )
        else:
            merged = copy.deepcopy(raw_distributions)

        if ENABLE_WEIGHT_CALIBRATION:
            calibrated_weights = apply_weight_reset(
                [distribution['weight'] for distribution in merged],
                [distribution['strategy_value'] for distribution in merged],
            )
        else:
            mixture_masses = np.asarray(
                [distribution['weight'] for distribution in merged], dtype=float
            )
            mass_sum = float(np.sum(mixture_masses))
            if mass_sum <= 0.0:
                raise ValueError("Mixture masses must sum to a positive value.")
            calibrated_weights = (mixture_masses / mass_sum).tolist()

        for distribution, calibrated_weight in zip(merged, calibrated_weights):
            distribution['reset_weight'] = float(calibrated_weight)

        self.reset_distributions = merged
        self.round_telemetry['merged_pool_size'] = len(merged)
        self.round_telemetry['full_covariance_jitter_events'] = (
            get_full_covariance_jitter_events()
        )

    def visualize_top_weights(self):
        """Print up to five components ranked by calibrated global weight."""
        if not self.reset_distributions:
            return
        sorted_distributions = sorted(
            self.reset_distributions,
            key=lambda distribution: distribution['reset_weight'],
            reverse=True,
        )
        print("   [Server] Top five calibrated component weights:")
        for index, distribution in enumerate(sorted_distributions[:5]):
            source = distribution.get(
                'source_agent_ids', distribution.get('agent_id', 'merged')
            )
            print(
                f"      Rank {index + 1}: Weight = "
                f"{distribution['reset_weight']:.4f} | Source Agent = {source}"
            )

    def distribute_distributions_to_agent(self, target_agent_id):
        """Sample at most P calibrated components, or send the full global packet."""
        if not self.reset_distributions:
            return []
        if FULL_GLOBAL_PACKET:
            indices = np.arange(len(self.reset_distributions))
        elif not AGENT_SPECIFIC_SAMPLING and self._shared_packet_indices is not None:
            indices = self._shared_packet_indices
        else:
            sample_count = min(
                len(self.reset_distributions), SERVER_DISTRIBUTION_NUM
            )
            if sample_count == 0:
                return []
            weights = np.asarray(
                [distribution['reset_weight'] for distribution in self.reset_distributions],
                dtype=float,
            )
            probabilities = weights / np.sum(weights)
            probabilities[-1] = 1.0 - np.sum(probabilities[:-1])
            indices = np.random.choice(
                len(self.reset_distributions),
                size=sample_count,
                replace=False,
                p=probabilities,
            )
            if not AGENT_SPECIFIC_SAMPLING:
                self._shared_packet_indices = np.array(indices, copy=True)

        packet = [
            {
                'reset_weight': self.reset_distributions[index]['reset_weight'],
                'mean': copy.deepcopy(self.reset_distributions[index]['mean']),
                'covariance': copy.deepcopy(
                    self.reset_distributions[index]['covariance']
                ),
                'source_agent_ids': copy.deepcopy(
                    self.reset_distributions[index].get('source_agent_ids', [])
                ),
            }
            for index in indices
        ]
        covariance_scalars, theoretical_covariance_scalars = (
            self._covariance_scalar_counts(DPGMM_COVARIANCE_TYPE)
        )
        self.round_telemetry['downlink_components_total'] += len(packet)
        self.round_telemetry['downlink_scalars_actual'] += len(packet) * (
            1 + DIM + covariance_scalars
        )
        self.round_telemetry['downlink_scalars_theoretical'] += len(packet) * (
            1 + DIM + theoretical_covariance_scalars
        )
        return packet

    def get_distribution_info_for_tracking(self):
        return len(self.reset_distributions)

    def get_round_telemetry(self):
        return copy.deepcopy(self.round_telemetry)
