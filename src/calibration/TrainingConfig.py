from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Centralised hyperparameter container shared by all MFG training experiments.

    Every field maps 1-to-1 to a module-level constant that previously lived
    in each experiment script.  Experiment-specific defaults are set in the
    subclass or at construction time.
    """

    num_seeds: int = 5  # Number of independent random seeds for statistical averaging.

    epochs: int = 500  # Total training epochs.
    batch_size: int = 10  # Batch size per epoch.
    delta_ratio: float = 0.2  # Fraction of the total horizon used for each training sub-interval.
    learning_rate: float = 5e-3  # Calibration network learning rate.
    test_interval: int = 20  # Evaluate on the test set every this many epochs.

    is_mfc: bool = False  # Toggle Mean Field Control (True) vs. Mean Field Game (False).
    num_samples: int = 200  # Number of synthetic training trajectories (only applies to synthetic datasets)
    num_test_samples: int = 20  # Number of synthetic test trajectories

    T: float = 2.0  # Total time horizon.
    N: int = 100  # Number of time-discretisation steps

    constant_gamma: bool = False  # Force the parameter network to output a scalar constant.
    gamma_size: int = 1  # Number of scalar parameters the network must estimate.
    gamma_loss_idx: int = None  # The max index of gamma values used to calculate the gamma loss.
    picard_damping: float = 0.0  # Damping factor for Picard iteration (0 = no damping).

    root_dir: str = "results"  # Output directory
    experiment_name: str = "base_experiment"  # Experiment name

    # Derived quantities (computed post-init)
    @property
    def dt(self) -> float:
        return self.T / (self.N - 1)

    @property
    def delta_steps(self) -> int:
        return int(self.N * self.delta_ratio)


@dataclass
class DiffEqConfig(TrainingConfig):
    """
    Centralised hyperparameter container shared by the neural ODE based MFG training experiments.

    Inherits from the TrainingConfig

    Every field maps 1-to-1 to a module-level constant that previously lived
    in each experiment script.  Experiment-specific defaults are set in the
    subclass or at construction time.
    """
    in_size: int = 3
    out_size: int = 3
    predict_g: bool = False  # Toggle universal differential equation (Rackauckas et al., 2021).
    width_size: int = 64
    depth: int = 3           # Number of hidden layers including the output layer
