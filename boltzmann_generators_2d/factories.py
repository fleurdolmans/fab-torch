"""
Factory functions for the 2D Boltzmann generator.

This package is 2D-only: it models a solute in a 2D Lennard-Jones bath.

Functions
---------
build_system(**overrides)              → SoluteSimulation2D
build_flow(system, **overrides)        → SoluteSplineFlow or RealNVP
build_trainer(**overrides)             → BoltzmannGenerator2D
build_sampler(system, **kwargs)        → MetropolisSampler
"""

import boltzmann_generators_2d.config as cfg


# ===========================================================================
# System / Potential
# ===========================================================================

def build_system(system_type: str = None, **overrides):
    """
    Return a SoluteSimulation2D potential object.

    Parameters
    ----------
    system_type : str or None
        Accepted for API compatibility; must be '2d' or None.
    **overrides : keyword arguments forwarded to SoluteSimulation2D constructor.

    Returns
    -------
    SoluteSimulation2D
    """
    if system_type is not None and system_type != '2d':
        raise ValueError(
            f"boltzmann_generators_2d only supports system_type='2d', got '{system_type}'."
        )

    _c = cfg._CONFIGS['2d']
    params = dict(
        n_particles=_c['n_particles'],
        epsilon=_c['epsilon'],
        sigma=_c['sigma'],
        l_box=_c['l_box'],
        k_box=_c['k_box'],
        center_solute=_c['center_solute'],
        k_center=_c['k_center'],
    )
    params.update(overrides)

    from .Library.potentials import SoluteSimulation2D
    return SoluteSimulation2D(**params)


# ===========================================================================
# Flow
# ===========================================================================

def build_flow(system, system_type: str = None, **overrides):
    """
    Return an untrained normalising flow for the 2D system.

    Parameters
    ----------
    system : SoluteSimulation2D returned by build_system()
    system_type : str or None
        Accepted for API compatibility; must be '2d' or None.
    **overrides : architecture hyperparameters.
        flow_arch : 'spline_2d' (default) or 'realnvp'
        n_blocks, n_nodes, n_layers, num_bins, tail_bound

    Returns
    -------
    SoluteSplineFlow  (arch='spline_2d')
    RealNVP           (arch='realnvp')
    """
    if system_type is not None and system_type != '2d':
        raise ValueError(
            f"boltzmann_generators_2d only supports system_type='2d', got '{system_type}'."
        )

    _c = cfg._CONFIGS['2d']
    arch = overrides.pop('flow_arch', _c['flow_arch'])

    if arch in ('spline_2d', 'spline'):
        from .Library.flow.spline import build_solute_spline_flow

        # n_layers is the config-facing name; n_hidden is the constructor name
        if 'n_layers' in overrides:
            overrides['n_hidden'] = overrides.pop('n_layers')

        # prior_sigma is not supported by build_solute_spline_flow in this Library
        overrides.pop('prior_sigma', None)

        params = dict(
            n_particles=_c['n_solvent'],
            n_blocks=_c['n_blocks'],
            n_nodes=_c['n_nodes'],
            n_hidden=_c['n_layers'],
            num_bins=_c['num_bins'],
            tail_bound=_c['tail_bound'],
        )
        params.update(overrides)
        return build_solute_spline_flow(system=system, **params)

    else:
        # RealNVP
        from .Library.boltzmann import BoltzmannGenerator2D
        trainer_params = {k: v for k, v in overrides.items()}
        trainer = BoltzmannGenerator2D(trainer_params if trainer_params else None)
        return trainer.build(system)


# ===========================================================================
# Trainer
# ===========================================================================

def build_trainer(system_type: str = None, **overrides):
    """
    Return a BoltzmannGenerator2D trainer.

    Parameters
    ----------
    system_type : str or None
        Accepted for API compatibility; must be '2d' or None.
    **overrides : override any default hyperparameters.

    Returns
    -------
    BoltzmannGenerator2D
    """
    if system_type is not None and system_type != '2d':
        raise ValueError(
            f"boltzmann_generators_2d only supports system_type='2d', got '{system_type}'."
        )

    from .Library.boltzmann import BoltzmannGenerator2D
    return BoltzmannGenerator2D(overrides if overrides else None)


# ===========================================================================
# Sampler
# ===========================================================================

def build_sampler(system, system_type: str = None, sigma_mc=0.1, temp=1.0, stride=10):
    """
    Return a configured MetropolisSampler for the 2D system.

    Parameters
    ----------
    system : SoluteSimulation2D returned by build_system()
    system_type : str or None
        Accepted for API compatibility; must be '2d' or None.
    sigma_mc : float   MC step size
    temp : float       reduced temperature
    stride : int       save every stride-th frame

    Returns
    -------
    MetropolisSampler
    """
    from .Library.sampling import MetropolisSampler

    if system_type is not None and system_type != '2d':
        raise ValueError(
            f"boltzmann_generators_2d only supports system_type='2d', got '{system_type}'."
        )

    return MetropolisSampler(
        model=system,
        temp=temp,
        sigma=sigma_mc,
        stride=stride,
    )
