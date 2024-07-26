"""Tools for building a new TorchANI architecture"""
from torchani import AEVComputer, ANIModel
import torch


def make_aev_computer(
        species: list[str],
        radial_cutoff: float = 5.2,
        angular_cutoff: float = 3.5,
        radial_eta: float = 16.,
        angular_eta: float = 8.,
        zeta: float = 32.,
        num_radial_terms: int = 16,
        num_angular_dist_terms: int = 4,
        num_angular_angl_terms: int = 8,
        angular_start: float = 0.9,
        radial_start: float = 0.9
) -> AEVComputer:
    """Make the environment computer

    Defaults are the parameters used in ANI-1x.
    Creates an evenly-spaced set of radial and angular terms using :meth:`~torchani.aev.AEVComputer.cover_linearly`.

    Args:
        species: List of species used for the model
        radial_cutoff: Maximum distance of radial terms in atomic environment (Units: A)
        angular_cutoff: Maximum distance of angular terms in atomic environment (Units: A)
        radial_eta: Inverse width of angular terms (Units: A)
        angular_eta: Inverse spatial width of angular terms (Units: A)
        zeta: Inverse angular width of angular terms
        num_radial_terms: Number of radial terms
        num_angular_dist_terms: Number of radial divisions in angular terms
        num_angular_angl_terms: Number of angular divisions in angular terms
        angular_start: Minimum distance for angular terms
        radial_start: Minimum distance for radial terms
    Returns:
        Tool which computes the atomic environment of each atom
    """
    return AEVComputer.cover_linearly(
        radial_cutoff=radial_cutoff,
        angular_cutoff=angular_cutoff,
        radial_eta=radial_eta,
        angular_eta=angular_eta,
        zeta=zeta,
        radial_dist_divisions=num_radial_terms,
        angular_dist_divisions=num_angular_dist_terms,
        angle_sections=num_angular_angl_terms,
        angular_start=angular_start,
        radial_start=radial_start,
        num_species=len(species)
    )


def make_output_nets(species: list[str],
                     aev_computer: AEVComputer,
                     hidden_units: int = 128,
                     hidden_layers: int = 2,
                     hidden_decay: float = 0.8) -> ANIModel:
    """Make the component which maps AEV components to the energy

    The output component of the ANI model are a series of multi-layer perceptions (MLPs)
    with :class:`~torch.nn.CELU` activation layers and gradually-decreasing numbers of hidden units.

    Args:
        species: List of species known to the model
        aev_computer: Tool which computes the atomic environments
        hidden_units: Number of hidden units in the first layer of the MLP
        hidden_layers: Number of hidden layers in the MLP
        hidden_decay: Factor the number of hidden units decreases by each layer
    Returns:
        Dense layer portion of the ANI model
    """

    aev_length = aev_computer.aev_length

    output_nets = []
    for specie in species:
        # First layer is mandatory
        output_net = []
        layer_size = hidden_units
        first_layer = torch.nn.Linear(aev_length, layer_size)
        output_net.append(first_layer)

        for i in range(hidden_layers):
            next_layer_size = int(layer_size * hidden_decay)
            output_net.append(torch.nn.CELU())
            output_net.append(torch.nn.Linear(layer_size, next_layer_size))
            layer_size = next_layer_size

        # Last layer is mandatory
        output_net.append(torch.nn.CELU())
        output_net.append(torch.nn.Linear(layer_size, 1))
        output_nets.append(torch.nn.Sequential(*output_net))

    return ANIModel(output_nets)
