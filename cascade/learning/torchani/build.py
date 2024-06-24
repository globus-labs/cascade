"""Tools for building a new TorchANI architecture"""

from torchani import AEVComputer, ANIModel
import torch


def make_aev_computer(species: list[str]) -> AEVComputer:
    """Make the environment computer

    Args:
        species: List of species used for the model
    Returns:
        Tool which computes the atomic environement of each atom
    """

    # TODO (wardlt): Make these options adjustable
    Rcr = 5.2000e+00
    Rca = 3.5000e+00
    EtaR = torch.tensor([1.6000000e+01])
    ShfR = torch.tensor(
        [9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00,
         3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00])
    Zeta = torch.tensor([3.2000000e+01])
    ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00])
    EtaA = torch.tensor([8.0000000e+00])
    ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00])
    return AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, len(species))


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
