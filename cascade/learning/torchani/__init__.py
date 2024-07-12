"""Interface and glue code for to models built using `TorchANI <https://github.com/aiqm/torchani>_"""
import ase
from ase import units
import numpy as np
import pandas as pd
from ase.calculators.calculator import Calculator
from torch.utils.data import DataLoader
from functools import partial

from torchani.nn import SpeciesEnergies, Sequential
from torchani.ase import Calculator as ANICalculator
from torchani import AEVComputer, ANIModel, EnergyShifter
from torchani.aev import SpeciesAEV
from torchani.data import collate_fn
from ignite.engine import Engine, Events
import torch

from cascade.learning.base import BaseLearnableForcefield, State
from cascade.learning.utils import estimate_atomic_energies

__all__ = ['TorchANI', 'ANIModelContents']

ANIModelContents = tuple[AEVComputer, ANIModel, dict[str, float]]
"""Contents of the serialized form of a model:

1. Compute for atomic environments
2. The model which maps environments to energies
3. Ordered dict of chemical symbol to atomic energies (all Py3 dicts are ordered)
"""

my_collate_dict = {
    'species': -1,
    'coordinates': 0.0,
    'forces': 0.0,
    'energies': 0.0,
    'cells': 0.0
}


def ase_to_ani(atoms: ase.Atoms, species: list[str]) -> dict[str, torch.Tensor]:
    """Make an ANI-format dictionary from an ASE Atoms object

    Args:
        atoms: Atoms object to be converted
        species: List of species used to determine index given chemical symbol
    Returns:
        Atoms object in a tensor format
    """

    # An entry _must_ have the species and coordinates
    output = {
        'species': torch.from_numpy(np.array([species.index(s) for s in atoms.symbols])),
        'coordinates': torch.from_numpy(atoms.positions).float(),
        'cells': torch.from_numpy(atoms.cell.array).float()
    }

    if atoms.calc is not None:
        if 'energy' in atoms.calc.results:
            output['energies'] = torch.from_numpy(np.atleast_1d(atoms.get_potential_energy())).float()
        if 'forces' in atoms.calc.results:
            output['forces'] = torch.from_numpy(atoms.get_forces()).float()
    return output


def make_data_loader(data: list[ase.Atoms],
                     species: list[str],
                     batch_size: int,
                     train: bool,
                     **kwargs) -> DataLoader:
    """Make a data loader based on a list of Atoms

    Args:
        data: Data to use for the loader
        species: Map of chemical species to index in network
        batch_size: Batch size to use for the loader
        train: Whether this is a loader for training data. If so, sets ``shuffle`` and ``drop_last`` to True.
        kwargs: Passed to the ``DataLoader`` constructor
    """
    # Append training settings if set to train
    if train:
        kwargs['shuffle'] = True
        kwargs['drop_last'] = True

    return DataLoader([ase_to_ani(a, species) for a in data],
                      collate_fn=lambda x: collate_fn(x, my_collate_dict),
                      batch_size=max(min(batch_size, len(data)), 1),
                      **kwargs)


def forward_batch(batch: dict[str, torch.Tensor],
                  aev_computer: AEVComputer,
                  nn: ANIModel,
                  atom_energies: np.ndarray,
                  pbc: torch.Tensor,
                  forces: bool = True,
                  device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the forward step on a batch of entries

    Args:
        batch: Batch from the data loader
        aev_computer: Atomic environment computer
        nn: Model which maps atomic environments to energies
        atom_energies: Array holding the reference energy for each species
        pbc: Periodic boundary conditions used by all members of the batch
        forces: Whether to compute forces
        device: Device on which to run computations
    Returns:
        - Energies for each member
        - Forces for each member
    """
    # Move the data to the device
    batch_z = batch['species'].to(device)
    batch_x = batch['coordinates'].float().to(device).requires_grad_(forces)
    batch_c = batch['cells'].float().to(device)

    # Compute the energy offset per member (run on the CPU because it's fast)
    batch_o = atom_energies[batch['species'].numpy()].sum(axis=1)
    batch_o = torch.from_numpy(batch_o).to(device)

    # Compute the AEVs individually because TorchANI assumes all entries have the same cell size
    batch_a = []
    for row_z, row_x, row_c in zip(batch_z, batch_x, batch_c):
        row_z = torch.unsqueeze(row_z, 0)
        row_x = torch.unsqueeze(row_x, 0)
        batch_a.append(aev_computer((row_z, row_x), row_c, pbc).aevs)
    batch_a = torch.concat(batch_a)
    batch_a = SpeciesAEV(batch_z, batch_a)

    # Get the energies for each member
    _, batch_e_pred = nn(batch_a)
    batch_e_pred = batch_e_pred + batch_o
    if forces:
        batch_f_pred = -torch.autograd.grad(batch_e_pred.sum(), batch_x, create_graph=True)[0]
        return batch_e_pred, batch_f_pred
    else:
        return batch_e_pred, None


def adjust_energy_scale(aev_computer: AEVComputer,
                        model: ANIModel,
                        loader: DataLoader,
                        atom_energies: np.ndarray,
                        device: str | torch.device = 'cpu') -> tuple[float, float]:
    """Adjust the last layer of an ANIModel such that its standard deviation matches that of the training data

    Args:
        aev_computer: Tool which computes atomic environments
        model: Model to be adjusted
        loader: Data loader
        atom_energies: Reference energy for each specie
        device: Device on which to perform inference
    Returns:
        Scale and shift factors
    """

    # Iterate over the dataset to get the actual and observed standard deviation of atomic energies
    pbc = torch.from_numpy(np.ones((3,), bool)).to(device)  # TODO (don't hard code to 3D)
    true_energies = []
    pred_energies = []
    for batch in loader:
        # Get the actual energies and predicted energies for the system
        batch_e_pred, _ = forward_batch(batch, aev_computer, model, atom_energies, pbc, forces=False, device=device)
        batch_e = batch['energies'][:, 0].cpu().numpy()

        # Get the energy per atom w/o the reference energy
        batch_o = atom_energies[batch['species'].numpy()].sum(axis=1)
        batch_n = (batch['species'] >= 0).sum(dim=1, dtype=batch_e_pred.dtype).cpu().numpy()

        pred_energies.extend((batch_e_pred.detach().cpu().numpy() - batch_o) / batch_n)
        true_energies.extend((batch_e - batch_o) / batch_n)

    # Get the ratio in standard deviations
    true_std = np.std(true_energies)
    pred_std = np.std(pred_energies)
    factor = true_std / pred_std

    # Update the last layer of each network to match the new scaling
    with torch.no_grad():
        for m in model.values():
            last_linear = m[-1]
            assert isinstance(last_linear, torch.nn.Linear), f'Last layer is not linear. It is {type(last_linear)}'
            assert last_linear.out_features == 1, f'Expected last layer to have one output. Found {last_linear.out_features}'
            last_linear.weight *= factor

    # Recompute the energies given the new shift factor
    pred_energies = []
    for batch in loader:
        batch_e_pred, _ = forward_batch(batch, aev_computer, model, atom_energies, pbc, forces=False, device=device)
        batch_o = atom_energies[batch['species'].numpy()].sum(axis=1)
        batch_n = (batch['species'] >= 0).sum(dim=1, dtype=batch_e_pred.dtype).cpu().numpy()
        pred_energies.extend((batch_e_pred.detach().cpu().numpy() - batch_o) / batch_n)

    # Get the shift for the mean
    true_mean = np.mean(true_energies)
    pred_mean = np.mean(pred_energies)
    shift = pred_mean - true_mean

    # Update the last layer of each network to match the new scaling
    with torch.no_grad():
        for m in model.values():
            last_linear = m[-1]
            last_linear.bias -= shift

    return factor, shift


class TorchANI(BaseLearnableForcefield[ANIModelContents]):
    """Interface to the high-dimensional neural networks implemented by `TorchANI <https://github.com/aiqm/torchani>`_"""

    def evaluate(self,
                 model_msg: bytes | ANIModelContents,
                 atoms: list[ase.Atoms],
                 batch_size: int = 64,
                 device: str = 'cpu') -> tuple[np.ndarray, list[np.ndarray]]:

        # TODO (wardlt): Put model in "eval" mode, skip making the graph when computing gradients in `forward_batch` <- performance optimizations

        # Unpack the model
        if isinstance(model_msg, bytes):
            model_msg = self.get_model(model_msg)
        aev_computer, model, atomic_energies = model_msg
        model.to(device)
        aev_computer.to(device)

        # Unpack the reference energies as a float32 array
        species = list(atomic_energies.keys())
        ref_energies = np.array([atomic_energies[s] for s in species]).astype(np.float32)

        # Build the data loader
        loader = make_data_loader(atoms, species, batch_size, train=False)

        # Run inference on all data
        energies = []
        forces = []
        pbc = torch.from_numpy(np.ones((3,), bool)).to(device)  # TODO (don't hard code to 3D)
        for batch in loader:
            batch_e_pred, batch_f_pred = forward_batch(batch, aev_computer, model, ref_energies, pbc, device=device)
            energies.extend(batch_e_pred.detach().cpu().numpy())  # Energies are the same regardless of size of input

            # The shape of the force array differs depending on size
            batch_n = (batch['species'] >= 0).sum(dim=1).cpu().numpy()  # Number of real atoms per batch
            for entry_f, entry_n in zip(batch_f_pred.detach().cpu().numpy(), batch_n):
                forces.append(entry_f[:entry_n, :])

        # Move model back from device
        model.to('cpu')
        aev_computer.to('cpu')

        return np.array(energies), list(forces)

    def train(self,
              model_msg: bytes | State,
              train_data: list[ase.Atoms],
              valid_data: list[ase.Atoms],
              num_epochs: int,
              device: str = 'cpu',
              batch_size: int = 32,
              learning_rate: float = 1e-3,
              huber_deltas: tuple[float, float] = (0.5, 1),
              force_weight: float = 10,
              reset_weights: bool = False,
              scale_energies: bool = True,
              **kwargs) -> tuple[bytes, pd.DataFrame]:
        # Unpack the model and move to device
        if isinstance(model_msg, bytes):
            model_msg = self.get_model(model_msg)
        aev_computer, model, atomic_energies = model_msg
        species = list(atomic_energies.keys())
        model.to(device)
        aev_computer.to(device)

        # Re-fit the atomic energies
        atomic_energies.update(estimate_atomic_energies(train_data))
        ref_energies = np.array([atomic_energies[s] for s in species], dtype=np.float32)  # Don't forget to cast to f32!

        # Reset the weights, if desired
        def init_params(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=1.0)
                torch.nn.init.zeros_(m.bias)

        if reset_weights:
            model.apply(init_params)

        # Build the data loader
        pbc = torch.from_numpy(np.ones((3,), bool)).to(device)  # TODO (don't hard code to 3D)
        train_loader = make_data_loader(train_data, species, batch_size, train=True)
        valid_loader = make_data_loader(valid_data, species, batch_size, train=False)

        # Adjust output layers to match data distribution
        if scale_energies:
            adjust_energy_scale(aev_computer, model, train_loader, ref_energies, device)

        # Prepare optimizer and loss functions
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

        huber_e, huber_f = huber_deltas
        loss_e = torch.nn.HuberLoss(reduction='none', delta=huber_e)
        loss_f = torch.nn.HuberLoss(reduction='none', delta=huber_f)

        def train_step(engine, batch):
            """Borrowed from the training step used inside MACE"""
            model.train()
            opt.zero_grad()

            # Run the forward step
            batch_e_pred, batch_f_pred = forward_batch(batch, aev_computer, model, ref_energies, pbc, device=device)

            # Compute the losses
            # TODO (wardlt): Add stresses
            batch_e = batch['energies'].to(device)[:, 0]
            batch_f = batch['forces'].to(device)
            batch_n = (batch['species'] >= 0).sum(dim=1, dtype=batch_e.dtype).to(device)

            energy_loss = (loss_e(batch_e_pred, batch_e) / batch_n.sqrt()).mean()
            force_loss = (loss_f(batch_f_pred, batch_f).sum(dim=(1, 2)) / batch_n).mean()
            loss = energy_loss + force_weight * force_loss
            loss.backward()
            opt.step()
            return loss.item()

        def evaluate_model(loader: DataLoader,
                           accumulator: list[dict[str, float]]) -> None:
            """Evaluate the model against all data in loader, store in global list

            Args:
                loader: a pytorch data loader
                accumulator: a list in which to append results
            """
            e_qm, e_ml, f_qm, f_ml = [], [], [], []
            for batch in loader:
                # Get the true results
                batch_e = batch['energies'].cpu().numpy()[:, 0]  # Make it a 1D array
                batch_f = batch['forces'].cpu().numpy()

                batch_e_pred, batch_f_pred = forward_batch(batch, aev_computer, model, ref_energies, pbc, device=device)
                batch_e_pred = batch_e_pred.detach().cpu().numpy()

                e_qm.extend(batch_e)
                e_ml.extend(batch_e_pred)

                # Compute the forces
                f_qm.extend(batch_f.ravel())
                f_ml.extend(batch_f_pred.detach().cpu().numpy().ravel())

            # convert batch results into flat np arrays
            e_qm, e_ml, f_qm, f_ml = map(lambda a: np.asarray(a), (e_qm, e_ml, f_qm, f_ml))

            # compute and store metrics
            result = {}
            result['e_rmse'] = np.sqrt((e_qm - e_ml) ** 2).mean()
            result['e_mae'] = np.abs(e_qm - e_ml).mean()
            result['f_rmse'] = np.sqrt((f_qm - f_ml) ** 2).mean()
            result['f_mae'] = np.abs(f_qm - f_ml).mean()
            accumulator.append(result)
            return

        # instantiate the trainer
        trainer = Engine(train_step)

        # Set up model evaluators
        # TODO (miketynes): make this more idiomatic pytorch ignite
        # TODO (miketynes): add early stopping (probably depends on the above)
        perf_train: list[dict[str, float]] = []
        perf_val: list[dict[str, float]] = []
        evaluator_train = partial(evaluate_model,
                                  loader=train_loader,
                                  accumulator=perf_train)
        evaluator_val = partial(evaluate_model,
                                loader=valid_loader,
                                accumulator=perf_val)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluator_train)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluator_val)

        # Run the training
        trainer.run(train_loader, max_epochs=num_epochs)

        # coalesce the training performance
        perf_train, perf_val = map(pd.DataFrame.from_records, [perf_train, perf_val])
        perf_train['split'] = 'train'
        perf_val['split'] = 'val'
        perf = pd.concat([perf_train, perf_val]).reset_index(names='iteration')

        # Ensure GPU memory is cleared
        model.to('cpu')
        aev_computer.to('cpu')

        # serialize model
        model_msg = self.serialize_model((aev_computer, model, atomic_energies))
        return model_msg, perf

    def make_calculator(self, model_msg: bytes | ANIModelContents, device: str) -> Calculator:
        # Unpack the model
        if isinstance(model_msg, bytes):
            model_msg = self.get_model(model_msg)
        aev_computer, model, atomic_energies = model_msg

        # Make an output layer which re-adds the atomic energies and other which converts to Ha (TorchNANI
        ref_energies = torch.tensor(list(atomic_energies.values()), dtype=torch.float32)
        shifter = EnergyShifter(ref_energies)

        class ToHartree(torch.nn.Module):
            def forward(self, species_energies: SpeciesEnergies,
                        cell: torch.Tensor | None = None,
                        pbc: torch.Tensor | None = None) -> SpeciesEnergies:
                species, energies = species_energies
                return SpeciesEnergies(species, energies / units.Hartree)

        to_hartree = ToHartree()
        post_model = Sequential(
            aev_computer,
            model,
            shifter,
            to_hartree
        )  # Use ANI's Sequential, which is fine with `cell` and `pbc` as inputs

        # Assemble the calculator
        species = list(atomic_energies.keys())
        post_model.to(device)
        return ANICalculator(species, post_model, overwrite=False)
