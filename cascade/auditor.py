import numpy as np
from scipy.stats import zscore
from ase import Atoms

class BaseAuditor: 

    def audit(self, atoms: list[Atoms], n_audits: int) -> tuple[float, list[Atoms]]:
        raise NotImplementedError()
    

class ForceThresholdAuditor(BaseAuditor): 
    """Determines the likelihood all calculations are below the threshold"""
    correction_factor : float
    threshold         : float

    def audit(self, 
              atoms: list[Atoms], 
              n_audits: int, 
              uq_key: str = 'force_std_max') -> tuple[float, list[Atoms]]:
        """"""
        # compute z-scores (force ens deviation * correction_factor) across whole trajectory
        # multiply them as if they are independent. this gives you the probability 
        traj_uq = np.asarray([a.info[uq_key] for a in atoms])
        z = zscore(traj_uq*self.correction_factor)

        pass

    # todo: think about this. Need a way to update the correction factor
    def receive_audit_result(self, expected_error, observed_error) -> None: 
        pass

class EnsembleDriftAuditor(BaseAuditor): 
    
    def audit(self, atoms: list[Atoms], n_audits: int) -> tuple[float, list[Atoms]]:
      pass