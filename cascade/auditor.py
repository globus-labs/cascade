import numpy as np
from scipy.stats import multivariate_normal
from ase import Atoms

class BaseAuditor: 

    def audit(self, atoms: list[Atoms], n_audits: int) -> tuple[float, list[Atoms]]:
        raise NotImplementedError()
    

class ForceThresholdAuditor(BaseAuditor): 
    """Determines the likelihood all calculations are below the threshold"""


    def __init__(self,
                 correction_factor : float = 1,
                 threshold: float = 1):
        
        self.correction_factor = correction_factor
        self.threshold = threshold

    def audit(self, 
              atoms: list[Atoms], 
              n_audits: int,
              n_sample: int = 100, 
              sort_audits: bool = False) -> tuple[float, list[int]]:
        """Estimate the probability that any atom is off more than threshold and the frames with the higest UQ

        Args:
            atoms: list of ase atoms
            n_audits: number of frames to return 
            n_sample: number of samples to take when estimating the error probability
            sort_audits: whether to return frames in decreasing UQ order. If false, uses argpartition which is linear time
        Returns: 
            p_any: an estimate of the probability that the forces on any atom in any frame is above the threshold
            audit_frames: a indices of the frames with the highest ensemble std, aggregated by max
        """

        force_preds = np.asarray([a.info['forces_ens'] for a in atoms])

        # flatten the predictions we have one dim for the ensemble and one for the rest
        ## last dim is spatial (3)
        n_frames, n_models, n_atoms, _ = force_preds.shape
        force_preds_flat = force_preds.reshape((n_models, n_frames * n_atoms * 3))
        
        # build the error distribution 
        force_cov = np.cov(force_preds_flat.T)
        force_err_dist = multivariate_normal(cov=force_cov, allow_singular=True)

        # take a sample from the error distribution
        force_var_samples_flat = force_err_dist.rvs(n_sample)
        force_var_samples = force_var_samples_flat.reshape((n_sample, n_atoms*n_frames, 3))
        
        # find the magnitude 
        force_samples_mag = np.linalg.norm(force_var_samples, axis=-1)

        # the probability that any 1 magnitude exceeds the threshold
        p_any = (force_samples_mag > self.threshold).any(axis=1).mean()


        # get the frames with the highest UQ
        ## take std over ens dimension and then max over remaining atom, spatial dims
        max_uq_by_frame = force_preds.std(1).max((1,2))

        # get the worst frames
        if sort_audits:
            top_uq_ix = np.argsort(max_uq_by_frame)[::-1][:n_audits]
        else:
            top_uq_ix = np.argpartition(max_uq_by_frame, -n_audits)[-n_audits:]
        audits = [atoms[i] for i in top_uq_ix]
        return p_any, top_uq_ix

    # todo: think about this. Need a way to update the correction factor
    def receive_audit_result(self, expected_error, observed_error) -> None: 
        pass

class EnsembleDriftAuditor(BaseAuditor): 
    
    def audit(self, atoms: list[Atoms], n_audits: int) -> tuple[float, list[Atoms]]:
      pass