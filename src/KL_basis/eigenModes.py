import numpy as np
from scipy.linalg import eigh


def computeEigenModes(IF, pupil, disp=False, sort=True):
    # Computes the eigen mode basis of the DM of a DM given a set of DM modes (1 mode == 1 command vector)
    # The eigen modes are free from piston, tip and tilt. Tip and tilt are added at the end of the process.
    #
    # Input:
    #   - DM IFs functions, matrix of shape [Nact x Npix]. 
    #          This can be the complete grid of actuators or a preliminary filtered one (e.g. with slave actuators, etc).
    #   - Pupil: Illuminated pupil shape
    #   - sort: inverse the order of modes or not  
    #           Important as TT replace modes with id 0 and 1 afterwards!
    #
    # Output:
    #   - M2C matrix: KL in command space
    #   - KLphiVec:   KL in DM space
    #
    pupil = pupil.flatten()
            
    # Dimensions
    print(IF.shape)
    N_phi   = IF.shape[0]
    N_actus = IF.shape[1]

    npupil = int(np.sqrt(len(pupil)))

    # Geometric covariance matrix
    Delta = IF.T @ IF
    #print(Delta.shape)

    # Define piston and tip-tilt modes (Tp matrix)
    piston_mode = np.ones(N_phi)*pupil
    tip_mode_y  = np.linspace(-1, 1, N_phi)*pupil  # Placeholder for actual tip mode
    tip_mode_y  = tip_mode_y/np.std(tip_mode_y[pupil>0])

    tip_mode_x  = tip_mode_y.reshape(npupil, npupil).T.flatten()

    # Modes to remove from the KL basis
    Tp = np.vstack([piston_mode, tip_mode_x, tip_mode_y]).T
    Nmode_removed = Tp.shape[1]

    # Projection matrix
    epsilon = 1e-13  # Small regularization term
    Delta_reg = Delta + epsilon * np.eye(Delta.shape[0])
    Delta_inv = np.linalg.inv(Delta_reg)
    tau = Delta_inv @ IF.T @ Tp
    #print(tau.shape)

    # Generate G matrix
    G = np.eye(N_actus) - tau @ np.linalg.inv(tau.T @ Delta @ tau) @ tau.T @ Delta

    # Diagonalize G to obtain B0
    s, B0 = eigh(G.T @ Delta @ G)
    #print(B0.shape)
    #print(IF.shape, B0.shape, s.shape)

    # normalize
    Lambda_inv_sqrt = np.diag(1 / np.sqrt(np.diag(B0.T @ Delta @ B0)))
    #Lambda_inv_sqrt = np.diag(1/s**0.5)
    #print(Lambda_inv_sqrt.shape)
    B = G @ B0 @ Lambda_inv_sqrt
    
    if sort: # To make sure low orders are first
        B = B[:,::-1]
    #print(IF.shape,B.shape)

    KLphiVec = IF@B
    #print(KLphiVec.shape, B.shape, IF.shape)

    # Add Tip & Tilt
    u,s,v = np.linalg.svd(IF, full_matrices=False)

    IFi = v.T@np.diag(s)@u.T
    # NB -- pinv not working for some reason...
    #print(IFi.shape, Tp.shape)
    #TT_C = IFi@Tp
    TT_C = IF.T@Tp # Not technically right than pinv(), but much faster...

    for i in [1,2]:
        mu = np.std(TT_C[:,i])
        TT_C[:,i] /= mu/np.nanstd(KLphiVec[pupil>0,3])
    B = np.roll(B, (2,0))
    B[:,0:2] = TT_C[:,1:3]
    
    # Cleaning last mode
    B = np.delete(B, -1, 1)
    #print(IF.shape,B.shape)
    
    KLphiVec = IF@B

    return B, KLphiVec