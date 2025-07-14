import numpy as np
from scipy.linalg import eigh

def computeEigenModes(IF, pupil, sort=False, includeTT=True, includeModes=0):
    # Computes the eigen mode basis of the DM of a DM given a set of DM modes (1 mode == 1 command vector)
    # The eigen modes are free from piston, tip and tilt. Tip and tilt are added at the end of the process.
    #
    # Input:
    #   - DM IFs functions, matrix of shape [Nact x Npix]. 
    #          This can be the complete grid of actuators or a preliminary filtered one (e.g. with slave actuators, etc).
    #   - Pupil: Illuminated pupil shape
    #   - sort: inverse the order of modes or not  
    #           Important as TT replace modes with id 0 and 1 afterwards!
    #   - includeTT: make a KL basis orthogonal to Tip-Tilt modes
    #   - includeModes: user defined modes. The KL basis will then be orthogonal to them.
    #             In this case, includesTT si set to False.
    #
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

    
    # Define piston and tip-tilt modes (Tp matrix)    
    # Modes to remove from the KL basis -- Always remove piston!
    piston_mode = np.ones(N_phi)*pupil
    Tp          = np.vstack([piston_mode])

    if np.abs(includeModes).sum()>0:
        includeTT = False        
        Tp = np.vstack([Tp, includeModes])                

        
    if includeTT:
        tip_mode_y  = np.linspace(-1, 1, N_phi)*pupil  # Placeholder for actual tip mode
        tip_mode_y  = tip_mode_y/np.std(tip_mode_y[pupil>0])

        tip_mode_x  = tip_mode_y.reshape(npupil, npupil).T.flatten()

        Tp = np.vstack([Tp, tip_mode_x, tip_mode_y])                
    Tp = Tp.T
    
    Nmode_removed  = Tp.shape[1]
    Nmode_included = Nmode_removed - 1
    print(Nmode_included)

    # Project included modes to IFs, and remove them
    IF_ = IF.copy()
    for i in range(3): # Iterate a couple of times to clean up...
        for i in range(Tp.shape[1]):
            mode = Tp[:,i]
            coef = mode@IF_/(mode.T@mode) # Amplitude of IFs projected over the modes to remove
            IF_ = IF_ - coef[np.newaxis,:]*mode[:,np.newaxis] # Remove the projected part from every IF
    

    # Compute the M2C from the IF_ covariance matrix
    C = IF_.T@IF_
    u,s,v=np.linalg.svd(C)  # NB: u == v.T
    M2C = u

    # Roll the final matrix to insert back the desired modes (except piston!)
    M2C = np.roll(M2C,Nmode_included, axis=1)

    IFi = np.linalg.pinv(IF)
    for i in range(Nmode_included):
        mode = Tp[:,i+1] # Shift by 1 to exclude piston!
        M2C[:,i] = IFi@mode/(mode.T@mode)
    M2C = M2C[:,:-1] # Remove 1 mode since removing piston also remove a degree of freedom

        
    # Compute the phase vectors over the DM
    klPhiVec                     = IF_@M2C
    klPhiVec[:,0:Nmode_included] = IF@M2C[:,0:Nmode_included]

    # Renormalize modes to 1
    norm     = (np.ptp(klPhiVec[pupil>0,:],axis=0)+1e-15)
    M2C      = M2C      / norm[np.newaxis,:]
    klPhiVec = klPhiVec / norm[np.newaxis,:]

    if sort:
        M2C = M2C[:,::-1]
        klPhiVec = klPhiVec[:,::-1]

    return M2C, klPhiVec

def computeEigenModes_old(IF, pupil, disp=False, sort=True, includeTT=True, includeModes=0):
    # Computes the eigen mode basis of the DM of a DM given a set of DM modes (1 mode == 1 command vector)
    # The eigen modes are free from piston, tip and tilt. Tip and tilt are added at the end of the process.
    #
    # Input:
    #   - DM IFs functions, matrix of shape [Nact x Npix]. 
    #          This can be the complete grid of actuators or a preliminary filtered one (e.g. with slave actuators, etc).
    #   - Pupil: Illuminated pupil shape
    #   - sort: inverse the order of modes or not  
    #           Important as TT replace modes with id 0 and 1 afterwards!
    #   - includeTT: make a KL basis orthogonal to Tip-Tilt modes
    #   - includeModes: user defined modes. The KL basis will then be orthogonal to them.
    #             In this case, includesTT si set to False.
    #
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
    # Modes to remove from the KL basis -- Always remove piston!
    piston_mode = np.ones(N_phi)*pupil
    Tp          = np.vstack([piston_mode])

    if np.abs(includeModes).sum()>0:
        includeTT = False        
        Tp = np.vstack([Tp, includeModes])                

        
    if includeTT:
        tip_mode_y  = np.linspace(-1, 1, N_phi)*pupil  # Placeholder for actual tip mode
        tip_mode_y  = tip_mode_y/np.std(tip_mode_y[pupil>0])

        tip_mode_x  = tip_mode_y.reshape(npupil, npupil).T.flatten()

        Tp = np.vstack([Tp, tip_mode_x, tip_mode_y])                
    Tp = Tp.T
    
    Nmode_removed  = Tp.shape[1]
    Nmode_included = Nmode_removed - 1
        

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
    TT_C = IF.T@Tp # Not technically right, but much faster than pinv()...

    if Nmode_removed>0:
        for i in range(Nmode_included):
            mu = np.std(TT_C[:,i+1])
            TT_C[:,i] /= mu/np.nanstd(KLphiVec[pupil>0,3])
        B = np.roll(B, (Nmode_included,0))
        B[:,range(Nmode_included)] = TT_C[:,range(1, 1+Nmode_included)]
    
    # Cleaning last mode
    B = np.delete(B, -1, 1)
    #print(IF.shape,B.shape)
    
    KLphiVec = IF@B

    return B, KLphiVec

def computeEigenModes_notsquarepupil(IF, pupil, pupil_grid_pix, disp=False, sort=True, includeTT=True, includeModes=0):
    # Computes the eigen mode basis of the DM of a DM given a set of DM modes (1 mode == 1 command vector)
    # The eigen modes are free from piston, tip and tilt. Tip and tilt are added at the end of the process.
    #
    # Input:
    #   - DM IFs functions, matrix of shape [Nact x Npix]. 
    #          This can be the complete grid of actuators or a preliminary filtered one (e.g. with slave actuators, etc).
    #   - Pupil: Illuminated pupil shape
    #   - sort: inverse the order of modes or not  
    #           Important as TT replace modes with id 0 and 1 afterwards!
    #   - includeTT: make a KL basis orthogonal to Tip-Tilt modes
    #   - includeModes: user defined modes. The KL basis will then be orthogonal to them.
    #             In this case, includesTT si set to False.
    #
    #
    # Output:
    #   - M2C matrix: KL in command space
    #   - KLphiVec:   KL in DM space
    #
    
    pupil_grid_x,pupil_grid_y = pupil_grid_pix
    pupil = pupil.flatten()
            
    # Dimensions
    #print(IF.shape)
    N_phi   = IF.shape[0]
    N_actus = IF.shape[1]

    # Geometric covariance matrix
    Delta = IF.T @ IF
    #print(Delta.shape)

    # Define piston and tip-tilt modes (Tp matrix)    
    # Modes to remove from the KL basis -- Always remove piston!
    piston_mode = np.ones(N_phi)*pupil
    Tp          = np.vstack([piston_mode])

    if np.abs(includeModes).sum()>0:
        includeTT = False        
        Tp = np.vstack([Tp, includeModes])                

        
    if includeTT:
        tip_mode_y  = np.linspace(-1, 1, N_phi)*pupil  # Placeholder for actual tip mode
        tip_mode_y  = tip_mode_y/np.std(tip_mode_y[pupil>0])

        tip_mode_x  = tip_mode_y.reshape(pupil_grid_x, pupil_grid_y).T.flatten()

        Tp = np.vstack([Tp, tip_mode_x, tip_mode_y])                
    Tp = Tp.T
    
    Nmode_removed  = Tp.shape[1]
    Nmode_included = Nmode_removed - 1
        

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
    TT_C = IF.T@Tp # Not technically right, but much faster than pinv()...

    if Nmode_removed>0:
        for i in range(Nmode_included):
            mu = np.std(TT_C[:,i+1])
            TT_C[:,i] /= mu/np.nanstd(KLphiVec[pupil>0,3])
        B = np.roll(B, (Nmode_included,0))
        B[:,range(Nmode_included)] = TT_C[:,range(1, 1+Nmode_included)]
    
    # Cleaning last mode
    B = np.delete(B, -1, 1)
    #print(IF.shape,B.shape)
    
    KLphiVec = IF@B

    return B, KLphiVec