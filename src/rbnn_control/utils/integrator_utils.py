# Name(s): Justice Mason, Arthur Yang
# Date(s): 02/27/2024

import os, sys
import numpy as np
import torch

class LieGroupVaritationalIntegratorGeneral():
    """
    Lie group variational integrator with gavity.

    ...

    """
    def __init__(self):
        super().__init__()
        
    def skew(self, v: torch.Tensor):
        
        S = torch.zeros([v.shape[0], 3, 3], device=v.device)
        S[:, 0, 1] = -v[..., 2]
        S[:, 1, 0] = v[..., 2]
        S[:, 0, 2] = v[..., 1]
        S[:, 2, 0] = -v[..., 1]
        S[:, 1, 2] = -v[..., 0]
        S[:, 2, 1] = v[..., 0]
    
        return S
    
    def calc_M(self, R: torch.Tensor, V) -> torch.Tensor:
        """
        Calculate moments.

        ...

        Parameters
        ----------
        R:: torch.Tensor
            input rotation matrix

        V:: torch.nn.Module
            gravitational potential function -- most likely a neural network

        Returns
        -------
        M::torch.Tensor
            gravitational moment

        """
        # Calculate gravitational potential value
        bs, _, _ = R.shape
        q = R.reshape(bs, 9)
        V_q = V(q)

        # Calculate gradient of potential 
        dV =  torch.autograd.grad(V_q.sum(), q, create_graph=True)[0]
        dV = dV.reshape(bs, 3, 3)

        # Calculate skew(M) and extract M
        SM = torch.bmm(torch.transpose(dV, -2, -1), R) - torch.bmm(torch.transpose(R, -2, -1), dV)
        M = torch.stack((SM[..., 2, 1], SM[..., 0, 2], SM[..., 1, 0]), dim=-1).float()
        return M

    def cayley_transx(self, fc: torch.Tensor):
        """
        Calculate the Cayley transform.

        ...

        Parameter
        ---------
        fc:: torch.Tensor
            fc value

        Return
        ------
        F:: torch.Tensor
            F value

        """
       
        F = torch.einsum('bij, bjk -> bik', (torch.eye(3, device=fc.device) + self.skew(fc)), torch.linalg.inv(torch.eye(3, device=fc.device) - self.skew(fc)))
        return F
    
    def calc_fc_init(self, a_vec: torch.Tensor, moi:torch.Tensor) -> torch.Tensor:
        """
        Calculate the initial value fc.

        ...

        Parameter
        ---------
        a_vec :: torch.Tensor

        moi :: torch.Tensor 
            Moment-of-inertia tensor -- shape (3, 3).

        Return
        ------
        fc_init :: torch.Tensor
            iInitial value for fc

        """
        
        fc_init = torch.einsum('bij, bj -> bi', torch.linalg.inv(2 * moi - self.skew(a_vec)), a_vec)
        return fc_init
    
    def calc_Ac(self, a_vec: torch.Tensor, moi: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        Calculate the initial value fc.

        ...

        Parameter
        ---------
        a_vec::torch.Tensor
            
        moi::torch.Tensor 
            moment-of-inertia tensor

        fc::torch.Tensor
            fc tensor

        Return
        ------
        Ac::torch.Tensor
            Value of Ac

        """
        
        Ac = a_vec + torch.einsum('bij, bj -> bi', self.skew(a_vec), fc) + torch.einsum('bj, b -> bj', fc, torch.einsum('bj, bj -> b', a_vec, fc)) - (2 * torch.einsum('ij, bj -> bi', moi, fc))
        return Ac
        
    def calc_grad_Ac(self, a_vec: torch.Tensor, moi: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        Calculate the gradient of Ac.

        ...

        Parameters
        ----------
        a_vec :: torch.Tensor
            Vector -- shape (bs, 3)

        moi :: torch.Tensor
            Moment-of-inertia matrix -- shape (3, 3).

        fc :: torch.Tensor
            fc value -- shape (bs, 3).

        Returns
        -------
        grad_Ac :: torch.Tensor
            Gradient of Ac matrix -- shape (bs, 3, 3).

        """
        grad_Ac = self.skew(a_vec) + torch.einsum('b, bij -> bij', torch.einsum('bi, bi -> b', a_vec, fc), torch.unsqueeze(torch.eye(3, device=a_vec.device), 0).repeat(fc.shape[0], 1, 1)) + torch.einsum('bi, bj -> bij', fc, a_vec) - (2 * moi)
        return grad_Ac
    
    def optimize_fc(self, R_vec: torch.Tensor, pi_vec: torch.Tensor, moi: torch.Tensor, V = None, fc_list: list = [], timestep: float = 1e-3, max_iter: int = 5, tol: float = 1e-8) -> list:
        """
        Optimize the fc value.
        
        ...

        Parameters
        ----------
        R_vec :: torch.Tensor
            Orientation vector -- shape (bs, 3).
        
        pi_vec :: torch.Tensor
            Angular momentum vector -- shape (bs, 3).

        moi :: torch.Tensor
            Moment-of-inertia matrix -- shape (3, 3).

        V :: torch.nn.Module, default=None
            Potential energy function -- should be either a torch modulue or lambda function.

        fc_list :: list, default=[]
            List of fc values.

        timestep:: torch.Tensor, defualt=1e-3
            Timestep value used during integration.

        max_iter :: int, default=5
            Maximum number of iterations to use for the Newton Raphson method (NRM).

        tol :: float, default=1e-8
            Tolerance to exit the NRM loop.

        Returns
        -------
        fc_list::torch.Tensor

        """
        # Count then number of iterations
        it = 0

        # If there is no potential, no moment due to potential
        if V is None:
            M_vec = torch.zeros_like(pi_vec)
        else:
            M_vec = self.calc_M(R=R_vec, V=V)

        # Initialize fc
        if not fc_list:
            a_vec = timestep * (pi_vec + (0.5 * timestep) * M_vec)
            fc_list.append(self.calc_fc_init(a_vec=a_vec, moi=moi))
        
        eps = torch.ones(fc_list[-1].shape[0])
        
        # Optimization loop -- Newton Raphson method
        while  torch.any(eps > tol) and it < max_iter:
            
            fc_i = fc_list[-1]
            a_vec = timestep * (pi_vec + (0.5 * timestep) * M_vec)
            
            Ac = self.calc_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
            grad_Ac = self.calc_grad_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
           
            fc_ii = fc_i - torch.einsum('bij, bj -> bi', torch.linalg.inv(grad_Ac),  Ac)
            
            eps = torch.linalg.norm(fc_ii - fc_i, axis=-1)
            fc_list.append(fc_ii)
            it += 1
            
        return fc_list
    
    def step(self, R_i: torch.Tensor, pi_i: torch.Tensor, moi: torch.Tensor, u_i: torch.Tensor = None, u_ii: torch.Tensor = None, V = None, fc_list: list = [], timestep: float = 1e-3):
        """
        Calculate next step using the dynamics and kinematics.

        ...

        Parameters
        ----------
       pi_i :: torch.Tensor
            Initial condition for angular momentum vector -- shape (batch size, 3).

        R_i :: torch.Tensor
            Intial condition for orientation matrix  -- shape (batch size, 3, 3).

        moi :: torch.Tensor
            Moment-of-inertia tensor -- shape (3, 3).

        u_i :: torch.Tensor, default=None
            Control moment input for timestep i -- shape (batch size, 3).

        u_ii :: torch.Tensor, default=None
            Control moment input for timestep ii -- shape (batch size, 3).

        V :: torch.nn.Module, default=None
            Potential energy function -- should be either a torch modulue or lambda function.

        fc_list :: list, default=[]
            fc list

        timestep :: float, defualt=1e-3
            Timestep used for integration.

        Returns
        -------
        pi_ii :: torch.Tensor
            Angular momentum vector for next timestep -- shape (batch size, 3).

        R_ii :: torch.Tensor
            Orientation matrix for next timestep  -- shape (batch size, 3, 3).
        
        fc_list :: list

        """
        # Calculate list of optimal fc
        fc_list = self.optimize_fc(R_vec=R_i, pi_vec=pi_i, moi=moi, timestep=timestep, fc_list=fc_list, V=V)
        
        # Selected optimal fc
        fc_opt = fc_list[-1]

        # Update pose using kinematics
        F_i = self.cayley_transx(fc=fc_opt)
        R_ii = torch.einsum('bij, bjk -> bik', R_i, F_i)
        
        # Calculate moment due to potential function
        if V is None:
            M_i = torch.zeros_like(pi_i)
            M_ii = torch.zeros_like(pi_i)
        else:
            M_i = self.calc_M(R=R_i, V=V)
            M_ii = self.calc_M(R=R_ii, V=V)
        
        # Grab control moment
        if u_i is None:
            u_i = torch.zeros_like(pi_i)

        if u_ii is None:
            u_ii = torch.zeros_like(pi_i)
        
        # Update angular momentum state
        pi_ii = torch.einsum('bji, bj -> bi', F_i, pi_i) + torch.einsum('bji, bj -> bi', 0.5 * timestep * F_i, u_i + M_i) + (0.5 * timestep) * (M_ii + u_ii)
        
        return R_ii, pi_ii, fc_list
    
    def integrate(self, pi_init: torch.Tensor, R_init: torch.Tensor, moi: torch.Tensor, u_control: torch.Tensor = None, V = None, timestep: float = 1e-3, traj_len: int = 100):
        """
        Method to integrate a full trajectory.
        ...

        Parameters
        ----------
        pi_init :: torch.Tensor
            Initial condition for angular momentum vector -- shape (batch size, 3).

        R_init :: torch.Tensor
            Intial condition for orientation matrix  -- shape (batch size, 3, 3).

        moi :: torch.Tensor
            Moment-of-inertia tensor -- shape (3, 3).

        u_control :: torch.Tensor, default=None
            Control input tensor -- shape (batch size, trajectory length, 3).

        V :: torch.nn.Module, default=None
            Potential energy function -- should be either a torch modulue or lambda function.

        timestep :: float, defualt=1e-3
            Timestep used for integration.

        traj_len :: int, default=100
            Trajectory length of the full trajectory.
        
        Returns
        -------
        R_traj :: torch.Tensor
        pi_traj :: torch.Tensor

        """
        pi_list = [pi_init.float()]
        R_list = [R_init.float()]
        
        # Integrate full trajectory
        for it in range(1, traj_len):
            # Initialize inputs 
            fc_list = []
            R_i = R_list[-1]
            pi_i = pi_list[-1]

            # Control moment
            if u_control is None:
                u_i = torch.zeros_like(pi_i)
                u_ii = torch.zeros_like(pi_ii)
            else:
                u_i = u_control[:, it-1, ...]
                u_ii = u_control[:, it, ...]

            # Calculate next timestep
            R_ii, pi_ii, fc_list = self.step(R_i=R_i, pi_i=pi_i, moi=moi, u_i=u_i, u_ii=u_ii, V=V, fc_list=fc_list, timestep=timestep)
            
            # Append to state lists
            R_list.append(R_ii)
            pi_list.append(pi_ii)
        
        # Append full trajectory together
        R_traj = torch.stack(R_list, axis=1)
        pi_traj = torch.stack(pi_list, axis=1)
        return R_traj, pi_traj
    