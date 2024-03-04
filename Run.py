import numpy as np, torch, sys
import src.rbnn_control.utils.integrator_utils as integrator_utils
import src.rbnn_control.utils.image_utils as image_utils
from scipy.spatial.transform import Rotation as R

integrator = integrator_utils.LieGroupVaritationalIntegratorGeneral()

batch_size = 1
traj_length = 1000
pi_init = torch.zeros(batch_size, 3)
R_init = R.from_euler('zyx', [0, 0, 0], degrees=True)
R_init = torch.tensor(R_init.as_matrix(), dtype=torch.float32)
R_init = torch.reshape(R_init, (batch_size, R_init.shape[0], R_init.shape[1]))
print(R_init.size())
moi = torch.diag(torch.tensor([1, 2, 4], dtype=torch.float32))
u_control = torch.ones(batch_size, traj_length, 3)
print(R_init)
R_traj, pi_traj = integrator.integrate(pi_init=pi_init, R_init=R_init, moi=moi, u_control=u_control, V = None, timestep=1e-3, traj_len=traj_length)
print(R_traj.size())
print(pi_traj.size())
print(R_traj[0, 5, ...].cpu().detach().numpy().shape)

generator = image_utils.ImageGeneratorBatch(R=R_traj, object='Cube', savepath='image/', filepath='blender_models/3dpend.blend', size=256, ratio=1, quality=90)