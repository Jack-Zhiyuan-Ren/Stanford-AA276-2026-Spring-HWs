"""
Helper functions to query the learned values and gradients of vf.ckpt, cbf.ckpt.
NOTE: To use NeuralCBF defined below, you need to have your solution files from Homework 1
copied to this directory. Alternatively, write your script that uses NeuralCBF
in your ../hw1/ folder from Homework 1 (a copy of this file should already be there).
Make sure your hw1 venv is activated to use NeuralCBF.
NOTE: Due to package version incompatibilities between the two neural libraries,
you will probably need to use separate scripts and venvs.
"""

import torch

"""
Helper class for querying vf.ckpt.
neuralvf = NeuralVF()
values = neuralvf.values(x)
gradients = neuralvf.gradients(x)
"""
# class NeuralVF:
#     def __init__(self, ckpt_path='outputs/vf.ckpt'):
#         import os
#         import sys
#         sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#         sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

#         from libraries.DeepReach_MPC.utils import modules
#         from libraries.DeepReach_MPC.dynamics.dynamics import Quadrotor

#         dynamics = Quadrotor(collisionR=0.5, collective_thrust_max=20, set_mode='avoid')
#         model = modules.SingleBVPNet(in_features=dynamics.input_dim, out_features=1, type='sine', mode='mlp',
#                                     final_layer_factor=1., hidden_features=512, num_hidden_layers=3, 
#                                     periodic_transform_fn=dynamics.periodic_transform_fn)
#         model.cuda()
#         model.load_state_dict(torch.load(ckpt_path)['model'])

#         self.dynamics = dynamics
#         self.model = model

#     def values(self, x):
#         """
#         args:
#             x: torch tensor with shape      [batch_size, 13]
#         returns:
#             values: torch tensor with shape [batch_size]
#         """
#         coords = torch.concatenate((torch.ones((len(x), 1)), x), dim=1)
#         model_input = self.dynamics.coord_to_input(coords)
#         with torch.no_grad():
#             model_results = self.model({'coords': model_input.cuda()})
#         values = self.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].detach().squeeze(dim=-1))
#         return values.cpu()
    
#     def gradients(self, x):
#         """
#         args:
#             x: torch tensor with shape         [batch_size, 13]
#         returns:
#             gradients: torch tensor with shape [batch_size, 13]
#         """
#         coords = torch.concatenate((torch.ones((len(x), 1)), x), dim=1)
#         model_input = self.dynamics.coord_to_input(coords)
#         model_results = self.model({'coords': model_input.cuda()})
#         gradients = self.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))[:, 1:]
#         return gradients.cpu()

### For the twoCars8D system
class NeuralVF:
    def __init__(self, ckpt_path='outputs/vf.ckpt'):
        import os
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

        from libraries.DeepReach_MPC.utils import modules
        from libraries.DeepReach_MPC.dynamics.dynamics import TwoCars8D

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dynamics = TwoCars8D(
            collisionR=1,
            set_mode='avoid',
        )

        model = modules.SingleBVPNet(
            in_features=dynamics.input_dim,
            out_features=1,
            type='sine',
            mode='mlp',
            final_layer_factor=1.,
            hidden_features=512,
            num_hidden_layers=3,
            periodic_transform_fn=dynamics.periodic_transform_fn,
        )

        ckpt = torch.load(ckpt_path, map_location=self.device)

        ckpt_input_dim = ckpt["model"]["net.net.0.0.weight"].shape[1]
        if ckpt_input_dim != dynamics.input_dim:
            raise ValueError(
                f"Checkpoint input_dim={ckpt_input_dim}, but dynamics.input_dim={dynamics.input_dim}. "
                "This means the checkpoint and dynamics class do not match."
            )

        model.to(self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        self.dynamics = dynamics
        self.model = model

    def values(self, x):
        """
        args:
            x: torch tensor with shape      [batch_size, 8]
        returns:
            values: torch tensor with shape [batch_size]
        """
        x = x.to(self.device)

        coords = torch.concatenate(
            (torch.ones((len(x), 1), device=self.device), x),
            dim=1,
        )

        model_input = self.dynamics.coord_to_input(coords)

        with torch.no_grad():
            model_results = self.model({'coords': model_input})

        values = self.dynamics.io_to_value(
            model_results['model_in'].detach(),
            model_results['model_out'].detach().squeeze(dim=-1),
        )

        return values.cpu()

    def gradients(self, x):
        """
        args:
            x: torch tensor with shape         [batch_size, 8]
        returns:
            gradients: torch tensor with shape [batch_size, 8]
        """
        x = x.to(self.device)

        coords = torch.concatenate(
            (torch.ones((len(x), 1), device=self.device), x),
            dim=1,
        )

        model_input = self.dynamics.coord_to_input(coords)
        model_results = self.model({'coords': model_input})

        gradients = self.dynamics.io_to_dv(
            model_results['model_in'],
            model_results['model_out'].squeeze(dim=-1),
        )[:, 1:]

        return gradients.cpu()

    
"""
Helper class for querying cbf.ckpt.
neuralcbf = NeuralCBF()
h_values = neuralcbf.h_values(x)
h_gradients = neuralcbf.h_gradients(x)
"""
class NeuralCBF:
    def __init__(self, ckpt_path='outputs/cbf.ckpt'):
        try:
            from neural_clbf.controllers import NeuralCBFController
        except Exception as e:
            print(str(e))
            print('MAKE SURE YOU HAVE THE VENV FROM HW1 ACTIVATED')
        try:
            self.model = NeuralCBFController.load_from_checkpoint(ckpt_path)
        except Exception as e:
            print(str(e))
            print('MAKE SURE YOUR FILES FROM HOMEWORK 1 ARE IN THE SAME DIRECTORY AS THIS FILE')

    def values(self, x):
        """
        args:
            x: torch tensor with shape    [batch_size, 13]
        
        returns:
            h(x): torch tensor with shape [batch_size]
        """
        return -self.model.V_with_jacobian(x)[0]
    
    def gradients(self, x):
        """
        args:
            x: torch tensor with shape       [batch_size, 13]

        returns:
            dhdx(x): torch tensor with shape [batch_size, 13]
        """
        return -self.model.V_with_jacobian(x)[1].squeeze(1)