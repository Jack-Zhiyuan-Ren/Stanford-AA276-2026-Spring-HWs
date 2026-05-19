import numpy as np

class ControllerA:
    """
    Controller for the cart-pole system.
    This is a template for you to implement however you desire.
    
    reset(.) is called before each cart-pole simulation.
    u_fn(.) is called at each simulation step.
    data_to_visualize(.) is called after each simulation.

    We provide example code for a random-walk controller.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.s_history = []
        self.t_history = []
        self.u_history = []

    def u_fn(self, s, t):
        """Control function for the cart-pole system.

        Args:
            s (np.ndarray): The current state: [x, theta, x_dot, theta_dot]
                NOTE: you might want to first wrap theta in [0, 2pi]
            t (float): The current time

        Returns:
            u (np.ndarray): The control input [u]
        """
        self.s_history.append(s)
        self.t_history.append(t)
        u = np.random.uniform(-10, 10)
        self.u_history.append(u)      
        return np.array([u])                      

    def data_to_visualize(self):
        """
        Use this to add any number of data visualizations to the animation.
        This is purely to help you debug, in case you find it helpful.
        See example code below to plot the control on a new axes at axes index 1.

        Returns:
            data_to_visualize (dict): Each dictionary entry should have the form:
                'y-axes label' (str): [axes index (int), data to visualize (np.ndarray), line styles (dict)]
        """
        s_history = np.array(self.s_history)
        t_history = np.array(self.t_history)
        u_history = np.array(self.u_history)
        return {
            'u (N)': [2, u_history, {'color': 'k'}],
            '$\\theta$ (rad)': [3, s_history[:, 1] % (2*np.pi), {'color': 'k'}],
            '$\\theta_\\text{min}$ (rad)': [3, (np.pi/2)*np.ones(len(t_history)), {'color': 'r', 'linestyle': '--'}],
            '$\\theta_\\text{max}$ (rad)': [3, (3*np.pi/2)*np.ones(len(t_history)), {'color': 'r', 'linestyle': '--'}]
        }
    
class ControllerB:
    """
    Controller for the cart-pole system.
    This is a template for you to implement however you desire.
    
    reset(.) is called before each cart-pole simulation.
    u_fn(.) is called at each simulation step.
    data_to_visualize(.) is called after each simulation.

    We provide example code for a random controller.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.s_history = []
        self.t_history = []
        self.u_history = []

    def u_fn(self, s, t):
        """Control function for the cart-pole system.

        Args:
            s (np.ndarray): The current state: [x, theta, x_dot, theta_dot]
                NOTE: you might want to first wrap theta in [0, 2pi]
            t (float): The current time

        Returns:
            u (np.ndarray): The control input [u]
        """
        self.s_history.append(s)
        self.t_history.append(t)
        u = np.random.uniform(-10, 10)
        self.u_history.append(u)      
        return np.array([u])                      

    def data_to_visualize(self):
        """
        Use this to add any number of data visualizations to the animation.
        This is purely to help you debug, in case you find it helpful.
        See example code below to plot the control on a new axes at axes index 1.

        Returns:
            data_to_visualize (dict): Each dictionary entry should have the form:
                'y-axes label' (str): [axes index (int), data to visualize (np.ndarray), line styles (dict)]
        """
        s_history = np.array(self.s_history)
        t_history = np.array(self.t_history)
        u_history = np.array(self.u_history)
        return {
            'u (N)': [2, u_history, {'color': 'k'}],
            '$\\theta$ (rad)': [3, s_history[:, 1] % (2*np.pi), {'color': 'k'}],
            '$\\theta_\\text{min}$ (rad)': [3, (np.pi/2)*np.ones(len(t_history)), {'color': 'r', 'linestyle': '--'}],
            '$\\theta_\\text{max}$ (rad)': [3, (3*np.pi/2)*np.ones(len(t_history)), {'color': 'r', 'linestyle': '--'}]
        }