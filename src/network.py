
import numpy as np
from copy import deepcopy


class Random_Network:
    def __init__(self, Psi, step_func, spiking_condition_func):
        self.Psi = Psi

        self.step_func = step_func
        self.spiking_condition_func = spiking_condition_func

        self.network_log = []
        self.reconstruction_log = []

    def step_network(self, t,**kwargs):
        kwargs['t'] = t
        self.Psi = self.step_func(self, self.Psi, **kwargs)

    def log_network(self, t,**kwargs):
        if t % 1000 == 0:
            self.network_log.append(deepcopy(self.Psi))
        self.reconstruction_log.append(deepcopy({k: v for k, v in self.Psi.items() if k in {'r', 'r_total','s', 'U', 'p'}}))
        # bottom up cause
        self.reconstruction_log[-1]['c_hat'] = self.Psi['D_c'].dot(self.Psi['r'])
        # top down cause
        self.reconstruction_log[-1]['c_td'] = self.Psi['D_td'].dot(self.Psi['x_td'])
        # context loss
        self.reconstruction_log[-1]['cl'] = self.Psi['context_loss_memory_mean']

def iterate_trace(t,s,e_tau):
    return t * e_tau + s

""" Base Psi """
def Psi_base_mu(d_stim, d_td, d_cau, n_neurons, dt):
    Psi = {}
    Psi['d_stim'] = d_stim
    Psi['d_td'] = d_td
    Psi['d_cau'] = d_stim
    Psi['n_neurons'] = n_neurons
    Psi['dt'] = dt

    return Psi

""" Network definition """
def mu_inference_Psi(Psi,tau, **kwargs):
    Psi["nu"] = kwargs.get('nu', np.zeros(Psi["n_neurons"]))
    Psi["nu_U"] = -np.log(Psi["nu"])
    
    Psi["tau"] = tau
    Psi["e_tau"] = np.exp(-Psi["dt"] / Psi["tau"])

    Psi["x"] = np.zeros(Psi["d_stim"])
    Psi["x_td"] = np.zeros(Psi["d_td"])
    Psi["U"] = np.zeros(Psi["n_neurons"])

    Psi["p"] = np.zeros(Psi["n_neurons"])
    Psi["s"] = np.zeros(Psi["n_neurons"])

    # Traces used to calculate r,q,o
    #Psi["trace"] = np.zeros(Psi["n_neurons"])

    # r(eadout) trace r
    Psi["r"] = np.zeros(Psi["n_neurons"])
    #Psi["r_p"] = np.zeros(Psi["n_neurons"])
    Psi["c_0"] = kwargs.get("c_0",np.zeros(Psi["d_stim"]))
    Psi["delta_t"] = kwargs.get("delta_t",1)
    #Psi["delta_i"] = int(Psi["delta_t"]/Psi["dt"])
    Psi["r_total"] = 0

    Psi["D_c"] = kwargs.get('D_c', np.zeros((Psi['d_cau'],Psi["n_neurons"]))+np.random.randn(Psi['d_cau'],Psi["n_neurons"])*0.5)
    Psi["D_x"] = kwargs.get('D_x', np.zeros((Psi['d_stim'], Psi['d_cau'])))
    Psi["D_td"] = kwargs.get('D_td', np.zeros((Psi['d_cau'],Psi['d_td'])))
    Psi["beta_x"] = kwargs.get("beta_x",1.)

    Psi['beta_c_low'] = kwargs.get('beta_c_low',1)
    Psi['beta_c_high'] = kwargs.get('beta_c_high',25)

    Psi["beta_c_hat"] = np.atleast_1d(kwargs.get("beta_c_hat",1.) )

    Psi["context_loss_memory_length"] = int(0.5/Psi['dt'])
    Psi["context_loss_memory"] = np.zeros(Psi["context_loss_memory_length"])
    Psi["context_loss_memory_mean"] = 0

    Psi['context_switching_threshold'] = kwargs.get('context_switching_threshold',0.2)

    #Psi['bottom_up_beta'] = Psi['beta_x']

    Psi["loss"] = 0

def mu_inference_step_func(network, Psi, **kwargs):
    
    Psi['x'] = kwargs['x']
    Psi['x_td'] = kwargs['x_td']

    Psi["U"] = Psi["D_c"].T.dot(
        Psi["beta_x"]*Psi['D_x'].T.dot(Psi["x"]) + Psi["beta_c_hat"]*Psi['D_td'].dot(Psi['x_td'])
      - (Psi['D_x'].T.dot(Psi['D_x'])*Psi["beta_x"] + np.eye(3)*Psi["beta_c_hat"]).dot(Psi["D_c"].dot(Psi["r"]))
    ) - Psi['nu_U'] \
    -np.diag(Psi["D_c"].T.dot((Psi["D_x"].T.dot(Psi["D_x"])*Psi['beta_x'] + np.eye(3)*Psi['beta_c_hat']).dot(Psi["D_c"])))/2


    Psi = network.spiking_condition_func(Psi)
    Psi["r"] = Psi["r"] * Psi['e_tau'] + Psi["s"]
    Psi["r_total"] = np.sum(Psi["r"])


    Psi["loss_components"] = loss(Psi)
    Psi["context_loss_memory"][1:] = Psi["context_loss_memory"][:-1]
    Psi["context_loss_memory"][0] = Psi["loss_components"]
    Psi["context_loss_memory_mean"] = np.mean(Psi["context_loss_memory"])

    #Context switching

    if Psi['context_loss_memory_mean'] < Psi['context_switching_threshold']:
        Psi['beta_c_hat'] = Psi['beta_c_high']
    else:
        Psi['beta_c_hat'] = Psi['beta_c_low']


    return Psi

def loss(Psi):
    e_td = Psi['D_c'].dot(Psi['r']) - Psi['D_td'].dot(Psi['x_td'])
    return np.inner(e_td,e_td)

def mu_inference_spiking_condition(Psi):
    p = np.random.rand(Psi["n_neurons"])
    Psi['p'] = 1/(1+np.exp(-Psi["U"])/Psi["dt"])
    #Psi['p'] = Psi['dt']*np.exp(Psi["U"])
    Psi['s'] = (Psi['p'] > p)*1
    #if np.sum(Psi['s']) > 1:
    #    Psi['s'] = random_one(Psi['s'])
    return Psi