
from tqdm import tqdm
import pickle

def init_network(inference_network):
    """
    Initializes a network for inference.

    Args:
    - inference_network (Random_Network): The inference network to be initialized.

    Returns:
    - inference_network (Random_Network): The generated inference network.
    """

    inference_network.Psi['r']*=0
    with open("signal_model_update.pickle", "rb") as file:
        t,bu,td,t1,t2 = pickle.load(file)

    loading_bar = tqdm(enumerate(zip(bu,td)), total= len(bu))
    for t_i, (bu_, td_) in loading_bar:
        inference_network.step_network(t_i,x=bu_, x_td = td_)
        inference_network.log_network(t_i)

    # We save the network based on its nu value    
    with open(f"./network.pickle",'wb') as file:
        pickle.dump(inference_network,file)

def generate_network_snapshots(nu,runs=50):

    """ First we generate the snapshots with full mismatch """
    print(f" --- full mismatch --- ")

    with open("signal_model_update_snapshot.pickle", "rb") as file:
        t,bu,td,t1,t2 = pickle.load(file)

    with_runs = []
    loading_bar = tqdm(range(runs))
    for run in loading_bar:
        with open(f"./network.pickle",'rb') as file:
            inference_network = pickle.load(file)

        inference_network.network_with_log = []
        inference_network.reconstruction_log = []

        
        for t_i, (bu_, td_) in enumerate(zip(bu,td)):
            inference_network.step_network(t_i,x=bu_, x_td = td_)
            inference_network.log_network(t_i)
        
        with_runs.append(inference_network)
    
    with open(f"./regular_runs.pickle",'wb') as file:
        pickle.dump(with_runs,file)

    """ Next we generate the snapshots with partial mismatch """
    print(f" --- partial mismatch --- ")

    with open("signal_model_update_snapshot_k.pickle", "rb") as file:
        t,bu,td,t1,t2 = pickle.load(file)

    with_runs = []
    loading_bar = tqdm(range(runs))
    for run in loading_bar:
        with open(f"./network.pickle",'rb') as file:
            inference_network = pickle.load(file)

        inference_network.network_with_log = []
        inference_network.reconstruction_log = []

        for t_i, (bu_, td_) in enumerate(zip(bu,td)):
            inference_network.step_network(t_i,x=bu_, x_td = td_)
            inference_network.log_network(t_i)
        
        with_runs.append(inference_network)
    
    with open(f"./mismatch_runs.pickle",'wb') as file:
        pickle.dump(with_runs,file)
