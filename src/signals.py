import numpy as np

def rolling_average(s,dt,Delta_t):
    
    def convolve_along_axis(signal, kernel, axis):
        return np.apply_along_axis(lambda x: np.convolve(x, kernel,mode='same'), axis, signal)

    kernel = np.exp(-np.arange(3*int(Delta_t/dt))*dt/Delta_t)
    kernel /= np.sum(kernel)


    num_spikes = convolve_along_axis(s, kernel, axis=1)

    rates = num_spikes / dt

    return rates

def signal_context_switching_simple(n_timesteps, dt, x_duration, p, prediction_offset, k=0,tau=0.1):
    """
    Returns a signal vector which indexes at time 't' and generates both bottom up and top down signals.
    
    Parameters:
    n_timesteps (int): Total number of timesteps.
    dt (float): The time difference used in the pattern.
    x_duration (float): Duration of the pattern.
    p (float): Probability of the "right" bottom up pattern.
    prediction_offset (int): The length of the prediction time window.
    k (int, optional): Parameter for evidence experiment, defaults to 0.

    Returns:
    x_bottom_up (np.array): The bottom up signals.
    x_top_down (np.array): The top down signals.
    ts_1 (int): Timeseries marker 1.
    ts_2 (int): Timeseries marker 2.
    """

    # Split the total time steps into the duration of the patterns.
    N_x = 3
    N_td = 2

    total_patterns = int(np.ceil(n_timesteps*dt/x_duration/2))

    ts_1 = int((total_patterns-1)*2*x_duration/dt)
    ts_2 = int((total_patterns-2)*2*x_duration/dt)

    # Generate the different patterns. Set with probability p of the first pattern.
    patterns = np.eye(N_x)
    patterns_i = np.random.choice([0,1],total_patterns,p=[p, (1-p)])

    # Change the last two patterns deterministically.
    patterns_i[-1] = 0
    patterns_i[-2] = 1

    # This is for the evidence experiment.
    if not k == 0:
        patterns = np.array([
            [1,0,0],
            [k,1-k,0],
            [0,0,1]
        ])

    # Initialize signals.
    x_bottom_up = np.zeros((n_timesteps,N_x))
    x_top_down = np.zeros((n_timesteps,N_td))

    alpha = np.exp(-dt/tau)

    # Iterate through time steps.
    for t in np.arange(n_timesteps - 1)+1:
        # x bottom up signal generation.
        if np.floor(t * (dt/x_duration) ) % 2 == 0:
            p_i = patterns_i[int(np.floor(t * (dt/x_duration) / 2))]
            x_bottom_up[t] = x_bottom_up[t-1]*alpha + (1-alpha)*patterns[p_i]

        else:
            x_bottom_up[t] = x_bottom_up[t-1]*alpha + (1-alpha)*patterns[-1]

        # x top down signal generation.
        if (np.floor((t + prediction_offset /dt ) * (dt/x_duration) ) % 2 == 0):
            x_top_down[t] = x_top_down[t-1]*alpha + (1-alpha)*np.array([1,0]) 
        else:
            x_top_down[t] = x_top_down[t-1]*alpha + (1-alpha)*np.array([0,1]) 

    return np.arange(n_timesteps)*dt, x_bottom_up, x_top_down, ts_1, ts_2

def calcium_kernel(t, tau_rise, tau_decay, dt):
    """ calcium kernel """

    return (1 - np.exp(-t * dt / tau_rise)) * np.exp(-t * dt / tau_decay)

def spike_to_F(spike_data, tau_rise, tau_decay,dt,baseline):
    """ we convolve the spike data with a calcium kernel to get the fluorescence signal """

    kernel = calcium_kernel(np.arange(5*int(tau_decay/dt)), tau_rise, tau_decay, dt)
    F = np.array([np.convolve(sd, kernel, mode='same')[:-int(5/2*tau_decay/dt)] for sd in spike_data])

    F+= baseline

    return F