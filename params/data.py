import numpy as np


base = {
    "nv_p_d": 3.550,  # Mcal nutritional value produced per person per day
    "fw_p_d": 1.160,  # Mcal food waste per person per day
}

params = {
    # base case model
    "SCS": {
        "horizon": 20,   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
        "T" : 10,  # food waste time horizon
        "n_ps" : 1,   # set number producers
        "n_cs" : 10,  # set number consumers
        "n_scs" : 1,   # set number social-charities
        'fw_model': {
            'type': 'simple',
            'store': .73,  # relative amount of food stored at every time step
        },
        # "food_waste" : 0.15,  # food waste percentage/ratio at producer level
        "T_start" : 3,  # when producers start to give food products to charity
        "T_end" : 4,  # when producers end to give food products to charity
        "alpha_start" : 0.0,  # amount/percentage of food given to charity at beginning
        "alpha_end" : 0.0,  # amount/percentage of food given to charity at end
        "alpha_first_day" :0.2,  # probability that food gets consumed from consumer on first day
        "alpha_last_day" :0.3,  # probability that food gets consumed from consumer on last day
        "x0_c": np.reshape(np.genfromtxt('params/SCS_final_consumer_state.csv', delimiter=','), (10, 1)),  # initial consumers state
        "x0": np.zeros((10, 1)),  # nodes initial state
        "inp_params": ('base case', 20),   # see params.py
        "ec_mpc": False,  # optimize P input with economic MPC
        "mpc_h": 5,
        "ec_mpc_c": False,  # optimize C inout with economic MPC
        "mpc_h_c": 5,
        "food_intake": base["nv_p_d"] - base["fw_p_d"],  # daily food intake required by consumers
        "fb": {}
    },
    # base case model with MPC FOR CONSUMER
    "EC_MPC": {
        "horizon": 20,   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
        "T" : 10,  # food waste time horizon
        "n_ps" : 1,   # set number producers
        "n_cs" : 10,  # set number consumers
        "n_scs" : 1,   # set number social-charities
        'fw_model': {
            'type': 'simple',
            'store': .73,  # relative amount of food stored at every time step
        },
        "T_start" : 3,  # when producers start to give food products to charity
        "T_end" : 4,  # when producers end to give food products to charity
        "alpha_start" : 0.0,  # amount/percentage of food given to charity at beginning
        "alpha_end" : 0.0,  # amount/percentage of food given to charity at end
        "alpha_first_day" :0.2,  # probability that food gets consumed from consumer on first day
        "alpha_last_day" :0.3,  # probability that food gets consumed from consumer on last day
        "x0_c": np.reshape(np.genfromtxt('params/SCS_final_consumer_state.csv', delimiter=','), (10, 1)),  # initial consumers state
        "x0": np.zeros((10, 1)),  # nodes initial state
        "inp_params": ('base case', 20),   # see params.py
        "ec_mpc": False,  # optimize P input with economic MPC
        "mpc_h": 20,
        "ec_mpc_c": True,  # optimize C inout with economic MPC
        "mpc_h_c": 20,
        "food_intake": base["nv_p_d"] - base["fw_p_d"],  # daily food intake required by consumers
        "fb": {}
    },
    # feedback control model
    "FB1": {
        "horizon": 20,   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
        "T" : 10,  # food waste time horizon
        "n_ps" : 1,   # set number producers
        "n_cs" : 10,  # set number consumers
        "n_scs" : 1,   # set number social-charities
        'fw_model': {
            'type': 'simple',
            'store': .73,  # relative amount of food stored at every time step
        },
        # "food_waste" : 0.15,  # food waste percentage/ratio at producer level
        "T_start" : 3,  # when producers start to give food products to charity
        "T_end" : 4,  # when producers end to give food products to charity
        "alpha_start" : 0.0,  # amount/percentage of food given to charity at beginning
        "alpha_end" : 0.0,  # amount/percentage of food given to charity at end
        "alpha_first_day" :0.2,  # probability that food gets consumed from consumer on first day
        "alpha_last_day" :0.3,  # probability that food gets consumed from consumer on last day
        "x0_c": np.reshape(np.genfromtxt('params/SCS_final_consumer_state.csv', delimiter=','), (10, 1)),  # initial consumers state
        "x0": np.zeros((10, 1)),  # nodes initial state
        "inp_params": ('base case', 20),   # see params.py
        "ec_mpc": False,  # optimize P input with economic MPC
        "mpc_h": 5,
        "ec_mpc_c": False,  # optimize C inout with economic MPC
        "mpc_h_c": 5,
        "food_intake": base["nv_p_d"] - base["fw_p_d"],  # daily food intake required by consumers
        "fb": {
            "K": .3,  # feedback gain
            "T": 1,  # time constant
            "x0": 0  # initial state
        }
    },    
}
