import numpy as np


# standard base values
base = {
    # base values
    "name": "small",  # name for store / defining model
    "nv_p_d": 3.550,  # Mcal nutritional value produced per person per day
    "fw_p_d": 1.160,  # Mcal food waste per person per day
    "horizon": 100,   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
    "inp_params": ('base case', 100),   # see params.py
    "T": 10,   # food waste time horizon
    "n_ps" : 3 * 1,   # set number producers
    "n_cs" : 3 * 10,  # set number consumers
    "n_scs" : 3 * 1,   # set number social-charities
}


# big base values
# base = {
#     # base values
#     "name": "big",  # name for store / defining model
#     "nv_p_d": 3.550,  # Mcal nutritional value produced per person per day
#     "fw_p_d": 1.160,  # Mcal food waste per person per day
#     "horizon": 30,   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
#     "T": 10,   # food waste time horizon
#     "n_ps" : 20,   # set number producers
#     "n_cs" : 1000,  # set number consumers
#     "n_scs" : 10,   # set number social-charities
# }

params = {
    # busieness as usual model
    "SCS": {
        "name": base["name"],
        "horizon": base["horizon"],   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
        "T" : base["T"],  # food waste time horizon
        "n_ps" : base["n_ps"],   # set number producers
        "n_cs" : base["n_cs"],  # set number consumers
        "n_scs" : base["n_scs"],   # set number social-charities
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
        "x0": np.zeros((10, 1)),  # nodes initial state if not specified differenly
        "inp_params": base["inp_params"],   # see params.py
        "ec_mpc": False,  # optimize P input with economic MPC
        "mpc_h": 5,
        "ec_mpc_c": False,  # optimize C inout with economic MPC
        "mpc_h_c": 5,
        "food_intake": base["nv_p_d"] - base["fw_p_d"],  # daily food intake required by consumers
        "fb": {}
    },
    # base case model with MPC FOR CONSUMER
    "EC_MPC": {
        "name": base["name"],
        "horizon": base["horizon"],   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
        "T" : base["T"],  # food waste time horizon
        "n_ps" : base["n_ps"],   # set number producers
        "n_cs" : base["n_cs"],  # set number consumers
        "n_scs" : base["n_scs"],   # set number social-charities
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
        "inp_params": base["inp_params"],   # see params.py
        "ec_mpc": False,  # optimize P input with economic MPC
        "mpc_h": 20,
        "ec_mpc_c": True,  # optimize C inout with economic MPC
        "mpc_h_c": 20,
        "food_intake": base["nv_p_d"] - base["fw_p_d"],  # daily food intake required by consumers
        "fb": {}
    },
    # feedback control model
    "FB1": {
        "name": base["name"],
        "horizon": base["horizon"],   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
        "T" : base["T"],  # food waste time horizon
        "n_ps" : base["n_ps"],   # set number producers
        "n_cs" : base["n_cs"],  # set number consumers
        "n_scs" : base["n_scs"],   # set number social-charities
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
        "inp_params": base["inp_params"],   # see params.py
        "ec_mpc": False,  # optimize P input with economic MPC
        "mpc_h": 5,
        "ec_mpc_c": False,  # optimize C inout with economic MPC
        "mpc_h_c": 5,
        "food_intake": base["nv_p_d"] - base["fw_p_d"],  # daily food intake required by consumers
        "fb": {
            "K": .1/2,  # feedback gain
            "T": 1,  # time constant
            "x0": 0  # initial state
        }
    },    
    # greedy algorithm
    "greedy": {
        "name": base["name"],
        "horizon": base["horizon"],   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
        "T" : base["T"],  # food waste time horizon
        "n_ps" : base["n_ps"],   # set number producers
        "n_cs" : base["n_cs"],  # set number consumers
        "n_scs" : base["n_scs"],   # set number social-charities
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
        "inp_params": base["inp_params"],   # see params.py
        "ec_mpc": False,  # optimize P input with economic MPC
        "mpc_h": 5,
        "ec_mpc_c": False,  # optimize C inout with economic MPC
        "mpc_h_c": 5,
        "food_intake": base["nv_p_d"] - base["fw_p_d"],  # daily food intake required by consumers
        "fb": {
            # "K": .3,  # feedback gain
            # "T": 1,  # time constant
            # "x0": 0  # initial state
        },
        "greedy": True,
    },    
}
