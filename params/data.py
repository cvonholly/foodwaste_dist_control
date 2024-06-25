import numpy as np


base = {
    "nv_p_d": 3.550,  # Mcal nutritional value produced per person per day
    "fw_p_d": 1.160,  # Mcal food waste per person per day

}

params = {
    "test": {
        "horizon": 10,   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
        "T" : 5,  # food waste time horizon
        "n_ps" : 2,   # set number producers
        "n_cs" : 4,  # set number consumers
        "n_scs" : 3,   # set number social-charities

        "food_waste" :0.08,  # food waste percentage/ratio at producer level
        "T_start" :3,  # when producers start to give food products to charity
        "T_end" :4,  # when producers end to give food products to charity
        "alpha_start" :0.25,  # amount/percentage of food given to charity at beginning
        "alpha_end" :0.5,  # amount/percentage of food given to charity at end
        "alpha_first_day" :0.2,  # probability that food gets consumed from consumer on first day
        "alpha_last_day" :0.8,  # probability that food gets consumed from consumer on last day
        "f0" :1,  #inflow to producer
        "x0": np.zeros((5, 1)),  # nodes initial state
        "inp_params": (1, 3)   # (x input, t time) tuple repreesenting input flow for producers
    },
    # base case model
    "SCS": {
        "horizon": 10,   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
        "T" : 5,  # food waste time horizon
        "n_ps" : 1,   # set number producers
        "n_cs" : 10,  # set number consumers
        "n_scs" : 1,   # set number social-charities

        "food_waste" :0.3,  # food waste percentage/ratio at producer level
        "T_start" : 3,  # when producers start to give food products to charity
        "T_end" : 4,  # when producers end to give food products to charity
        "alpha_start" : 0.25,  # amount/percentage of food given to charity at beginning
        "alpha_end" : 0.5,  # amount/percentage of food given to charity at end
        "alpha_first_day" :0.2,  # probability that food gets consumed from consumer on first day
        "alpha_last_day" :0.5,  # probability that food gets consumed from consumer on last day
        "f0" :1,  #inflow to producer
        "x0": np.zeros((5, 1)),  # nodes initial state
        "inp_params": ('base case', 3)   # (x input, t time) tuple repreesenting input flow for producers
    },
    "SHFW": {
        "horizon": 10,   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
        "T" : 5,  # food waste time horizon
        "n_ps" : 2,   # set number producers
        "n_cs" : 4,  # set number consumers
        "n_scs" : 3,   # set number social-charities

        "food_waste" :0.3,  # food waste percentage/ratio at producer level
        "T_start" :3,  # when producers start to give food products to charity
        "T_end" :4,  # when producers end to give food products to charity
        "alpha_start" :0.25,  # amount/percentage of food given to charity at beginning
        "alpha_end" :0.5,  # amount/percentage of food given to charity at end
        "alpha_first_day" :0.2,  # probability that food gets consumed from consumer on first day
        "alpha_last_day" :0.5,  # probability that food gets consumed from consumer on last day
        "f0" :1,  #inflow to producer
        "x0": np.zeros((5, 1)),  # nodes initial state
        "inp_params": (1, 3)   # (x input, t time) tuple repreesenting input flow for producers
    },
    "SLFW": {
        "horizon": 10,   # simulation horizon: has to be sufficiently longer then T for simulation to be correct !
        "T" : 5,  # food waste time horizon
        "n_ps" : 2,   # set number producers
        "n_cs" : 4,  # set number consumers
        "n_scs" : 3,   # set number social-charities

        "food_waste" :0.08,  # food waste percentage/ratio at producer level
        "T_start" :3,  # when producers start to give food products to charity
        "T_end" :4,  # when producers end to give food products to charity
        "alpha_start" :0.25,  # amount/percentage of food given to charity at beginning
        "alpha_end" :0.5,  # amount/percentage of food given to charity at end
        "alpha_first_day" :0.2,  # probability that food gets consumed from consumer on first day
        "alpha_last_day" :0.8,  # probability that food gets consumed from consumer on last day
        "f0" :1,  #inflow to producer
        "x0": np.zeros((5, 1)),  # nodes initial state
        "inp_params": (1, 3)   # (x input, t time) tuple repreesenting input flow for producers
    },
    
}