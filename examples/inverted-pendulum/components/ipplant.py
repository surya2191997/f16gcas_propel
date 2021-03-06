import numpy as np


def model_state_update(model, time_t, state_pendulum, input_controller):
    a, b, _, _ = _ss_inv_pend_cont(model.parameters)
    return list((a @ np.array(state_pendulum)[:, np.newaxis] + b @ np.array(input_controller)[:, np.newaxis]).flatten())


def _ss_inv_pend_cont(parameters):
    """
    Inverted Pendulum System -- Linear State Space Representation
    Taken from
    http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlDigital#4
    :return: A, B, C and D matrices
    """
    mm = parameters["mm"]        # Mass of the cart
    m = parameters["m"]         # Mass of the pendulum
    b = parameters["b"]        # coefficient of friction on cart
    ii = parameters["ii"]       # Mass Moment of Inertia on Pendulum
    g = parameters["g"]         # Gravitational acceleration
    length = parameters["length"]         # length to pendulum COM
    p = ii * (mm + m) + mm * m * length ** 2     # Denominator

    # Continuous time system
    a = np.array([[0, 1, 0, 0],
                  [0, -(ii+m*length**2)*b/p,  (m**2*g*length**2)/p,   0],
                  [0, 0, 0, 1],
                  [0, -(m*length*b)/p,       m*g*length*(mm+m)/p,  0]])

    b = np.array([[0],
                  [(ii+m*length**2)/p],
                  [0],
                  [m*length/p]])

    c = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])

    d = np.array([[0], [0]])
    return a, b, c, d
