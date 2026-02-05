from dataclasses import dataclass
import yaml
import os
from ament_index_python.packages import get_package_share_directory


@dataclass
class DynamicsConfig:
    """
    Vehicle dynamics configuration dataclass.
    This dataclass contains the vehicle dynamics parameters used for control and planning.

    Args:
        MIN_STEER: float - minimum steering angle [rad]
        MAX_STEER: float - maximum steering angle [rad]
        MIN_DSTEER: float - minimum steering rate [rad/s]
        MAX_DSTEER: float - maximum steering rate [rad/s]
        MAX_SPEED: float - maximum speed [m/s]
        MIN_SPEED: float - minimum speed [m/s]
        MAX_ACCEL: float - maximum acceleration [m/s^2]
        MIN_ACCEL: float - minimum acceleration [m/s^2]

        WHEELBASE: float - distance between front and rear axles [m]
        MU: float - friction coefficient
        C_SF: float - cornering stiffness front
        C_SR: float - cornering stiffness rear
        LF: float - distance from center of mass to front axle
        LR: float - distance from center of mass to rear axle
        H: float - height of center of mass
        M: float - mass of the vehicle
        I: float - moment of inertia
    """

    MIN_STEER: float
    MAX_STEER: float
    MIN_DSTEER: float
    MAX_DSTEER: float
    MAX_SPEED: float
    MIN_SPEED: float
    MAX_ACCEL: float
    MIN_ACCEL: float
    WHEELBASE: float

    # Vehicle parameters
    MU: float
    C_SF: float
    C_SR: float
    BF: float
    BR: float
    DF: float
    DR: float
    CF: float
    CR: float

    LF: float
    LR: float
    H: float
    M: float
    I: float


def _dynamics_config_from_gym_params(gym_params):
    """
    Generate a dynamics configuration object from gym environment parameters.
    This function extracts and computes various vehicle dynamics parameters
    from a dictionary of gym environment parameters. It also calculates
    additional parameters if they are not explicitly provided in the input.
    Args:
        gym_params (dict): A dictionary containing the following keys:
            - "s_min" (float): Minimum steering angle.
            - "s_max" (float): Maximum steering angle.
            - "sv_min" (float): Minimum steering velocity.
            - "sv_max" (float): Maximum steering velocity.
            - "v_max" (float): Maximum speed.
            - "v_min" (float): Minimum speed.
            - "a_max" (float): Maximum acceleration.
            - "a_min" (float, optional): Minimum acceleration. Defaults to -a_max.
            - "lf" (float): Distance from the center of mass to the front axle.
            - "lr" (float): Distance from the center of mass to the rear axle.
            - "mu" (float): Friction coefficient.
            - "C_Sf" (float): Linear cornering stiffness for the front tires.
            - "C_Sr" (float): Linear cornering stiffness for the rear tires.
            - "h" (float): Height of the center of mass.
            - "m" (float): Mass of the vehicle.
            - "I" (float): Moment of inertia of the vehicle.
            - Optional keys:
                - "Bf" (float): Tire stiffness factor for the front tires.
                - "Br" (float): Tire stiffness factor for the rear tires.
                - "Df" (float): Peak vertical force on the front tires scaled by friction.
                - "Dr" (float): Peak vertical force on the rear tires scaled by friction.
                - "Cf" (float): Pacejka shape factor for the front tires.
                - "Cr" (float): Pacejka shape factor for the rear tires.
    Returns:
        DynamicsConfig: An object containing the vehicle dynamics parameters. Check the `DynamicsConfig` class for more details.
    Notes:
        - If certain optional parameters are not provided in `gym_params`,
          they are computed using typical values or derived from other parameters.
        - The function assumes a gravitational acceleration of 9.81 m/s^2.
    """
    MIN_STEER = gym_params["s_min"]
    MAX_STEER = gym_params["s_max"]
    MIN_DSTEER = gym_params["sv_min"]
    MAX_DSTEER = gym_params["sv_max"]
    MAX_SPEED = gym_params["v_max"]
    MIN_SPEED = gym_params["v_min"]
    MAX_ACCEL = gym_params["a_max"]
    MIN_ACCEL = gym_params["a_min"] if "a_min" in gym_params else -gym_params["a_max"]
    WHEELBASE = gym_params["lf"] + gym_params["lr"]

    MU = gym_params["mu"]
    C_SF = gym_params["C_Sf"]
    C_SR = gym_params["C_Sr"]
    LF = gym_params["lf"]
    LR = gym_params["lr"]
    H = gym_params["h"]
    M = gym_params["m"]
    I = gym_params["I"]

    if "bf" in gym_params:
        BF = gym_params["Bf"]
    if "br" in gym_params:
        BR = gym_params["Br"]
    if "df" in gym_params:
        DF = gym_params["Df"]
    if "dr" in gym_params:
        DR = gym_params["Dr"]
    if "cf" in gym_params:
        CF = gym_params["Cf"]
    if "cr" in gym_params:
        CR = gym_params["Cr"]

    g = 9.81
    Fz_total = M * g
    # Distribute weight based on distances from the center of mass
    Fzf = Fz_total * LR / (LF + LR)
    Fzr = Fz_total * LF / (LF + LR)

    if "df" not in gym_params:
        # Compute peak vertical force on the front tires scaled by friction
        DF = MU * Fzf

    if "dr" not in gym_params:
        # Compute peak vertical force on the rear tires scaled by friction
        DR = MU * Fzr

    if "cf" not in gym_params:
        # Typical Pacejka shape factor for the front tires (commonly around 1.3)
        CF = 1.3

    if "cr" not in gym_params:
        # Typical Pacejka shape factor for the rear tires (commonly around 1.3)
        CR = 1.3

    if "bf" not in gym_params:
        # Compute tire stiffness factor for the front tires using the linear cornering stiffness
        BF = C_SF / (CF * DF) if DF != 0 else 0.0

    if "br" not in gym_params:
        # Compute tire stiffness factor for the rear tires using the linear cornering stiffness
        BR = C_SR / (CR * DR) if DR != 0 else 0.0

    return DynamicsConfig(
        MIN_STEER=MIN_STEER,
        MAX_STEER=MAX_STEER,
        MIN_DSTEER=MIN_DSTEER,
        MAX_DSTEER=MAX_DSTEER,
        MAX_SPEED=MAX_SPEED,
        MIN_SPEED=MIN_SPEED,
        MAX_ACCEL=MAX_ACCEL,
        MIN_ACCEL=MIN_ACCEL,
        WHEELBASE=WHEELBASE,
        MU=MU,
        C_SF=C_SF,
        C_SR=C_SR,
        BF=BF,
        BR=BR,
        DF=DF,
        DR=DR,
        CF=CF,
        CR=CR,
        LF=LF,
        LR=LR,
        H=H,
        M=M,
        I=I,
    )

def load_dynamics_config_from_yaml(yaml_file: str = "models_param.yaml"):
    """
    Load dynamics parameters from a YAML file and return a DynamicsConfig object.
    It initializes with default f1tenth parameters and overrides them with values from the YAML.
    """
    # 기본 파라미터 설정
    params = {
        'MIN_STEER': -0.4189, 'MAX_STEER': 0.4189,
        'MIN_DSTEER': -3.2, 'MAX_DSTEER': 3.2,
        'MAX_SPEED': 20.0, 'MIN_SPEED': 0.0,
        'MAX_ACCEL': 9.51, 'MIN_ACCEL': -13.26,
        'MU': 1.0489, 
        'C_SF': 4.718, 'C_SR': 5.4562,
        'BF': 1.0, 'BR': 1.0, 'DF': 1.0, 'DR': 1.0, 'CF': 1.0, 'CR': 1.0,
        'LF': 0.15875, 'LR': 0.17145, 
        'H': 0.074, 'M': 3.74, 'I': 0.04712,
        'WHEELBASE': 0.3302
    }

    if not os.path.isabs(yaml_file):
        try:
            share_dir = get_package_share_directory('mppi')
            yaml_file = os.path.join(share_dir, 'config', yaml_file)
        except Exception:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_file = os.path.join(current_dir, yaml_file)

    if not os.path.exists(yaml_file):
        print(f"Warning: {yaml_file} not found. Using default f1tenth parameters.")
        return DynamicsConfig(**params)

    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    params_dict = data
    if 'mppi_node' in data and 'ros__parameters' in data['mppi_node']:
        params_dict = data['mppi_node']['ros__parameters']
    
    # YAML 키를 DynamicsConfig 필드에 매핑
    key_map = {
        'm': 'M', 'Iz': 'I', 'lf': 'LF', 'lr': 'LR',
        'Bf': 'BF', 'Cf': 'CF', 'Df': 'DF',
        'Br': 'BR', 'Cr': 'CR', 'Dr': 'DR',
        # 제약 조건 및 기타 파라미터
        's_min': 'MIN_STEER', 's_max': 'MAX_STEER',
        'sv_min': 'MIN_DSTEER', 'sv_max': 'MAX_DSTEER',
        'v_min': 'MIN_SPEED', 'v_max': 'MAX_SPEED',
        'a_min': 'MIN_ACCEL', 'a_max': 'MAX_ACCEL',
        'mu': 'MU', 'h': 'H'
    }

    for yaml_key, obj_attr in key_map.items():
        if yaml_key in params_dict:
            params[obj_attr] = float(params_dict[yaml_key])

    # 휠베이스 등 파생 파라미터 재계산
    if 'lf' in params_dict and 'lr' in params_dict:
        params['WHEELBASE'] = params['LF'] + params['LR']

    # Pacejka 계수 기반 코너링 강성 계산
    if 'Bf' in params_dict and 'Cf' in params_dict and 'Df' in params_dict:
        params['C_SF'] = params['BF'] * params['CF'] * params['DF']
    if 'Br' in params_dict and 'Cr' in params_dict and 'Dr' in params_dict:
        params['C_SR'] = params['BR'] * params['CR'] * params['DR']

    return DynamicsConfig(**params)


def update_config_from_dict(self, config_dict):
    """
    Update the dynamics configuration object with new parameters from a dictionary.

    Args:
        config_dict (dict): A dictionary containing the new parameters to update.
    """
    for key, value in config_dict.items():
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Key '{key}' not found in DynamicsConfig.")
