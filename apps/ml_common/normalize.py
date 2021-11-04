import numpy as np
from datetime import datetime

day = 24 * 60 * 60
year = (365.2425) * day

def timestamp_to_sincos(timestamp):
    return (
        np.sin(timestamp * (2 * np.pi / day)), #day sin
        np.cos(timestamp * (2 * np.pi / day)), #day cos
        np.sin(timestamp * (2 * np.pi / year)), #year sin
        np.cos(timestamp * (2 * np.pi / year)) #year cos
    )

def datetime_to_sincos(dt):
    return timestamp_to_sincos(datetime.timestamp(dt))

t_norm = lambda x: (x - 18.9) / 8.4
t_denorm = lambda x: x * 8.4 + 18.9
h_norm = lambda x: x / 100
h_denorm = lambda x: x * 100
l_norm = lambda x: x / 8000
l_denorm = lambda x: x * 8000
co2_norm = lambda x: (x - 726) / 333
co2_denorm = lambda x: x * 333 + 726

#rt[0] = (rt[0] - 12.40) / 10.56 # t
#rt[1] = (rt[1] - 77.78) / 21.29 # h
#rt[2] = (rt[2] - 968.93) / 1643.67 # l

