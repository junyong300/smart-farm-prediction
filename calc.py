import math

def wbt(t, rh):
    if rh is None or rh > 100 or rh <= 0:
        raise Exception("Invalid humidity")
    if rh == 100:
        return t
    return (t * math.atan(0.151977 * math.pow((rh + 8.313659), 0.5))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * math.pow(rh, 1.5) * math.atan(0.023101 * rh)
        - 4.686035)

def dpt(t, rh):
    '''dewpoint temp 이슬점'''
    dpc = (math.log(rh / 100) + ((17.27 * t) / (237.3 + (t * 1)))) / 17.27
    return ((237.3 * dpc) / (1 - dpc))

def pws(t):
    '''saturated water vaper pressure 포화수증기압'''
    return (6.116441 * math.pow(10, (7.591386 * t / (240.7263 + t))))

def pw(pws, rh):
    return (pws * rh / 100)

def ah(t, pw):
    return 2.16679 * pw * 100 / (273.15 + t)


def smc(t, pws):
    return 2.16679 * pws * 100 / (273.15 + t)

def hd(smc, ah):
    return smc - ah

def hdByTH(t, rh):
    _pws = pws(t)
    _pw = pw(_pws, rh)
    _smc = smc(t, _pws)
    _ah = ah(t, _pw)

    return hd(_smc, _ah)