# import stuff
import ROOT

# randomness initialization
uniform_rndm = ROOT.gRandom.Uniform  # uniform random number generator
gaus_rndm = ROOT.gRandom.Gaus  # gaussian random number generator

def gaussian(datapoints=100):
    v0 = []
    target = []

    for point in range(datapoints):
        x = uniform_rndm(-5, 5)
        y = ROOT.TMath.Gaus(x, 1, 2) + gaus_rndm(0, 0.2)

        v0.append(x)
        target.append(y)

    reg_vars = [v0]
    return {'reg_vars': reg_vars,
            'target': target}


def exponential(datapoints=100):
    v0 = []
    target = []

    for point in range(datapoints):
        x = uniform_rndm(-5, 5)
        y = ROOT.TMath.Exp(x) - 3 + gaus_rndm(0, 0.2)

        v0.append(x)
        target.append(y)

    reg_vars = [v0]
    return {'reg_vars': reg_vars,
            'target': target}


def sinusoidal_1(datapoints=100):
    v0 = []
    target = []

    for point in range(datapoints):
        x = uniform_rndm(0, 10)
        y = ROOT.TMath.Sin(x / 2) + gaus_rndm(0, 0.2)

        v0.append(x)
        target.append(y)

    reg_vars = [v0]
    return {'reg_vars': reg_vars,
            'target': target}


def sinusoidal_2(datapoints=100):
    v0 = []
    target = []

    for point in range(datapoints):
        x = uniform_rndm(0, 15)
        y = ROOT.TMath.Sin(3 * x) + x + gaus_rndm(0, 0.2)

        v0.append(x)
        target.append(y)

    reg_vars = [v0]
    return {'reg_vars': reg_vars,
            'target': target}


datasets = {
    'gaussian': {'function': gaussian,
                 'name': 'gaussian',
                 'nreg': 1},
    'exponential': {'function': exponential,
                    'name': 'exponential',
                    'nreg': 1},
    'sinusoidal_1': {'function': sinusoidal_1,
                     'name': 'sinusoidal_1',
                     'nreg': 1},
    'sinusoidal_2': {'function': sinusoidal_2,
                     'name': 'sinusoidal_2',
                     'nreg': 1}
}
