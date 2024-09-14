import ROOT

# import matplotlib.pyplot as plt


# ---------
# randomness initialization

uniform_rndm = ROOT.gRandom.Uniform  # uniform random number generator
gaus_rndm = ROOT.gRandom.Gaus  # gaussian random number generator


# ----------
# half moons
# stolen from scikit-learn

def moons(datapoints=200000):
    nPoints_signal = datapoints // 2  # half the points (whole number) are signal
    nPoints_background = datapoints - nPoints_signal  # other half are background

    # declare empty arrays for signal and background
    signal_x = []
    signal_y = []
    signal_sig = []  # will be a bunch of 1s

    background_x = []
    background_y = []
    background_sig = []  # will be a bunch of -1s

    for point in range(nPoints_signal):
        theta = uniform_rndm(0, ROOT.TMath.Pi())  # give the point a random value between 0 and pi (its theta value)

        signal_x.append(gaus_rndm(0, 0.2) + ROOT.TMath.Cos(theta))  # get the x value of the point
        signal_y.append(gaus_rndm(0, 0.1) + ROOT.TMath.Sin(theta))  # get the y value of the point
        signal_sig.append(1)

    for point in range(nPoints_background):
        theta = uniform_rndm(0, ROOT.TMath.Pi())  # give the point a random value between 0 and pi (its theta value)

        background_x.append(gaus_rndm(0, 0.2) + 1 - ROOT.TMath.Cos(theta))  # get the x value of the point
        background_y.append(gaus_rndm(0, 0.1) + 1 - ROOT.TMath.Sin(theta))  # get the y value of the point
        background_sig.append(-1)

    signal_vars = [signal_x, signal_y]
    background_vars = [background_x, background_y]

    return {'signal_vars': signal_vars,
            'signal_sig': signal_sig,
            'background_vars': background_vars,
            'background_sig': background_sig
            }


# -------------------------
# big circle inside small one
# only kind of stolen from scikit learn

def enclosed_circles(datapoints=200000):
    nPoints_signal = datapoints // 2  # half the points (whole number) are signal
    nPoints_background = datapoints - nPoints_signal  # other half are background

    # declare empty arrays for signal and background
    signal_x = []
    signal_y = []
    signal_sig = []  # will be a bunch of 1s

    background_x = []
    background_y = []
    background_sig = []  # will be a bunch of -1s

    for point in range(nPoints_signal):
        signal_x.append(gaus_rndm(0, 1))  # get the x value of the point
        signal_y.append(gaus_rndm(0, 1))  # get the y value of the point
        signal_sig.append(1)

    for point in range(nPoints_background):

        while True:
            x = gaus_rndm(0, 3)
            y = gaus_rndm(0, 3)

            if abs(x) > 1 or abs(y) > 1:  # if its outside most of the inner circle

                background_x.append(x)  # put the point in our array
                background_y.append(y)
                background_sig.append(-1)

                break  # keep trying until a point hits our criteria

    signal_vars = [signal_x, signal_y]
    background_vars = [background_x, background_y]

    return {'signal_vars': signal_vars,
            'signal_sig': signal_sig,
            'background_vars': background_vars,
            'background_sig': background_sig
            }


def wedges(datapoints=200000):
    nPoints_signal = datapoints // 2  # half the points (whole number) are signal
    nPoints_background = datapoints - nPoints_signal  # other half are background

    # declare empty arrays for signal and background
    signal_x = []
    signal_y = []
    signal_sig = []  # will be a bunch of 1s

    background_x = []
    background_y = []
    background_sig = []  # will be a bunch of -1s

    for point in range(nPoints_signal):
        signal_x.append(gaus_rndm(0, 1))  # get the x value of the point
        signal_y.append(gaus_rndm(0, 1))  # get the y value of the point
        signal_sig.append(1)

    for point in range(nPoints_background):

        while True:
            x = gaus_rndm(0, 3)
            y = gaus_rndm(0, 3)

            if abs(x) > 1 and abs(y) > 1:  # if its outside most of the inner circle

                background_x.append(x)  # put the point in our array
                background_y.append(y)
                background_sig.append(-1)

                break  # keep trying until a point hits our criteria

    signal_vars = [signal_x, signal_y]
    background_vars = [background_x, background_y]

    return {'signal_vars': signal_vars,
            'signal_sig': signal_sig,
            'background_vars': background_vars,
            'background_sig': background_sig
            }


def circle_portions(datapoints=200000):
    nPoints_signal = datapoints // 2  # half the points (whole number) are signal
    nPoints_background = datapoints - nPoints_signal  # other half are background

    # declare empty arrays for signal and background
    signal_x = []
    signal_y = []
    signal_sig = []  # will be a bunch of 1s

    background_x = []
    background_y = []
    background_sig = []  # will be a bunch of -1s

    for point in range(nPoints_signal):

        rndm_val = uniform_rndm(0, 1)

        if rndm_val >= 0.5:
            theta = uniform_rndm(0, 0.5 * ROOT.TMath.Pi())  # 1st quadrant
        else:
            theta = uniform_rndm(-1 * ROOT.TMath.Pi(), -0.5 * ROOT.TMath.Pi())  # 3rd quadrant

        signal_x.append(gaus_rndm(0, 0.2) + ROOT.TMath.Cos(theta))  # get the x value of the point
        signal_y.append(gaus_rndm(0, 0.1) + ROOT.TMath.Sin(theta))  # get the y value of the point
        signal_sig.append(1)

    for point in range(nPoints_background):

        rndm_val = uniform_rndm(0, 1)

        if rndm_val >= 0.5:
            theta = uniform_rndm(0.5 * ROOT.TMath.Pi(), ROOT.TMath.Pi())  # 2nd quadrant
        else:
            theta = uniform_rndm(-0.5 * ROOT.TMath.Pi(), 0)  # 4th quadrant

        background_x.append(gaus_rndm(0, 0.2) + ROOT.TMath.Cos(theta))  # get the x value of the point
        background_y.append(gaus_rndm(0, 0.1) + ROOT.TMath.Sin(theta))  # get the y value of the point
        background_sig.append(-1)

    signal_vars = [signal_x, signal_y]
    background_vars = [background_x, background_y]

    return {'signal_vars': signal_vars,
            'signal_sig': signal_sig,
            'background_vars': background_vars,
            'background_sig': background_sig
            }


def gaussian_circles(datapoints=200000):  # trivial case

    nPoints_signal = datapoints // 2  # half the points (whole number) are signal
    nPoints_background = datapoints - nPoints_signal  # other half are background

    # declare empty arrays for signal and background
    signal_x = []
    signal_y = []
    signal_sig = []  # will be a bunch of 1s

    background_x = []
    background_y = []
    background_sig = []  # will be a bunch of -1s

    for point in range(nPoints_signal):
        x = gaus_rndm(1, 1)
        y = gaus_rndm(1, 1)

        signal_x.append(x)
        signal_y.append(y)
        signal_sig.append(1)

    for point in range(nPoints_background):
        rndm_val = uniform_rndm(0, 5)

        x = gaus_rndm(-1, 1)
        y = gaus_rndm(-1, 1)

        background_x.append(x)
        background_y.append(y)
        background_sig.append(-1)

    signal_vars = [signal_x, signal_y]
    background_vars = [background_x, background_y]

    return {'signal_vars': signal_vars,
            'signal_sig': signal_sig,
            'background_vars': background_vars,
            'background_sig': background_sig
            }


def gaussian_spheres(datapoints=200000):  # trivial case

    nPoints_signal = datapoints // 2  # half the points (whole number) are signal
    nPoints_background = datapoints - nPoints_signal  # other half are background

    # declare empty arrays for signal and background
    signal_x = []
    signal_y = []
    signal_z = []
    signal_sig = []  # will be a bunch of 1s

    background_x = []
    background_y = []
    background_z = []
    background_sig = []  # will be a bunch of -1s

    for point in range(nPoints_signal):
        x = gaus_rndm(1, 1)
        y = gaus_rndm(1, 1)
        z = gaus_rndm(1, 1)

        signal_x.append(x)
        signal_y.append(y)
        signal_z.append(z)
        signal_sig.append(1)

    for point in range(nPoints_background):
        rndm_val = uniform_rndm(0, 5)

        x = gaus_rndm(-1, 1)
        y = gaus_rndm(-1, 1)
        z = gaus_rndm(-1, 1)

        background_x.append(x)
        background_y.append(y)
        background_z.append(z)
        background_sig.append(-1)

    signal_vars = [signal_x, signal_y, signal_z]
    background_vars = [background_x, background_y, background_z]

    return {'signal_vars': signal_vars,
            'signal_sig': signal_sig,
            'background_vars': background_vars,
            'background_sig': background_sig
            }


def line(datapoints=200000):

    nPoints_signal = datapoints // 2
    nPoints_background = int(datapoints - nPoints_signal)  # other half are background

    # declare empty arrays for signal and background
    signal_x = []
    signal_y = []
    signal_sig = []  # will be a bunch of 1s

    background_x = []
    background_y = []
    background_sig = []  # will be a bunch of -1s

    for point in range(nPoints_signal):
        signal_x.append(uniform_rndm(3.0, 5.0))  # get the x value of the point
        signal_y.append(uniform_rndm(0, 2.0))  # get the y value of the point
        signal_sig.append(1)

    for point in range(nPoints_background):
        x = uniform_rndm(0, 5.0)
        background_x.append(x)  # get the x value of the point
        background_y.append(x + gaus_rndm(0, 0.3))  # get the y value of the point
        background_sig.append(-1)

    signal_vars = [signal_x, signal_y]
    background_vars = [background_x, background_y]

    return {'signal_vars': signal_vars,
            'signal_sig': signal_sig,
            'background_vars': background_vars,
            'background_sig': background_sig
            }


datasets = {
    'moons': {'function': moons,
              'name': 'moons',
              'nvar': 2},
    'enclosed_circles': {'function': enclosed_circles,
                         'name': 'enclosed_circles',
                         'nvar': 2},
    'wedges': {'function': wedges,
               'name': 'wedges',
               'nvar': 2},
    'circle_portions': {'function': circle_portions,
                        'name': 'circles_portions',
                        'nvar': 2},
    'gaussian_circles': {'function': gaussian_circles,
                         'name': 'gaussian_circles',
                         'nvar': 2},
    'gaussian_spheres': {'function': gaussian_spheres,
                         'name': 'gaussian_spheres',
                         'nvar': 3},
    'line': {'function': line,
             'name': 'line',
             'nvar': 2}
}
