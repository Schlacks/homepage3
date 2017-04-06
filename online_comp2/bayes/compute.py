import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm, t
from scipy.stats import gamma, invgamma
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from matplotlib import cm
import os,time,glob


# This class implements the data generator. It takes the true mean and true sigma of the generative likelihood as arguments.

class gauss_bandit:
    def __init__(self, truemu, truesigma):
        self.truemu = truemu
        self.truesigma = truesigma
        self.mu = np.random.normal(self.truemu, self.truesigma, 1)[0]
        self.alpha = 1
        self.beta = 1
        self.n_null = 1

    # The pull script generates data from the true normal distribution. It takes the amount of data requested as argument.

    def pull(self, n):
        return np.random.normal(self.truemu, self.truesigma, n)

    # The update script is the heart of Bayes' Magic. The Hyper Parameters that give rise to the Normal-Inverse-Gamma Distribution
    # are being updated, perfectly exploiting the data available. Here, the normal distribution for the mean and the inverse
    # gamma distribution for the variance is treated separately.
    # The arguments it takes is the data generated from the 'unknown' distribution from the .pull script. More information is
    # provided at https://en.wikipedia.org/wiki/Conjugate_prior

    def update(self, x):
        self.alpha = self.alpha + x.size / 2
        self.beta = self.beta + 0.5 * ((x - np.average(x)) ** 2).sum() + (
        x.size * self.n_null / (2 * (x.size * self.n_null)) * (np.average(x) - self.mu) ** 2)
        self.mu = (self.n_null * self.mu + x.size * np.average(x)) / (self.n_null + x.size)
        self.n_null = self.n_null + x.size

    # The predict script provides the probability of attaining values within a specified interval using all information, received so far.
    # The arguments it takes are the limits of the interval.

    def predict(self, x1, x2):
        p1 = t.cdf((x1 - self.mu) / sqrt((self.beta * (self.n_null + 1)) / (self.alpha * self.n_null)), 2 * self.alpha)
        p2 = t.cdf((x2 - self.mu) / sqrt((self.beta * (self.n_null + 1)) / (self.alpha * self.n_null)), 2 * self.alpha)
        pges = round(100 * abs(p1 - p2), 3)
        #x = np.linspace(t.ppf(0.001, 2*self.alpha,loc=self.mu),t.ppf(0.999, 2*self.alpha,loc=self.mu), 100)
        x = np.linspace(self.mu-(t.ppf(0.99, 2*self.alpha,0))*sqrt(self.beta*(self.n_null+1)/(self.n_null*self.alpha)),self.mu+(t.ppf(0.99, 2*self.alpha,0))*sqrt(self.beta*(self.n_null+1)/(self.n_null*self.alpha)), 100)
        fig, ax = plt.subplots(1, 1)
        fillrange=np.linspace(x1,x2,100)
        ax.fill_between(fillrange, t.pdf(fillrange, df=2*self.alpha,loc=self.mu, scale=sqrt(self.beta*(self.n_null+1)/(self.n_null*self.alpha))),y2=0)
        ax.plot(x, t.pdf(x, df=2*self.alpha,loc=self.mu, scale=sqrt(self.beta*(self.n_null+1)/(self.n_null*self.alpha))))
        prob_statement=('The probability to obtain an outcome between '+ str(x1) + ' and '+ str(x2) + ' with the current knowledge is: '+ str(pges) +
              '%')
        if not os.path.isdir('static'):
            os.mkdir('static')
        else:
            # Remove old plot files
            for filename in glob.glob(os.path.join('static', '*.png')):
                os.remove(filename)
        # Use time since Jan 1, 1970 in filename in order make
        # a unique filename that the browser has not chached
        plotfile = os.path.join('static', str(time.time()) + '.png')
        plt.savefig(plotfile)
        return plotfile,prob_statement

    def first_img(self):
        X = np.linspace(self.mu - 3 * sqrt((self.beta / (self.alpha + 1)) / self.n_null),
                        self.mu + 3 * sqrt((self.beta / (self.alpha + 1)) / self.n_null), 500)
        Y = np.linspace(min(0, 0.5 * self.beta / (self.alpha + 1)), 2 * self.beta / (self.alpha + 1), 500)
        X, Y = np.meshgrid(X, Y)
        Z = norm.pdf(X, self.mu, np.sqrt(Y / self.n_null)) * invgamma.pdf(Y, self.alpha, scale=self.beta)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.4, linewidth=0)
        cset = ax.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=1)
        cset = ax.contour(X, Y, Z, zdir='x', offset=self.mu + 3 * sqrt((self.beta / (self.alpha + 1)) / self.n_null), cmap=cm.coolwarm,
                          alpha=1)
        cset = ax.contour(X, Y, Z, zdir='y', offset=2 * self.beta / (self.alpha + 1), cmap=cm.coolwarm, alpha=1)
        ax.set_xlabel('mu')
        ax.set_ylabel('Sigma^2')
        ax.set_zlabel('Density')
        ax.text2D(-0.11, 0.95, "Parameter Estimate at Peak (true value): \n mu= %.2f (%.2f) \n sigma**2= %.2f (%.2f)" % (self.mu, self.truemu, self.beta / (self.alpha + 1),(self.truesigma)**2),
                  transform=ax.transAxes)
        ax.text2D(-0.11, 0.90, 'Number of total trials: %s' % (self.n_null-1), transform=ax.transAxes)
        # if not os.path.isdir('static/NIGdist'):
        #
        #     os.mkdir('static/NIGdist')
        # else:
        #     # Remove old plot files
        #     for filename in glob.glob(os.path.join('static/NIGdist', '*.png')):
        #         os.remove(filename)
        # Use time since Jan 1, 1970 in filename in order make
        # a unique filename that the browser has not chached
        plotfile = os.path.join('static', str(time.time()) + '.png')
        plt.savefig(plotfile)
        return plotfile

# The pull_and_update function executes scripts of an initiated bandit. It takes the initiated bandit as argument and the amount
# data wished and executes .pull and .update script. It plots the resulting Normal-Inverse-Gamma distribution in a 3D plot with
# contour plots projected on three sides.

def pull_and_update(b, n):
    outcome = b.pull(n)
    b.update(outcome)
    X = np.linspace(b.mu - 3 * sqrt((b.beta / (b.alpha - 1)) / b.n_null),
                    b.mu + 3 * sqrt((b.beta / (b.alpha - 1)) / b.n_null), 500)
    Y = np.linspace(min(0, 0.5 * b.beta / (b.alpha + 1)), 2 * b.beta / (b.alpha + 1), 500)
    X, Y = np.meshgrid(X, Y)
    Z = norm.pdf(X, b.mu, np.sqrt(Y / b.n_null)) * invgamma.pdf(Y, b.alpha, scale=b.beta)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.4, linewidth=0)
    cset = ax.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=1)
    cset = ax.contour(X, Y, Z, zdir='x', offset=b.mu + 3 * sqrt((b.beta / (b.alpha - 1)) / b.n_null), cmap=cm.coolwarm,
                      alpha=1)
    cset = ax.contour(X, Y, Z, zdir='y', offset=2 * b.beta / (b.alpha + 1), cmap=cm.coolwarm, alpha=1)
    ax.set_xlabel('mu')
    ax.set_ylabel('Sigma^2')
    ax.set_zlabel('Density')
    ax.text2D(-0.11, 0.95, "Parameter Estimate at Peak (true value): \n mu= %.2f (%.2f) \n sigma**2= %.2f (%.2f)" % (b.mu, b.truemu, b.beta / (b.alpha + 1),(b.truesigma)**2),
              transform=ax.transAxes)
    ax.text2D(-0.11, 0.90, 'Number of total trials: %s' % (b.n_null-1), transform=ax.transAxes)
    #ax.set_title('Normal-inverse-gamma distribution of the Parameters mu and sigma^2')

    for filename in glob.glob(os.path.join('static', '*.png')):
        os.remove(filename)
    # Use time since Jan 1, 1970 in filename in order make
    # a unique filename that the browser has not chached
    plotfile = os.path.join('static', str(time.time()) + '.png')
    plt.savefig(plotfile)
    return plotfile, outcome
    #plt.show()
