import numpy as np
from scipy.stats import exponnorm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc
from scipy.optimize import minimize
from scipy.stats import skewnorm


def find_tot_counts(pdata):
    tc = 0
    for i in pdata:
        tc+=i
    return tc

def gaussian(A, x, mean, std):
    """
    Gives the value of the Gaussian(A:Amplitude, mean:mean, std:standard_deviation) at x
    """
    return A * (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean)**2) / (2 * std**2))

def emg(A, x, mean, std, decay):
    """
    Gives the value of the EMG(A:Amplitude, mean:mean, std:standard_deviation, decay:decay) at x
    """
    # Ensure decay is positive to prevent issues in fitting
    # decay = np.abs(decay)
    return A * (decay / 2) * np.exp((decay / 2) * (2 * mean + decay * std**2 - 2 * x)) * \
            erfc((mean + decay * std**2 - x) / (np.sqrt(2) * std))

def hist_gaus_fit(pdata):
    """
    Fits the histogram to a Gaussian function and plots 
    """
    # Calculate total counts as the sum of `pdata`
    A = np.sum(pdata)  # Assuming `A` is the total area under the curve
    nbins = pdata.shape[0]
    # Define the Gaussian function with A as a constant
    def gaussian(x, mean, std):
        return A * (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean)**2) / (2 * std**2))

    # Create the x-values based on the bin width and number of bins
    x = np.linspace(0.5, nbins-0.5, nbins)
    y = pdata  # y-values from the data

    # Calculate weighted mean and standard deviation for the initial guess
    mean_weighted = np.sum(x * pdata) / np.sum(pdata)
    std_weighted = np.sqrt(np.sum(pdata * (x - mean_weighted)**2) / np.sum(pdata))

    print(mean_weighted)
    print(std_weighted)

    initial_guess = [mean_weighted, std_weighted]

    # Fit the Gaussian model to the data
    popt, pcov = curve_fit(gaussian, x, y, p0=initial_guess)

    # Plot the fitted curve and the data
    plt.figure()
    plt.plot(x, gaussian(x, *popt), label="Gaussian Fit", color="red")
    plt.plot(x, pdata, label="Data", color="blue", linewidth = 0.5)
    plt.xlabel("Bins")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()

    # Return optimized parameters: mean and std
    return popt

def gaussian_fit_est(pdata):
    """
    Gives the estimated parameters after a fit with a Gaussian function
    """
    A = np.sum(pdata)  # Assuming `A` is the total area under the curve
    nbins = pdata.shape[0]
    # Define the Gaussian function with A as a constant
    def gaussian(x, mean, std):
        return A * (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean)**2) / (2 * std**2))

    # Create the x-values based on the bin width and number of bins
    x = np.linspace(0.5, nbins-0.5, nbins)
    y = pdata  # y-values from the data

    # Calculate weighted mean and standard deviation for the initial guess
    mean_weighted = np.sum(x * pdata) / np.sum(pdata)
    std_weighted = np.sqrt(np.sum(pdata * (x - mean_weighted)**2) / np.sum(pdata))

    print(mean_weighted)
    print(std_weighted)

    initial_guess = [mean_weighted, std_weighted]

    # Fit the Gaussian model to the data
    popt, pcov = curve_fit(gaussian, x, y, p0=initial_guess)

    return popt

def gaus_mle_plot(pdata):

    A = np.sum(pdata)  # Assuming `A` is the total area under the curve
    nbins = pdata.shape[0]

    x = np.linspace(0.5, nbins-0.5, nbins)
    y = pdata

    mean_weighted = np.sum(x * pdata) / np.sum(pdata)
    std_weighted = np.sqrt(np.sum(pdata * (x - mean_weighted)**2) / np.sum(pdata))

    plt.figure()
    plt.plot(x, gaussian(A, x, mean_weighted, std_weighted), label="Gaussian Fit", color="red")
    plt.plot(x, y, label="Data", color="blue", linewidth = 0.5)
    plt.xlabel("Bins")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()

def gaus_mle(pdata):

    nbins = pdata.shape[0]

    x = np.linspace(0.5, nbins-0.5, nbins)
    y = pdata

    mean_weighted = np.sum(x * pdata) / np.sum(pdata)
    std_weighted = np.sqrt(np.sum(pdata * (x - mean_weighted)**2) / np.sum(pdata))

    return mean_weighted, std_weighted  

def hist_emg_fit(pdata):
    """
    Fits the histogram to an EMG function and plots
    """
    # Calculate total counts as the sum of `pdata`
    A = np.sum(pdata)  # Assuming `A` is the total area under the curve
    nbins = pdata.shape[0]
    # Define the EMG function with A as a constant
    def emg(x, mean, std, decay):
        # Ensure decay is positive to prevent issues in fitting
        decay = np.abs(decay)
        return A * (decay / 2) * np.exp((decay / 2) * (2 * mean + decay * std**2 - 2 * x)) * \
               erfc((mean + decay * std**2 - x) / (np.sqrt(2) * std))

    # Create the x-values based on the bin width and number of bins
    x = np.linspace(0.5, nbins-0.5, nbins)
    y = pdata  # y-values from the data

    # Calculate weighted mean and standard deviation for the initial guess
    mean_weighted = np.sum(x * pdata) / np.sum(pdata)
    std_weighted = np.sqrt(np.sum(pdata * (x - mean_weighted)**2) / np.sum(pdata))

    # Initial guess for decay rate (lambda); set a small positive value initially
    decay_initial = 1 / (np.max(x) - np.min(x))

    # Set initial guess as [mean, std, decay]
    initial_guess = [mean_weighted, std_weighted, decay_initial]

    # Fit the EMG model to the data
    popt, pcov = curve_fit(emg, x, y, p0=initial_guess)

    # Plot the fitted curve and the data
    plt.figure()
    plt.plot(x, emg(x, *popt), label="EMG Fit", color="red")
    plt.scatter(x, pdata, s=1, label="Data", color="blue")
    plt.xlabel("Bins")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()

    # Return optimized parameters: mean, std, and decay
    return popt

def emg_fit_est(pdata):
    """
    Gives the estimated parameters after a fit with an EMG function
    """
    # Calculate total counts as the sum of `pdata`
    A = np.sum(pdata)  # Assuming `A` is the total area under the curve
    nbins = pdata.shape[0]
    # Define the EMG function with A as a constant
    def emg(x, mean, std, decay):
        # Ensure decay is positive to prevent issues in fitting
        decay = np.abs(decay)
        return A * (decay / 2) * np.exp((decay / 2) * (2 * mean + decay * std**2 - 2 * x)) * \
               erfc((mean + decay * std**2 - x) / (np.sqrt(2) * std))

    # Create the x-values based on the bin width and number of bins
    x = np.linspace(0.5, nbins-0.5, nbins)
    y = pdata  # y-values from the data

    # Calculate weighted mean and standard deviation for the initial guess
    mean_weighted = np.sum(x * pdata) / np.sum(pdata)
    std_weighted = np.sqrt(np.sum(pdata * (x - mean_weighted)**2) / np.sum(pdata))

    # Initial guess for decay rate (lambda); set a small positive value initially
    decay_initial = 1 / (np.max(x) - np.min(x))

    # Set initial guess as [mean, std, decay]
    initial_guess = [mean_weighted, std_weighted, decay_initial]

    # Fit the EMG model to the data
    popt, pcov = curve_fit(emg, x, y, p0=initial_guess)

    # Return optimized parameters: mean, std, and decay
    return popt

def emg_mle_plot(pdata):
    """
    Gives the MLE estimates for an EMG function and plots
    """
    def emg_pdf(x, mean, std, decay):
        """Exponentially modified Gaussian PDF."""
        decay = np.abs(decay)  # Ensure decay is positive
        return (decay / 2) * np.exp((decay / 2) * (2 * mean + decay * std**2 - 2 * x)) * \
            erfc((mean + decay * std**2 - x) / (np.sqrt(2) * std))

    def negative_log_likelihood(params, x, y):
        """Negative log-likelihood for the EMG function given histogram data."""
        mean, std, decay = params
        # Calculate predicted probabilities for each bin using the EMG PDF
        pdf_vals = emg_pdf(x, mean, std, decay)
        pdf_vals = np.clip(pdf_vals, 1e-10, None)  # Avoid log(0) by setting a minimum threshold
        # Weighted negative log likelihood
        nll = -np.sum(y * np.log(pdf_vals))
        return nll

    def mle_emg_fit(pdata):
        # Data setup
        nbins = pdata.shape[0]
        x = np.linspace(0.5, nbins - 0.5, nbins)
        y = pdata  # y-values from the data

        # Initial guesses for mean, std, and decay
        mean_weighted = np.sum(x * pdata) / np.sum(pdata)
        std_weighted = np.sqrt(np.sum(pdata * (x - mean_weighted)**2) / np.sum(pdata))
        decay_initial = 1 / (np.max(x) - np.min(x))
        initial_guess = [mean_weighted, std_weighted, decay_initial]

        # Perform MLE by minimizing the negative log-likelihood
        result = minimize(negative_log_likelihood, initial_guess, args=(x, y), method="L-BFGS-B",
                        bounds=[(None, None), (1e-5, None), (1e-5, None)])  # Bounds to keep std and decay positive

        # Extract optimized parameters
        mean_mle, std_mle, decay_mle = result.x

        # Plot the fitted EMG and the data
        plt.figure()
        plt.plot(x, emg_pdf(x, mean_mle, std_mle, decay_mle) * np.sum(y), label="EMG MLE Fit", color="red")
        plt.scatter(x, pdata, label="Data", color="blue")
        plt.xlabel("Bins")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()

        # Return optimized parameters: mean, std, and decay
        return mean_mle, std_mle, decay_mle
    
    return mle_emg_fit(pdata)

def skewnormp(A, x, loc, scale, shape):
        """Skew Normal PDF using SciPy's skewnorm."""
        return A * skewnorm.pdf(x, shape, loc=loc, scale=scale)

def skewnorm_mle_plot(pdata):
    """
    Gives the MLE estimates for skewnorm and plots 
    """
    def skewnorm_pdf(x, loc, scale, shape):
        """Skew Normal PDF using SciPy's skewnorm."""
        return skewnorm.pdf(x, shape, loc=loc, scale=scale)

    def negative_log_likelihood(params, x, y):
        """Negative log-likelihood for the skew normal given histogram data."""
        loc, scale, shape = params
        # Calculate predicted probabilities for each bin using the skew normal PDF
        pdf_vals = skewnorm_pdf(x, loc, scale, shape)
        pdf_vals = np.clip(pdf_vals, 1e-10, None)  # Avoid log(0) by setting a minimum threshold
        # Weighted negative log likelihood
        nll = -np.sum(y * np.log(pdf_vals))
        return nll

    def mle_skewnorm_fit(pdata):
        # Data setup
        nbins = pdata.shape[0]
        x = np.linspace(0.5, nbins - 0.5, nbins)
        y = pdata  # y-values from the data

        # Initial guesses for location, scale, and shape (skewness)
        loc_initial = np.sum(x * pdata) / np.sum(pdata)  # Mean estimate
        scale_initial = np.sqrt(np.sum(pdata * (x - loc_initial)**2) / np.sum(pdata))  # Std dev estimate
        shape_initial = 0.0  # Start with no skewness
        initial_guess = [loc_initial, scale_initial, shape_initial]

        # Perform MLE by minimizing the negative log-likelihood
        result = minimize(negative_log_likelihood, initial_guess, args=(x, y), method="L-BFGS-B",
                          bounds=[(None, None), (1e-5, None), (None, None)])  # Bounds to keep scale positive

        # Extract optimized parameters
        loc_mle, scale_mle, shape_mle = result.x

        # Plot the fitted skew normal distribution and the data
        plt.figure()
        plt.plot(x, skewnorm_pdf(x, loc_mle, scale_mle, shape_mle) * np.sum(y), label="Skew Normal MLE Fit", color="red")
        plt.plot(x, pdata, label="Data", color="blue", linewidth=0.5)
        plt.xlabel("Bins")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()

        # Return optimized parameters: loc, scale, and shape
        return loc_mle, scale_mle, shape_mle
    
    return mle_skewnorm_fit(pdata)

def emg_pdf(x, mean, std, decay):
    """Exponentially modified Gaussian PDF."""
    # decay = np.abs(decay)  # Ensure decay is positive
    return (decay / 2) * np.exp((decay / 2) * (2 * mean + decay * std**2 - 2 * x)) * \
        erfc((mean + decay * std**2 - x) / (np.sqrt(2) * std))

def negative_log_likelihood_emg(params, x, y):
    """Negative log-likelihood for the EMG function given histogram data."""
    mean, std, decay = params
    # Calculate predicted probabilities for each bin using the EMG PDF
    pdf_vals = emg_pdf(x, mean, std, decay)
    pdf_vals = np.clip(pdf_vals, 1e-10, None)  # Avoid log(0) by setting a minimum threshold
    # Weighted negative log likelihood
    nll = -np.sum(y * np.log(pdf_vals))
    return nll

def emg_mle(pdata):
    """
    Gives the paramters after MLE estimation with EMG distribution
    """
    # Data setup
    nbins = pdata.shape[0]
    x = np.linspace(0.5, nbins - 0.5, nbins)
    y = pdata  # y-values from the data

    # Initial guesses for mean, std, and decay
    mean_weighted = np.sum(x * pdata) / np.sum(pdata)
    std_weighted = np.sqrt(np.sum(pdata * (x - mean_weighted)**2) / np.sum(pdata))
    decay_initial = 1 / (np.max(x) - np.min(x))
    initial_guess = [mean_weighted, std_weighted, decay_initial]

    # Perform MLE by minimizing the negative log-likelihood
    result = minimize(negative_log_likelihood_emg, initial_guess, args=(x, y), method="L-BFGS-B",
                    bounds=[(None, None), (1e-5, None), (1e-5, None)])  # Bounds to keep std and decay positive

    # Extract optimized parameters
    mean_mle, std_mle, decay_mle = result.x

    # Plot the fitted EMG and the data
    plt.figure()
    plt.plot(x, emg_pdf(x, mean_mle, std_mle, decay_mle) * np.sum(y), label="EMG MLE Fit", color="red")
    plt.scatter(x, pdata, label="Data", color="blue")
    plt.xlabel("Bins")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()

    # Return optimized parameters: mean, std, and decay
    return mean_mle, std_mle, decay_mle

def skewnorm_pdf(x, loc, scale, shape):
        """Skew Normal PDF using SciPy's skewnorm."""
        return skewnorm.pdf(x, shape, loc=loc, scale=scale)

def negative_log_likelihood_sn(params, x, y):
    """Negative log-likelihood for the skew normal given histogram data."""
    loc, scale, shape = params
    # Calculate predicted probabilities for each bin using the skew normal PDF
    pdf_vals = skewnorm_pdf(x, loc, scale, shape)
    pdf_vals = np.clip(pdf_vals, 1e-10, None)  # Avoid log(0) by setting a minimum threshold
    # Weighted negative log likelihood
    nll = -np.sum(y * np.log(pdf_vals))
    return nll

def skewnorm_mle(pdata):
    """
    Gives the parameters after MLE estimation with skewnorm distribution
    """
    # Data setup
    nbins = pdata.shape[0]
    x = np.linspace(0.5, nbins - 0.5, nbins)
    y = pdata  # y-values from the data

    # Initial guesses for location, scale, and shape (skewness)
    loc_initial = np.sum(x * pdata) / np.sum(pdata)  # Mean estimate
    scale_initial = np.sqrt(np.sum(pdata * (x - loc_initial)**2) / np.sum(pdata))  # Std dev estimate
    shape_initial = 0.0  # Start with no skewness
    initial_guess = [loc_initial, scale_initial, shape_initial]

    # Perform MLE by minimizing the negative log-likelihood
    result = minimize(negative_log_likelihood_sn, initial_guess, args=(x, y), method="L-BFGS-B",
                        bounds=[(None, None), (1e-5, None), (None, None)])  # Bounds to keep scale positive

    # Extract optimized parameters
    loc_mle, scale_mle, shape_mle = result.x

    # Plot the fitted skew normal distribution and the data
    # plt.plot(x, skewnorm_pdf(x, loc_mle, scale_mle, shape_mle) * np.sum(y), label="Skew Normal MLE Fit", color="red")
    # plt.plot(x, pdata, label="Data", color="blue", linewidth=0.5)
    # plt.xlabel("Bins")
    # plt.ylabel("Counts")
    # plt.legend()
    # plt.show()

    # Return optimized parameters: loc, scale, and shape
    return loc_mle, scale_mle, shape_mle

def emg_mle(pdata):
    nbins = pdata.shape[0]
    x = np.linspace(0.5, nbins - 0.5, nbins)
    y = pdata  # y-values from the data

    # Initial guesses for mean, std, and decay
    mean_weighted = np.sum(x * pdata) / np.sum(pdata)
    std_weighted = np.sqrt(np.sum(pdata * (x - mean_weighted)**2) / np.sum(pdata))
    decay_initial = 1 / (np.max(x) - np.min(x))
    initial_guess = [mean_weighted, std_weighted, decay_initial]

    # Perform MLE by minimizing the negative log-likelihood
    result = minimize(negative_log_likelihood_emg, initial_guess, args=(x, y), method="L-BFGS-B",
                    bounds=[(None, None), (1e-5, None), (1e-5, None)])  # Bounds to keep std and decay positive

    # Extract optimized parameters
    mean_mle, std_mle, decay_mle = result.x
    return mean_mle, std_mle, decay_mle

def gaussian_mle(pdata):
    nbins = pdata.shape[0]
    x = np.linspace(0.5, nbins - 0.5, nbins)
    y = pdata  # y-values from the data

    # Initial guesses for mean, std, and decay
    mean_weighted = np.sum(x * pdata) / np.sum(pdata)
    std_weighted = np.sqrt(np.sum(pdata * (x - mean_weighted)**2) / np.sum(pdata))
    return mean_weighted, std_weighted

