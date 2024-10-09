#Estephany Barba Matta (Student number:24213428) Programming in Python assesment
#06/10/2024

# Import necessary libraries
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse

# Command-line argument parsing
def parse_arguments():
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("spectrum", type=str, help="Path to the spectrum data file.")
    return parser.parse_args()

def extract_spectrum_data(spectrum):
    """
    Function to extract wavelength and flux data from spectrum.txt.
    """
    wavelengths = []
    fluxes = []
    
    with open(spectrum, 'r') as file:  # Open file in read mode
        reader = csv.reader(file)
        # Skip the first 27 lines
        for _ in range(27):
            next(reader)

        # Read the remaining lines and extract data
        for row in reader:
            wavelength, flux = row[0].strip(), row[1].strip()  # Extract wavelength and flux from the row
            wavelengths.append(float(wavelength))
            fluxes.append(float(flux))

    # Convert lists to numpy arrays
    wavelengths = np.array(wavelengths)
    fluxes = np.array(fluxes)

    return wavelengths, fluxes

def continuum_line(wavelengths, a, b, c):
    """
    Continuum curve fitting function.
    """
    return a * wavelengths**2 + b * wavelengths + c

def colour_roi(wavelengths, wmin, wmax, in_roi='red', out_roi='blue'):
    """
    Function to assign colors based on whether wavelengths fall within the Region of interest (ROI).
    """
    roi_mask = (wavelengths >= wmin) & (wavelengths < wmax)  # Boolean mask for ROI
    roi_mask = np.array(roi_mask, dtype=int)  # Convert boolean mask to integer (1 for True, 0 for False)
    colourmap = np.array([out_roi, in_roi])  # Colour map for in and out of ROI
    return colourmap[roi_mask]

def filter_data_in_roi(wavelengths, fluxes, wmin, wmax):
    """
    Filter wavelengths and fluxes to get data within the Region of Interest (ROI).
    """
    roi_start = np.searchsorted(wavelengths, wmin, side='l')  # Start index in ROI
    roi_end = np.searchsorted(wavelengths, wmax, side='r')    # End index in ROI
    return wavelengths[roi_start:roi_end], fluxes[roi_start:roi_end]

def gaussian(wavelengths, A, mu, sig):
    """
    Gaussian model for fitting the data.
    """
    return A * np.exp(-0.5 * (wavelengths - mu)**2 / sig**2) / np.sqrt(2 * np.pi * sig**2)

def compound_model(wavelengths, a, b, c, A, mu, sig):
    """
    Compound model combining the Continuum and Gaussian fits.
    """
    return continuum_line(wavelengths, a, b, c) + gaussian(wavelengths, A, mu, sig)

def fitting_curve(wavelengths, fluxes, popt_l, f_filtered, w_filtered):
    """
    Fit the compound model to the spectrum data.
    """
    initial_estimates = [*popt_l, np.max(f_filtered), np.mean(w_filtered), np.std(w_filtered)]
    popt_fit, pcov_fit = curve_fit(compound_model, wavelengths, fluxes, p0=initial_estimates)
    errors_fit = np.sqrt(np.diag(pcov_fit))  # Get uncertainties of fitted parameters
    return initial_estimates, popt_fit, errors_fit

def plot_results(wavelengths, fluxes, popt_l, fitted_params, compound_fit, colours, wmin, wmax):
    """
    Function to create plots for the results.
    """
    # Plot 1: Spectrum
    plt.figure(figsize=(20, 12))
    plt.scatter(wavelengths, fluxes, s=5, color='blue',label='Spectrum')
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux (ADU)")
    plt.title("UVES Spectrum of 2MASS J13004255+1912354 Observed by ESO-VLT-U2")  
    plt.legend()
    plt.grid(True)
    plt.figtext(0.5, 0.05, "Figure 1: Spectrum", ha="center", fontsize=12)
    plt.figtext(0.1, 0.02, "Interpretation: This graph represents the spectrum of the celestial object 2MASS J13004255+1912354, observed using the UVES instrument on the ESO-VLT-U2 telescope. The spectrum shows how the object's \nlight intensity changes across different wavelengths.", ha="left", fontsize=11)

    # Plot 2: Continuum line
    plt.figure(figsize=(20, 12))
    plt.scatter(wavelengths, fluxes, color='blue', s=5, label='Spectrum')
    plt.plot(wavelengths, continuum_line(wavelengths, *popt_l), 'r--', label='Continuum Fit')
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux (ADU)")
    plt.title("Continuum Fitting for 2MASS J13004255+1912354 Spectrum")
    plt.legend()
    plt.grid(True)
    plt.figtext(0.5, 0.05, "Figure 2: Continuum curve Fit", ha="center", fontsize=12)
    plt.figtext(0.1, 0.02, "Interpretation: This graph demonstrates the continuum fitting for 2MASS J13004255+1912354.", ha="left", fontsize=11)


    # Plot 3: ROI
    plt.figure(figsize=(20, 12))
    roi_mask = (wavelengths >= wmin) & (wavelengths < wmax)
    plt.scatter(wavelengths[roi_mask], fluxes[roi_mask], color='red', s=5, label='Spectrum (ROI)') #to show ROI mask on the legend
    plt.scatter(wavelengths, fluxes, s=5, c=colours, label='Spectrum')
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux (ADU)")
    plt.title("Region of interest")
    plt.legend()
    plt.grid(True)
    plt.figtext(0.5, 0.05, "Figure 3: Highlighted Region of Interest (ROI) of Spectrum.", ha="center", fontsize=12)
    plt.figtext(0.1, 0.02, "Interpretation: This graph represents the spectrum with the ROI highlighted in red.", ha="left", fontsize=11)


    # Plot 4: Compound and Gaussian
    plt.figure(figsize=(20, 12))
    plt.scatter(wavelengths, fluxes, s=5, color='blue', label='Spectrum')
    plt.plot(wavelengths, compound_fit, 'k-', label='Compound Fit')
    plt.plot(wavelengths, continuum_line(wavelengths, *fitted_params[:3]), 'r--', label='Continuum Fit')
    plt.plot(wavelengths, gaussian(wavelengths, *fitted_params[3:]), 'g--', label='Gaussian Fit')
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux (ADU)")
    plt.title("Compound, Continuum, and Gaussian Fit for 2MASS J13004255+1912354")
    plt.legend()
    plt.grid(True)
    plt.figtext(0.5, 0.05, "Figure 4: Spectrum with Compound, Continuum, and Gaussian Fit.", ha="center", fontsize=12)
    plt.figtext(0.1, 0.02, "Interpretation: This graph represents the Spectrum with a Compound, Continuum, and Gaussian Fit. The Compound Fit represents a linear combination of a fit of the background emission (Continuum Fit) \nand a fit of the emission line peak (Gaussian Fit).", ha="left", fontsize=11)

    # Show plots
    plt.show()

def main(spectrum):
    """
    Main function to perform spectrum analysis and fitting.
    """
    # ROI definition
    wmin, wmax = 6675, 6700
    # Extract data
    wavelengths, fluxes = extract_spectrum_data(spectrum)

    # Fit the continuum line
    popt_l, pcov_l = curve_fit(continuum_line, wavelengths, fluxes)

    # Assign colors based on ROI
    colours = colour_roi(wavelengths, wmin, wmax)

    # Filter data within the ROI
    w_filtered, f_filtered = filter_data_in_roi(wavelengths, fluxes, wmin, wmax)

    # Fit the compound model to data
    initial_estimates, fitted_params, errors = fitting_curve(wavelengths, fluxes, popt_l, f_filtered, w_filtered)
    compound_fit = compound_model(wavelengths, *fitted_params)

    #Print parameters    
    # Separate continuum parameters and compound parameters
    continuum_params = fitted_params[:3]  # a, b, c
    continuum_errors = errors[:3]

    compound_params = fitted_params[3:]  # A, mu, sig
    compound_errors = errors[3:]
    
    # Print Continuum Fit Parameters
    print("Continuum fit parameters:")
    print(f"{'Name':<28} {'Value':<12} {'Error':<9}")
    print('-'* 51)
    continuum_labels = ['a (quadratic coefficient)', 'b (linear coefficient)', 'c (constant)']

    for param, value, error in zip(continuum_labels, continuum_params, continuum_errors):
        print(f"{param:<26} : {value:<12.3f} ±{error:<9.3f}")

    # Print Compound Fit Parameters
    print("\nCompound fit parameters:")
    print(f"{'Name':<28} {'Value':<12} {'Error':<9}")
    print('-' * 51)
    compound_labels = ['A (amplitude)', 'mu (mean)', 'sigma (standard deviation)']

    for param, value, error in zip(compound_labels, compound_params, compound_errors):
        print(f"{param:<26} : {value:<12.3f} ±{error:<9.3f}")
        
    # Plotting the results
    plot_results(wavelengths, fluxes, popt_l, fitted_params, compound_fit, colours, wmin, wmax)

# Main execution block
if __name__ == "__main__":
    args = parse_arguments()
    main(args.spectrum)
