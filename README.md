
# my_pipeline 
This Python script analyzes the astronomical spectrum of 2MASS J13004255+1912354 observed by ESO-VLT-U2.
# Features
* Extracts wavelength and flux data from the spectrum.txt file.
* Fits a quadratic continuum and a Gaussian model
*  Provides visualizations of the spectrum, Continuum, and Compound fits.
* Prints the parameters and their errors of the Continuum and Gaussian fits.
# Requirements
Install **necessary** libraries via:
> pip install numpy scipy matplotlib
# How to run
Run the script by passing the spectrum file as an argument:
> python my_pipeline.py spectrum.txt
# Outputs
* **Fit Parameters**: Prints the fitted continuum and Gaussian parameters along with their error.
* **Plots**: Generates plots of the spectrum, Continuum fit, region of interest, and Compound fit.