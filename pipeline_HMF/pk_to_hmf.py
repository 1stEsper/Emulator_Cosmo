import numpy as np
from scipy.integrate import simpson
from symbolic_pofk import linear_new, syren_baryon
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy.interpolate import interp1d


#Convert mass to Lagrangian radius (R)
def mass_to_radius(M, rho_m):
    return (3 * M / (4 * np.pi * rho_m)) ** (1/3)  #*500 * rho_c 

#Compute sigma(M) from P(k)ç
def sigma_vals_from_pk(k, pk, M_vals, rho_c):
    R = mass_to_radius(M_vals, rho_c)
    k2 = k**2
    # Vectorized integration for speed & stability
    kR = np.outer(k, R)              
    W = np.ones_like(kR)
    mask = kR != 0
    W[mask] = 3 * (np.sin(kR[mask]) - kR[mask] * np.cos(kR[mask])) / (kR[mask] ** 3)
    integrand = (pk[:, None]) * (W**2) * (k2[:, None]) 
    sig2 = simpson(integrand, k, axis=0) / (2*np.pi**2)
    return np.sqrt(sig2)

# Tinker08 at z=0
delta_tab = np.array([200, 300, 400, 600, 800, 1200, 1600])
A_tab     = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260])
a_tab     = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30])
b_tab     = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46])
c_tab     = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97])

def get_tinker_params_from_delta(delta_m):
    A = interp1d(delta_tab, A_tab, kind='linear', fill_value="extrapolate")(delta_m)
    a = interp1d(delta_tab, a_tab, kind='linear', fill_value="extrapolate")(delta_m)
    b = interp1d(delta_tab, b_tab, kind='linear', fill_value="extrapolate")(delta_m)
    c = interp1d(delta_tab, c_tab, kind='linear', fill_value="extrapolate")(delta_m)
    return A, a, b, c

#f_sigma for Tinker08 at delta_c=500c
def f_sigma_tinker500c(sigma_vals, z, omega_m0, omega_lambda, delta_c=500):
    # Calculate Omega_m(z)
    Ez2 = omega_m0 * (1 + z)**3 + omega_lambda
    omega_m_z = omega_m0 * (1 + z)**3 / Ez2
    delta_m = delta_c / omega_m_z

    A, a, b, c = get_tinker_params_from_delta(delta_m)
    sigma_vals = np.array(sigma_vals)

    return A * ((sigma_vals / b)**(-a) + 1.0) * np.exp(-c / sigma_vals**2)


#Compute HMF from sigma(M)
# def hmf_from_sigma(M_vals, sigma_vals, rho_m, z=0.0):
#     lnM = np.log(M_vals)
#     lns = np.log(sigma_vals)
#     dlns_dlnM = np.gradient(lns, lnM)

#     #multiplicity from Colossus for 500c
#     f_sigma = f_sigma_tinker500c(sigma_vals, z)
#     # f_sigma = mass_function.massFunction(M_vals, z, mdef='500c', model='tinker08',
#     #                                  q_in='M', q_out='f')
#     #HMF formula
#     return f_sigma * (rho_m / M_vals) * np.abs(dlns_dlnM)

def full_pipeline_hmf(As, Om, Ob, h, ns, w0, wa,
                      A_SN1, A_SN2, A_AGN1, A_AGN2, z=0.0,
                      model='Tinker08', hydro_model='IllustrisTNG',
                      M_vals=None, baryon_effect=None):
    #Default M range
    if M_vals is None:
        M_vals = np.logspace(12, 15, 200)
    #Scale factor
    a = 1.0 / (1.0 + z)

    #Compute linear P(k) via emulator
    k = np.logspace(-3, 2, 1000)
    pk_lin = linear_new.plin_new_emulated(
        k, As, Om, Ob, h, ns,
        mnu=0.0, w0=w0, wa=wa, a=a
    )

    # Compute sigma8 for baryon model
    sigma8 = linear_new.As_to_sigma8(
        As, Om, Ob, h, ns,
        mnu=0.0, w0=w0, wa=wa
    )
    if baryon_effect: 
    #Baryon suppression factor
        S = syren_baryon.S_hydro(
            k, Om, sigma8,
            A_SN1, A_SN2, A_AGN1, A_AGN2,
            a, hydro_model=hydro_model
        )
        pk = pk_lin * S
    
    else: 
        pk = pk_lin


    #Add cosmology to Colossus to get mean density
    cosmo_name = 'fullPipe'
    #Here, the dictionary of cosmological parameters.
    #Note that `H0` in Colossus expects the full Hubble constant in km/s/Mpc (h*100),
    #and include `sigma8` and `ns` so that Colossus uses the same power normalization and tilt as the pipeline.
    cosmo_params = {
        'flat': True,
        'Om0': Om,
        'Ob0': Ob,
        'H0': h * 100.0,
        'sigma8': sigma8,
        'ns': ns
    }
    try:
        cosmology.addCosmology(cosmo_name, **cosmo_params)
    except ValueError:
        cosmology.removeCosmology(cosmo_name)
        cosmology.addCosmology(cosmo_name, **cosmo_params)
    cosmo = cosmology.setCosmology(cosmo_name)
    #Mean matter density at redshift z 
    rho_m = cosmo.rho_m(z) * 1e9 
    # rho_c = cosmo.rho_c(z) * 1e9 
    #Compute sigma(M) and HMF
    sigma_vals = sigma_vals_from_pk(k, pk, M_vals, rho_m)

    f_sigma = f_sigma_tinker500c(sigma_vals, z, omega_m0=Om, omega_lambda= 1 -Om, delta_c=500)

    lnM  = np.log(M_vals)
    lns  = np.log(sigma_vals)
    dlns_dlnM = np.gradient(lns, lnM)

    hmf = f_sigma * (rho_m / M_vals) * np.abs(dlns_dlnM)
    


    return hmf, sigma_vals, sigma8
