#! /usr/bin/env python
# ==========================================================================
# Display Instrument Response Function
#
# Copyright (C) 2017-2018 Juergen Knoedlseder
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ==========================================================================
import sys
import math
import gammalib
import cscripts
try:
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from matplotlib.colors import LogNorm
    plt.figure()
    plt.close()
except (ImportError, RuntimeError):
    print('This script needs the "matplotlib" module')
    sys.exit()


# ========= #
# Plot Aeff #
# ========= #
def plot_aeff(sub, aeff, emin=None, emax=None, tmin=None, tmax=None,
              nengs=100, nthetas=100):
    """
    Plot effective area

    Parameters
    ----------
    sub : figure
        Subplot
    aeff : `~gammalib.GCTAAeff2d`
        Instrument Response Function
    emin : float, optional
        Minimum energy (TeV)
    emax : float, optional
        Maximum energy (TeV)
    tmin : float, optional
        Minimum offset angle (deg)
    tmax : float, optional
        Maximum offset angle (deg)
    nengs : int, optional
        Number of energies
    nthetas : int, optional
        Number of offset angles
    """
    # Determine energy range
    ieng = aeff.table().axis('ENERG')
    neng = aeff.table().axis_bins(ieng)
    if emin == None:
        emin = aeff.table().axis_lo(ieng, 0)
    if emax == None:
        emax = aeff.table().axis_hi(ieng, neng-1)

    # Determine offset angle range
    itheta = aeff.table().axis('THETA')
    ntheta = aeff.table().axis_bins(itheta)
    if tmin == None:
        tmin = aeff.table().axis_lo(itheta, 0)
    if tmax == None:
        tmax = aeff.table().axis_hi(itheta, ntheta-1)

    # Use log energies
    emin = math.log10(emin)
    emax = math.log10(emax)

    # Set axes
    denergy     = (emax - emin)/(nengs-1)
    dtheta      = (tmax - tmin)/(nthetas-1)
    logenergies = [emin+i*denergy for i in range(nengs)]
    thetas      = [tmax-i*dtheta  for i in range(nthetas)]

    # Initialise image
    image = []

    # Loop over offset angles
    for theta in thetas:

        # Initialise row
        row = []

        # Loop over energies
        for logenergy in logenergies:

            # Get effective area value
            value = aeff(logenergy, theta*gammalib.deg2rad)

            # Append value
            row.append(value)

        # Append row
        image.append(row)

    # Plot image
    c    = sub.imshow(image, extent=[emin,emax,tmin,tmax], aspect=0.5, norm=LogNorm(vmin=1e6, vmax=5e10))
    cbar = plt.colorbar(c, orientation='horizontal', shrink=0.8)
    cbar.set_label('cm$^2$')

    # Show boundary contours
    contours = sub.contour(logenergies, thetas, image, [1e7,1e8,1e9], colors=('white'))
    sub.clabel(contours, inline=1, fontsize=8, fmt="%1.1e")

    # Plot title and axis
    sub.set_title('Effective area')
    sub.set_xlabel('log10(E/TeV)')
    sub.set_ylabel('Offset angle (deg)')

    # Return
    return


# ======== #
# Plot PSF #
# ======== #
def plot_psf(sub, psf, emin=None, emax=None, tmin=None, tmax=None,
             nengs=100, nthetas=100):
    """
    Plot Point Spread Function

    Parameters
    ----------
    sub : figure
        Subplot
    psf : `~gammalib.GCTAPsf2D`
        Instrument Response Function
    emin : float, optional
        Minimum energy (TeV)
    emax : float, optional
        Maximum energy (TeV)
    tmin : float, optional
        Minimum offset angle (deg)
    tmax : float, optional
        Maximum offset angle (deg)
    nengs : int, optional
        Number of energies
    nthetas : int, optional
        Number of offset angles
    """
    # Determine energy range
    ieng = psf.table().axis('ENERG')
    neng = psf.table().axis_bins(ieng)
    if emin == None:
        emin = psf.table().axis_lo(ieng, 0)
    if emax == None:
        emax = psf.table().axis_hi(ieng, neng-1)

    # Determine offset angle range
    itheta = psf.table().axis('THETA')
    ntheta = psf.table().axis_bins(itheta)
    if tmin == None:
        tmin = psf.table().axis_lo(itheta, 0)
    if tmax == None:
        tmax = psf.table().axis_hi(itheta, ntheta-1)

    # Use log energies
    emin = math.log10(emin)
    emax = math.log10(emax)

    # Set axes
    denergy     = (emax - emin)/(nengs-1)
    dtheta      = (tmax - tmin)/(nthetas-1)
    logenergies = [emin+i*denergy for i in range(nengs)]
    thetas      = [tmax-i*dtheta  for i in range(nthetas)]

    # Initialise image
    image = []

    # Loop over offset angles
    for theta in thetas:

        # Initialise row
        row = []

        # Loop over energies
        for logenergy in logenergies:

            # Get containment radius value
            value = psf.containment_radius(0.68, logenergy, theta*gammalib.deg2rad) * \
                    gammalib.rad2deg * 60.0

            # Append value
            row.append(value)

        # Append row
        image.append(row)

    # Plot image
    c    = sub.imshow(image, extent=[emin,emax,tmin,tmax], aspect=0.5, vmin=0.0, vmax=15.0)
    cbar = plt.colorbar(c, orientation='horizontal', shrink=0.8)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.set_label('arcmin')

    # Show boundary contours
    contours = sub.contour(logenergies, thetas, image, [2,4,6,8,10,12,14], colors=('white'))
    sub.clabel(contours, inline=1, fontsize=8)

    # Plot title and axis
    if psf.classname() == 'GCTAPsfKing':
        sub.set_title('King function PSF 68% containment radius')
    else:
        sub.set_title('Gaussian PSF 68% containment radius')
    sub.set_xlabel('log10(E/TeV)')
    sub.set_ylabel('Offset angle (deg)')

    # Return
    return


# ====================== #
# Plot energy dispersion #
# ====================== #
def plot_edisp(edisp, emin=None, emax=None, tmin=None, tmax=None,
               nengs=100, nthetas=100):
    """
    Plot energy dispersion template

    Parameters
    ----------
    edisp : `~gammalib.GCTAEdisp2D`
        Instrument Response Function
    emin : float, optional
        Minimum energy (TeV)
    emax : float, optional
        Maximum energy (TeV)
    tmin : float, optional
        Minimum offset angle (deg)
    tmax : float, optional
        Maximum offset angle (deg)
    nengs : int, optional
        Number of energies
    nthetas : int, optional
        Number of offset angles
    """
    # Determine energy range
    if edisp.table().has_axis('ENERG'):
        ieng = edisp.table().axis('ENERG')
    else:
        ieng = edisp.table().axis('ETRUE')
    neng = edisp.table().axis_bins(ieng)
    if emin == None:
        emin = edisp.table().axis_lo(ieng, 0)
    if emax == None:
        emax = edisp.table().axis_hi(ieng, neng-1)

    # Determine migration range
    imigra   = edisp.table().axis('MIGRA')
    nmigra   = edisp.table().axis_bins(imigra)
    migramin = edisp.table().axis_lo(imigra, 0)
    migramax = edisp.table().axis_hi(imigra, nmigra-1)

    # Determine offset angle range
    itheta = edisp.table().axis('THETA')
    ntheta = edisp.table().axis_bins(itheta)
    if tmin == None:
        tmin = edisp.table().axis_lo(itheta, 0)
    if tmax == None:
        tmax = edisp.table().axis_hi(itheta, ntheta-1)

    # Use log energies
    emin = math.log10(emin)
    emax = math.log10(emax)

    # Set axes
    denergy     = (emax - emin)/(nengs-1)
    dmigra      = (migramax - migramin)/(nmigra-1)
    dtheta      = (tmax - tmin)/(nthetas-1)
    logenergies = [emin+i*denergy for i in range(nengs)]
    migras      = [migramin+i*dmigra  for i in range(nmigra)]
    thetas      = [tmax-i*dtheta  for i in range(nthetas)]

    # Initialise images
    image_mean = []
    image_std  = []

    # Initialise true and reconstructed energies
    etrue = gammalib.GEnergy()
    ereco = gammalib.GEnergy()

    # Loop over offset angles
    for theta in thetas:

        # Initialise rows
        row_mean = []
        row_std  = []

        # Compute detx and dety
        #detx = theta*gammalib.deg2rad
        #dety = 0.0

        # Loop over energies
        for logenergy in logenergies:

            # Set true energy
            etrue.log10TeV(logenergy)

            # Compute mean migration
            mean = 0.0
            std  = 0.0
            num  = 0.0
            for migra in migras:
                if migra > 0.0:
                    ereco.log10TeV(math.log10(migra) + logenergy)
                    value  = edisp(ereco, etrue, theta*gammalib.deg2rad)
                    mean  += migra * value
                    std   += migra * migra * value
                    num   += value
            if num > 0.0:
                mean /= num
                std  /= num
                arg   = std - mean * mean
                if arg > 0.0:
                    std = math.sqrt(arg)
                else:
                    std = 0.0

            # Append value
            row_mean.append(mean)
            row_std.append(std)

        # Append rows
        image_mean.append(row_mean)
        image_std.append(row_std)

    # First subplot
    f1 = plt.subplot(223)

    # Plot image
    c1    = f1.imshow(image_mean, extent=[emin,emax,tmin,tmax], aspect=0.5, vmin=0.8, vmax=1.5)
    cbar1 = plt.colorbar(c1, orientation='horizontal', shrink=0.8)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar1.locator = tick_locator
    cbar1.set_label('E$_{reco}$ / E$_{true}$')

    # Show boundary contours
    contours = f1.contour(logenergies, thetas, image_mean, [0.9,1.0,1.1,1.2,1.3,1.4], colors=('white'))
    f1.clabel(contours, inline=1, fontsize=8)

    # Plot title and axis
    f1.set_title('Mean of energy dispersion')
    f1.set_xlabel('log10(E/TeV)')
    f1.set_ylabel('Offset angle (deg)')

    # Second subplot
    f2 = plt.subplot(224)

    # Plot image
    c2    = f2.imshow(image_std, extent=[emin,emax,tmin,tmax], aspect=0.5, vmin=0, vmax=0.25)
    cbar2 = plt.colorbar(c2, orientation='horizontal', shrink=0.8)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator
    cbar2.set_label('E$_{reco}$ / E$_{true}$')

    # Show boundary contours
    contours = f2.contour(logenergies, thetas, image_std, [0.05,0.10,0.15,0.20], colors=('white'))
    f2.clabel(contours, inline=1, fontsize=8)

    # Plot title and axis
    f2.set_title('Standard deviation of energy dispersion')
    f2.set_xlabel('log10(E/TeV)')
    f2.set_ylabel('Offset angle (deg)')

    # Return
    return


# =============== #
# Plot Background #
# =============== #
def plot_bkg(sub, bkg, emin=None, emax=None, tmin=None, tmax=None,
             nengs=100, nthetas=100):
    """
    Plot Background template

    Parameters
    ----------
    sub : figure
        Subplot
    bkg : `~gammalib.GCTABackground3D`
        Instrument Response Function
    emin : float, optional
        Minimum energy (TeV)
    emax : float, optional
        Maximum energy (TeV)
    tmin : float, optional
        Minimum offset angle (deg)
    tmax : float, optional
        Maximum offset angle (deg)
    nengs : int, optional
        Number of energies
    nthetas : int, optional
        Number of offset angles
    """
    # Determine energy range
    ieng = bkg.table().axis('ENERG')
    neng = bkg.table().axis_bins(ieng)
    if emin == None:
        emin = bkg.table().axis_lo(ieng, 0)
    if emax == None:
        emax = bkg.table().axis_hi(ieng, neng-1)

    # Determine offset angle range
    if tmin == None:
        tmin = 0.0
    if tmax == None:
        tmax = 6.0

    # Use log energies
    emin = math.log10(emin)
    emax = math.log10(emax)

    # Set axes
    denergy     = (emax - emin)/(nengs-1)
    dtheta      = (tmax - tmin)/(nthetas-1)
    logenergies = [emin+i*denergy for i in range(nengs)]
    thetas      = [tmax-i*dtheta  for i in range(nthetas)]

    # Initialise image
    vmin  = None
    vmax  = None
    image = []

    # Loop over offset angles
    for theta in thetas:

        # Initialise row
        row = []

        # Compute detx and dety
        detx = theta*gammalib.deg2rad
        dety = 0.0

        # Loop over energies
        for logenergy in logenergies:

            # Get containment radius value
            value = bkg(logenergy, detx, dety)

            # Append value
            row.append(value)

            # Set minimum and maximum
            if value > 0.0:
                if vmin == None:
                    vmin = value
                elif value < vmin:
                    vmin = value
                if vmax == None:
                    vmax = value
                elif value > vmax:
                    vmax = value

        # Append row
        image.append(row)

    # Plot image
    c    = sub.imshow(image, extent=[emin,emax,tmin,tmax], aspect=0.5, norm=LogNorm(vmin=1e-12, vmax=1))
    cbar = plt.colorbar(c, orientation='horizontal', shrink=0.8)
    cbar.set_label('s$^{-1}$ MeV$^{-1}$ sr$^{-1}$')

    # Show boundary contours
    contours = sub.contour(logenergies, thetas, image, [1e-10, 1e-8, 1e-6, 1e-4, 1e-2], colors=('white'))
    sub.clabel(contours, inline=1, fontsize=8, fmt="%1.1e")

    # Plot title and axis
    sub.set_title('Background acceptance')
    sub.set_xlabel('log10(E/TeV)')
    sub.set_ylabel('Offset angle (deg)')

    # Return
    return
