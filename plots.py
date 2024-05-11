import matplotlib.pyplot as plt
import numpy as np
def plotMTFs(sar):
    # fig1, ((axTilt, axHydrodynamic, axRB, axVB), (axTiltPhase, axHydrodynamicPhase, axRBPhase, axVBPhase))=plt.subplots(2,4)
    fig1, ((axTilt, axHydrodynamic), (axRB, axVB))=plt.subplots(2,2)
    axTilt.pcolor(sar.kx, sar.ky, abs(sar.tilt_mtf()))
    axTiltContour=axTilt.contour(sar.kx, sar.ky, abs(sar.tilt_mtf()), colors='w')
    axTilt.clabel(axTiltContour, axTiltContour.levels, inline=True, fontsize=10)
    axTilt.set_title("Tilt MTF")
    axTilt.set_xlabel("Azimuth Wavenumber")
    axTilt.set_ylabel("Range Wavenumber")
    # axTiltPhase.pcolor(sar.kx, sar.ky, np.angle(sar.tilt_mtf()))
    # axTiltPhase.contour(sar.kx, sar.ky, np.angle(sar.tilt_mtf()), colors='w')
    # axTiltPhase.set_title("Tilt MTF")
    # axTiltPhase.set_xlabel("Azimuth Wavenumber")
    # axTiltPhase.set_ylabel("Range Wavenumber")
    axHydrodynamic.pcolor(sar.kx, sar.ky, abs(sar.hydrodynamic_mtf()))
    axHydrodynamicContour=axHydrodynamic.contour(sar.kx, sar.ky, abs(sar.hydrodynamic_mtf()), colors='w')
    axHydrodynamic.clabel(axHydrodynamicContour, axHydrodynamicContour.levels, inline=True, fontsize=10)
    axHydrodynamic.set_title("Hydrodynamic MTF")
    axHydrodynamic.set_xlabel("Azimuth Wavenumber")
    axHydrodynamic.set_ylabel("Range Wavenumber")
    # axHydrodynamicPhase.pcolor(sar.kx, sar.ky, np.angle(sar.hydrodynamic_mtf()))
    # axHydrodynamicPhase.contour(sar.kx, sar.ky, np.angle(sar.hydrodynamic_mtf()), colors='w')
    # axHydrodynamicPhase.set_title("Tilt MTF")
    # axHydrodynamicPhase.set_xlabel("Azimuth Wavenumber")
    # axHydrodynamicPhase.set_ylabel("Range Wavenumber")
    axRB.pcolor(sar.kx, sar.ky, abs(sar.range_bunching_mtf()))
    axRBContour=axRB.contour(sar.kx, sar.ky, abs(sar.range_bunching_mtf()), colors='w')
    axRB.clabel(axRBContour, axRBContour.levels, inline=True, fontsize=10)
    axRB.set_title("Range Bunching MTF")
    axRB.set_xlabel("Azimuth Wavenumber")
    axRB.set_ylabel("Range Wavenumber")
    # axRBPhase.pcolor(sar.kx, sar.ky, np.angle(sar.range_bunching_mtf()))
    # axRBPhase.contour(sar.kx, sar.ky, np.angle(sar.range_bunching_mtf()), colors='w')
    # axRBPhase.set_title("Tilt MTF")
    # axRBPhase.set_xlabel("Azimuth Wavenumber")
    # axRBPhase.set_ylabel("Range Wavenumber")
    axVB.pcolor(sar.kx, sar.ky, abs(sar.velocity_bunching_mtf()))
    axVBContour=axVB.contour(sar.kx, sar.ky, abs(sar.velocity_bunching_mtf()), colors='w')
    axVB.clabel(axVBContour, axVBContour.levels, inline=True, fontsize=10)
    axVB.set_title("Velociy Bunching MTF")
    axVB.set_xlabel("Azimuth Wavenumber")
    axVB.set_ylabel("Range Wavenumber")
    # axVBPhase.pcolor(sar.kx, sar.ky, np.angle(sar.velocity_bunching_mtf()))
    # axVBPhase.contour(sar.kx, sar.ky, np.angle(sar.velocity_bunching_mtf()), colors='w')
    # axVBPhase.set_title("Tilt MTF")
    # axVBPhase.set_xlabel("Azimuth Wavenumber")
    # axVBPhase.set_ylabel("Range Wavenumber")

    fig2, (axOrbitalMTF, axOrbital)=plt.subplots(1,2)
    axOrbitalMTF.pcolor(sar.kx, sar.ky, abs(sar.orbital_velocity_mtf()), cmap='gist_gray')
    axOrbitalMTF.contour(sar.kx, sar.ky, abs(sar.orbital_velocity_mtf()), colors='w')
    axOrbitalMTF.set_title("Orbital Velocity MTF")
    axOrbitalMTF.set_xlabel("Azimuth Wavenumber")
    axOrbitalMTF.set_ylabel("Range Wavenumber")
    plotColorbar=axOrbital.imshow(sar.mean_orbital_velocity(), origin='lower')
    plt.colorbar(plotColorbar, ax=axOrbital)
    # axOrbital.pcolor(sar.kx, sar.ky, abs(sar.orbital_velocity_mtf()), cmap='gist_gray')
    # axOrbital.contour(sar.kx, sar.ky, abs(sar.orbital_velocity_mtf()), colors='w')
    # axOrbital.set_title("Orbital Velocity MTF")
    # axOrbital.set_xlabel("Azimuth Wavenumber")
    # axOrbital.set_ylabel("Range Wavenumber")

    fig3, (axRARAbs, axRarPhase)=plt.subplots(1,2)
    axRARAbsPlot=axRARAbs.pcolor(sar.kx, sar.ky, abs(sar.RAR_MTF()), cmap='gist_gray')
    absContour=axRARAbs.contour(sar.kx, sar.ky, abs(sar.RAR_MTF()),20, colors='w')
    plt.colorbar(axRARAbsPlot, ax=axRARAbs)
    axRARAbs.clabel(absContour, absContour.levels, inline=True, fontsize=10)
    axRARAbs.set_title("RAR MTF")
    axRARAbs.set_xlabel("Azimuth Wavenumber")
    axRARAbs.set_ylabel("Range Wavenumber")
    axRarPhasePlot=axRarPhase.pcolor(sar.kx, sar.ky, abs(np.angle(sar.RAR_MTF())), cmap='gist_gray')
    phaseContour=axRarPhase.contour(sar.kx, sar.ky, np.rad2deg(abs(np.angle(sar.RAR_MTF()))), 20, colors='w')
    plt.colorbar(axRarPhasePlot, ax=axRarPhase)
    axRarPhase.clabel(phaseContour, phaseContour.levels, inline=True, fontsize=10)
    axRarPhase.set_title("RAR MTF")
    axRarPhase.set_xlabel("Azimuth Wavenumber")
    axRarPhase.set_ylabel("Range Wavenumber")


    # fig4, (sigmaPLot)=plt.subplots(1,1)
    # sigmaPLot.plot((sar.v_covariance(0)[128,:]))
    # sigmaPLot.plot((sar.v_covariance(0)[:,128]))

    # plt.colorbar(sigmaColorbar, ax=sigmaPLot)

def plotModulations(sar):
    fig, (ax)=plt.subplots(1)
    ax.plot(sar.surface[:,sar.N-1], label="surface")
    # ax.plot(sar.NRCS()[0,:], label="NRCS")
    tilt=np.real(np.fft.ifft2(np.fft.ifftshift(sar.tilt_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))
    ax.plot(tilt[:,sar.N-1], label="tilt")
    hydrodynamic=np.real(np.fft.ifft2(np.fft.ifftshift(sar.hydrodynamic_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))
    ax.plot(hydrodynamic[:,sar.N-1], label="hydrodynamic")
    rb=np.real(np.fft.ifft2(np.fft.ifftshift(sar.range_bunching_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))
    ax.plot(rb[:,sar.N-1], label="range bunching")
    # vb=np.real(np.fft.ifft2(np.fft.ifftshift(sar.velocity_bunching_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))
    # ax.plot(vb[:,255], label="velocity bunching")
    plt.legend()

def animate(Z):
    from matplotlib.animation import FuncAnimation
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Initialize imshow plot
    img = ax.imshow(Z[0,:,:], animated=True, origin='lower')

    # Define the update function for animation
    def update(frame):
        # Update data with random values
        img.set_array(Z[frame,:,:])
        return img,

    # Create animation
    ani = FuncAnimation(fig, update, frames=Z.shape[0], interval=200, blit=True)
    plt.show()

def plotSpectras(sar):
    fig, ((ax1,ax2), (ax3, ax4))=plt.subplots(2, 2)

    ax1.contour(sar.kx, sar.ky, abs(sar.PSI))#, extent=(sar.kx.min(),sar.kx.max(),sar.ky.min(),sar.ky.max()))
    ax1.set_title("Original Spectrum")
    ax2.contour(sar.kx, sar.ky,abs(np.fft.fftshift(np.fft.fft2(sar.surface))))#, extent=(sar.kx.min(),sar.kx.max(),sar.ky.min(),sar.ky.max()))
    ax2.set_title("Sea surface Spectrum")
    ax3.contour(sar.kx, sar.ky,abs(np.fft.fftshift(np.fft.fft2(sar.I-np.mean(sar.I)))))#, extent=(sar.kx.min(),sar.kx.max(),sar.ky.min(),sar.ky.max()))
    ax3.set_title("SAR Spectrum")
    ax4.contour(sar.kx, sar.ky,abs(sar.wave_field()))#, extent=(sar.kx.min(),sar.kx.max(),sar.ky.min(),sar.ky.max()))
    ax4.set_title("Inverse SAR")
