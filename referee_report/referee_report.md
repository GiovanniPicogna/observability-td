15.02.2023

# Assistant Editor's Comments:

- Please ensure that all textual labels in figures are at least as large as the caption text; any smaller and they become too difficult to read.

  - Done.

- There is an Acknowledgements header in the manuscript but no text. Please either remove the header or add the missing text.

  - Added the missing text.

# Reviewer's Comments:

Thank you very much for this interesting article. The goal is to provide the community with a way to distinguish between photoevaporation or planets causing the origin of transition disks. The paper moves the field significantly forward towards this goal.

## Main Comments

- The following two comments are both related to your temperature treatment which might be main advantage over modeling the problem with a 1D disk with prescribed photoevaporation and vertical prescriptions of the temperature. 
To be more specific, make clear at some point if and why your modeling gives an advantage over taking e.g. the DIAD model plus the Picogna 2019 photoevaporation prescription and only solve a 1D disk with radial dust transport and settled dust.
Very naively, one could imagine that the outcome would be the same as in your model since your simulations are locally isothermal and rely on tabulated temperatures anyway.
In case there is no clear advantage, make sure to mention that you confirm that it could be done with a more simple approach - although it is likely reasonable to test it.

  - The 2 bands studied in this work are ALMA Band 3 (3 mm) and Band 7 (0.85 mm). The contribution to these 2 wavelengths are given predominantly by dust particles with similar dust sizes, although contribution from smaller and bigger dust sizes can play a non-negligible role. Especially for Band 7, the 1D approach can fail in predicting the correct dust emission since, as shown in Figure 3 (second row, which represents the 1 mm dust particles) of the manuscript, the dust is stirred up in the whole domain and in particular close to the gap edge. The 1D approach could on the other hand allow a better handling of dust growth/fragmentation and indeed it has been tested in the literature (see e.g. Garate et al. 2021).

- p 3, r, l. 18ff: I assume the temperature (and thus the scale height) transitions from the X-ray regime to the hydrostatic regime at the sharp drops close to the cavity edge (at column densities of NX=2e22).
How do you measure temperature, is that the one at the midplane or some vertical average?
Please also comment on how a full radiation-hydrodynamic coupling instead of a artificial boundary in regimes could change the temperature profile. Would it be smoothed out?
Could then the inner pressure peak be affected since it is caused by the dT/dr change at the midplane? Is the inner peak relevant for the dust dynamics?
It seems to be and it seems to cause the double-peak shape best visible in the bottom left panel of Fig. 5 (unconvolved).

  - The temperature reported in Fig. 2b is the mid-plane temperature, since it is the most relevant for the settled dust grains. A fully coupled radiation-hydrodynamic model would not change drastically the temperature profile if the system has reached a radiative equilibrium configuration. The temperature prescription adopted in this work has been proven to be consistent with Monte Carlo temperature models (Owen et al. 2010, Picogna et al. 2019). It is true that the temperatures are tabulated, but they are so as a function of the local disk and stellar irradiation properties. The transition between the X-ray heated and the screened region is a physically thin layer which is challenging to spatially solve numerically but it has been previously tested that an increased resolution does not change the global properties of the disk and the wind. The inner pressure peak, having a short radiative equilibrium timescale, would not be affected by a fully coupled radiation hydrodynamic model, and it would not change significantly for an increased resolution. As shown in Figure 3 only the smallest grain size modelled (s = 0.01 cm) can reach the X-ray heated region and potentially be entrained in the disk wind. A fully coupled radiation hydrodynamics model would be better able to address the entrainment of dust of this and smaller sizes that can on short timescales be entrained in the wind flow. However this size has a limited impact on the simulated dust emission continuum in ALMA Band 3 and 7. I overplot the locations where dP/dR=0 on top of Fig. 4 to show the impact of pressure peaks on the dust continuum emission.  

- p 3, r, l. 49ff: The interpretation of the increased dust scale height does not seem to match the temperature plotted in Fig. 2. The temperature (e.g for Rcav=10 au) drops at 8 au but the dust scale height is increased up to 12.5 au. My guess for the apparent mismatch is that this is because Fig. 2 shows midplane temperature. Check this and somehow describe or visualize the relevant temperature (e.g. as a 2D plot)

  - The main driver of the dust distribution is the underlying gas pressure profile. Plotting Figure 3 with the gas pressure instead of the gas density one can see that the pressure peaks at around 12 au (for the 10 au case), where the majority of the dust collects and as a consequence we observe an increased dust scale height. We agree with the referee that it is not clear from the 1D temperature profile at the disk mid-plane. We included this new version of Figure 3 into the manuscript and updated the text.

## Minor

- p 2, l, l 40: give references for DIAD code at first mention, i.e. here and not only later.

  - done

- Eq. 1: delta_gap should have a unit, such as 0.1. According to eq. 1, rho_0 should be the density at R=Rcav, not the one inside of it (it drops to zero). I also assume it only applies to R<Rcav.

  - corrected

- p 2, l, l. 51:  more precisely: 'increasing gas cavity sizes' or 'increasing Rcav'

  - corrected

- p 2, r, l. 33:  Could be a misunderstanding but you initialize the dust after a few tens of orbits for the disk to equilibrize first. But you mention later that the disk is in a quasi-steady state only after a few thousand orbits. If you mean different kinds of equilibirum (vertical vs radial?) or the damping of some modes, be more specific why 10 orbits is long enough for the dust injection (could just be the result of testing).

  - I was indeed referring to a vertically hydrostatic equilibrium in the first sentence (as we wanted to insert the dust particles only after a clear equilibrium state is reached to avoid oscillations during dust vertical settling which would prolong the time necessary to reach an equilibrium dust distribution), while I was referring to a quasi-steady state in few hundreds (rather than few thousand - typo there) for the radial distribution close to the inner cavity edge in the second sentence as here the viscous spreading is competing with the gas removal by the disk wind, which takes place on a longer timescale depending on the strength of the wind (thus the stellar mass and X-ray luminosity) and and the alpha disk viscosity adopted. I rewrote the two sentences making it clearer that I was referring to two different kinds of equilibrium.

- p 3, l, l. 3: When mentioning 'minimum' Stokes number, it should only be valid for cavity edges of 10 au, and higher for larger cavities, right? If true, mention that it is for the 10au cavity in addition to mentioning the disc midplane).

  - corrected

- p 3, l, l. 29:  Please comment and check on whether a large difference is introduced by choosing your minimum size (100 micrometer) as the smallest particle size compared to using a typically assumed monomer size (0.01 to 1 micrometer, Mathis+ 1977). My guess is that it is fine for the considered wavelength range.

- p 3, l, l. 37:  Is A the surface area of the grid cell as seen from atop the disk or from the side? I assume the observer sees a face-on disk, right?

  - Yes, we corrected the text specifying that the grid surface area A is the one seen from atop.

- p 3, r, l. 28:  Language: either remove 'we show' or 'is plotted'

  - corrected

- p 3, r, l. 44:  Mention that the relevant pressure maximum is the one caused by the gas density (in contrast to the one caused by direct irradiation from the star)

  - added

- p 3. r, sect. 3.2.: Can you explain why the dust distributuion for 0.1 cm does not extend all the way to the midplane close to the cavity?

- p 5, fig 3: Mention the time of the simulation since the injection of the dust (relevant to interpret drift)

- p 5, fig 3: I do not insist on it if you are convinced about this being the ideal way to present you results, but I think the figure is hard to read with the two colormaps including the same colors. I suggest using a grayscale for the gas density or some other improvement you seem fit.

- p 6, r, l. 39:  Briefly mention the setup of nazari, e.g. 2D, dust as fluid?/particles?. In contrast to your approach, they seem to have more/infinite supply of dust.

  - added a brief description of their setup.

- p 6, r, l. 50:  language: "... increases much <more?> steeply ..."

  - corrected

- p 6, r, l. 50ff: To me, neither seems to be much steeper. I would only hightlight the minimum spectral index. The comparison further out is biased by boundary conditions. Within the gap you can probably not measure it due to lack of emission, no? If you agree you might consider zooming into the region from -5 to 10 au around the cavity

- p 6, Sec. 4.2:   You might want to highlight that gaps caused by planets could be more leaky than previously assumed (recent Stammler paper)
added

- p 7, fig 5: Mention which line is which from the bottom row
added

- p 8, fig 7: It seems non-sensical that the 10 au line extends to -30 au as measured from the cavity edge. That would be 20 au on the other side?
Yes, I was including part of the other side of the disc. I corrected it by cutting the plot to show only one side.

- Discussion: I guess you are aware, but a comparison to a more massive planet with spectral indices calculated would be preferable than the Nazari paper with the 30 Me mass planet only. I think qualitatively the results would be the same but you might mention that as suggestion for future works.
Yes, this is a very valid point. I reformulated a sentence in Sec. 4.2 to emphasize it.

- Discussion/Conclusion: The obvious follow-up to your discussion is whether observed spectral indices would match rather the prediction of ~2.2 (photoevaporative cavity) or 2.0 (planet-induced cavity). I found for GM auriga values above 2.2 (Huang et al. 2020) but this might be resolution limited. Check further references.

- Discussion/Conclusion:  There should be further dependencies which should be addressed such as varying alpha, including dust evolution, 3D modeling (more for the planetary case) etc. I suggest to mention that together with the consistent modeling of the planet vs photoevaporation hypothesis.
