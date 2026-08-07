# LigHTS full repository

This repository contains the environments and scripts used to extract and analyze all data supporting the claims of
the manuscript "LigHTS: Massively Parallel Biomimetic Photo-Functionalization for Imaging-Based Ultra-High-Throughput Screening"



# Repository content and organization

This repository contains seven folders, each corresponding to a pipeline to analyze different types of datasets. Each folder contains a README file with instructions for installing the environment and running the Python scripts. In particular:



## \- [flatgel\_analyzer](flatgel_analyzer)

Contains an environment and a Python pipeline optimized to extract parameters of height and flatness from confocal z-stacks of isotropic hydrogels coated with fluorescent fiducial markers

## \- [cell\_area](cell_area)

Contains an environment and a Python pipeline optimized to extract cell area using the tubulin and the actin channels.

## \- [groove\_analyzer](groove_analyzer)

Contains an environment and a Python pipeline optimized to extract height and surface geometry parameters (period and depth) from confocal z-stacks of anisotropic hydrogels coated with fluorescent fiducial markers.

## \- [cell\_migration](cell_migration)

Contains an environment and a Python pipeline optimized to segment and track via CellPose and LapTrack timeseries of migrating cells, and extract parameters to quantify their morphology, migration speed, and directionality.

## \- [holographic\_reconstruction](holographic_reconstruction)

Contains an environment and a Python pipeline optimized to extract geometrical surface parameters from holographic phase images.

## \- [uv-mask-diffraction-model](uv-mask-diffraction-model)

Contains an environment and a Python script to generate the diffraction patterns in figure 5b

## \- [nanoindentation\_analyzer](nanoindentation_analyzer)

Contains an environment and a Python suite to extract apparent reduced indentation moduli from Optics11 Chiaro spherical nanoindentation of GelMA hydrogels, producing per-gel and per-condition summaries, statistics, and figures.

