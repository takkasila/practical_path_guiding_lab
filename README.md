# Practical Path Guiding for Efficient Light-Transport Simulation
Implementation of ["Practical Path Guiding for Efficient Light-Transport Simulation 2017"](https://jannovak.info/publications/PathGuide/index.html) by Thomas Müller, Markus Gross, and Jan Novák in Mitsuba3. This is done as an exercise in the Computer Graphics Lab, University of Bonn. Please read the full report [lab_report.pdf](lab_report.pdf) for full detail.

![banner](banner/pipeline%20diagram.png)


# Package dependencies
- Mitsuba3
- matplotlib
- pandas
- progressbar


# File description
## `Rendering`
- **main.py** : render a scene using the path guiding algorithm.
- **path_tracing_render.py** : render a scene using path tracing with NEE.
- **repeat_high_spp_renderer.py** : repeatedly render the same scene from main.py using same SD-tree data that has been generated. For averaging performance result.
 ## `Analysis`
- **tree_plotter.py** : visualize quadtree radiance at a given position.
- **performance_plot.py** : plot variance and mean square error against budget. For comparison performance, you will have to run *path_tracing_render.py* and *repeat_high_spp_renderer.py* first.

# Qualititve comparision
![qualitative comparision](banner/qualitative%20comparision.png)