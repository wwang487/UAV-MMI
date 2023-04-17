# UAV-MMI
This is the code for:

Cost-effective Drone Monitoring and Evaluating Toolkits for Stream Habitat Health: Development and Application


## Data preparation

### Toolkit I -- Flight Design

***Step (1)*** - Please download an aerial image for your study site, and this orthophoto needs to be orthogonized and georeferenced.

***Step (2)*** - High-elev flight route:

Please delineate flight boundary and water surface boundary from aerial image.

Low-elev flight route:

Please delineate flight boundary, water surface boundary, trees, and buildings from aerial image. 

***Step (3)*** - Please convert all delineations into a projected coordinate system, e.g., UTM. Save the nodes of flight boundary and water surface boundary into a txt file, three columns of these txt files should be:

*OBJECTID, X,  Y*

A sample txt file called Boundary_Point.txt is in the folder ToolkitIInputs/txt/. All of this txt files are suggested to put into one folder.

***Step (4)*** - Please select a home point from orthophoto, this home points should follow two criteria: 1. Stay away (at least 2 m) from metals; 2) Has clear GPS signal, lands under dense canopy or roofs should not be selected. This home point should also be saved into a txt file, three columns of these txt files should be:

*OBJECTID, X,  Y*

And the txt file is suggested to save into the same folder as step (3).

***Step (5)*** - Please buffer buildings and trees, e.g., 5 m, and export the buffered blocked map as a tif file, the suggested pixel resolution is <= 0.1 m/pixel.

### Toolkit II -- Image Data Processing

***Step (1)*** Put the drone collection image for each flight mission into one folder. 

***Step (2)*** Organize total station measurements into a txt file, five columns of this txt file should be:

*OBJECTID, X,  Y, Z, Description*

### Toolkit III -- MMI Stream Habitat Health Assessment
