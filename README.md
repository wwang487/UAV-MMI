# UAV-MMI
This is the code for:

Cost-effective Drone Monitoring and Evaluating Toolkits for Stream Habitat Health: Development and Application


## Data preparation

### Toolkit I -- Flight design

Step (1) - Please download an aerial image for your study site, and this orthophoto needs to be orthogonized and georeferenced.

Step (2) - High-elev flight route:

Please delineate flight boundary and water surface boundary from aerial image.

Low-elev flight route:

Please delineate flight boundary, water surface boundary, trees, and buildings from aerial image. 

Step (3) - Please convert all delineations into a projected coordinate system, e.g., UTM. Save the nodes of flight boundary and water surface boundary into a txt file, three columns of these txt files should be:

OBJECTID, X,  Y

A sample txt file called Boundary_Point.txt is in the folder ToolkitIInputs/txt/

Step (4) - Please buffer buildings and trees, e.g., 5 m, 
