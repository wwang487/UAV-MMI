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

An example delineation looks like this:

![image](https://github.com/wwang487/UAV-MMI/blob/main/Examples/Aerial_Img_Example.jpg)

***Step (3)*** - Please convert all delineations into a projected coordinate system, e.g., UTM. Save the nodes of flight boundary and water surface boundary into a txt file, three columns of these txt files should be:

*OBJECTID, X,  Y*

An example boundary point txt file looks like this:

![image](https://github.com/wwang487/UAV-MMI/blob/main/Examples/Example_txt_0.png)

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

***Step (1)*** Project the orthophoto and terrain map (DEM) into a projected coordinate system.

***Step (2)*** Delineate the polylines of stream bank, eroded zones, vegetated buffer, and tall vegetation from orthophotos. If there are in-stream islands, also delineate polylines of in-stream islands. Save the vertices of these polylines as new point features, and save the coordinates of each point feature into one txt file. If all processes are done in ArcMap, the txt file should have five columns.  

*FID, Id, ORIG_FID, X, Y*
An example vertices coordinate txt file looks like this:

![image](https://github.com/wwang487/UAV-MMI/blob/main/Examples/Example_txt_1.png)

### In-situ Validation

***Step (1)*** Organize the transect-based measurements into a txt file. The txt file should be formed like the follows:

