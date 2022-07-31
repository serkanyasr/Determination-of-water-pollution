# Determination of water pollution

To find the closest distance of the detected edge to the reference point with a certain rotation. Within the determined edge range, it can be determined whether the part is defective or not.

## The following steps were followed as headings in the program:
<br>

- Taking photographs of the solution created by knowing the amount of chlorine in the chlorine-water solution prepared in the laboratory environment so that it is not affected by light. Increasing the amount of chlorine (mg) linearly (0.1, 0.2, 0.3) and observing the changes in color change and reading the R, G, B values with the help of the program.

- To train photos with known B,G,R values according to the amount of chlorine added with Linear Regression under the name of Machine learning. The purpose of training is to read the B,G,R value of a solution with unknown chlorine content and use it to estimate the chlorine content. 

<br>
<br>


1- 5 images in shades of red were used. One white image with RGB (255,255,255) values was used as a reference.

2- Each picture was uploaded to the program separately and renewed in 800x800 dimensions. The aim is to read the correct value by making each image the same size

3- In order to minimize the margin of error in the read B,G,R" value, a square area of 200x200 was taken with reference to the midpoints of the uploaded images. By selecting a certain area, it gets rid of the noise in the video in the instant reading process.

4- In order to minimize the margin of error, the average values of B, G, R values within the square area were taken and the noise was reduced by performing various morphological operations. The purpose of this process is the pixel-sized black dots that we call noise in the image. By averaging and morphological operations, these points are minimized. 
<br><br><br>
![Resim3](https://user-images.githubusercontent.com/80819652/182040256-81078193-1d5d-41ae-8991-4d2286efa89c.png)

![Figure_1](https://user-images.githubusercontent.com/80819652/182040290-85b1ba80-c4f6-4e9f-a9cb-bdb49f78b4e4.png)


<br><br><br>
![Figure_2](https://user-images.githubusercontent.com/80819652/182040325-3665bea3-06cf-4fe2-a640-a3f6a751bea3.png)


<br><br><br>
![.](https://user-images.githubusercontent.com/80819652/182040368-5e5c5ae1-0efd-4bc5-b4a9-1ac60c4adf74.PNG)

<br><br><br>


![.](https://user-images.githubusercontent.com/80819652/182040387-23124099-6bd5-4f7f-b994-07097f4638b8.png)
