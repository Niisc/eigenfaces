# Eigenfaces
## Background information
Eigenfaces is a facial recognition technique that involves reducing the dimensionality of a dataset of facial images. For example, given a dataset of 400 images, each 64 pixels by 64 pixels, the dimensionality of the image space can be reduced from 4096 by 4096 to 400 by 4096 by only considering the eigenspace created by the eigenfaces.

### How the code works
TODO
## How to use
### Build instructions linux
Download and install (arrayfire)[https://github.com/arrayfire/arrayfire] on your computer. Make sure to have make, cmake installed. \
In `/Eigenfaces` run the following commands \
```sh
mkdir build && cd build
```
then we can build the provided code by running
```sh
cmake .. && make 
```
### Running
To run it simply use
```
./eigenfaces.out
```
Command line args are not supported, so if you want to test out stuff just change the code and recompile.

## Acknowledgements
As a first introduction into SVD this (article)[https://www.ams.org/publicoutreach/feature-column/fcarc-svd] \
Another great way to visualize can be seen (here)[https://www.youtube.com/watch?v=FD4DeN81ODY] \
Dataset used is olivetti faces, taken from (here)[https://github.com/lloydmeta/Olivetti-PNG/tree/master/images]. \
Other usefull links for any advetureurs
https://sandipanweb.wordpress.com/2018/01/06/eigenfaces-and-a-simple-face-detector-with-pca-svd-in-python/ \



