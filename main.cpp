#include <af/algorithm.h>
#include <af/blas.h>
#include <af/data.h>
#include <af/device.h>
#include <af/image.h>
#include <af/lapack.h>
#include <af/seq.h>
#include <af/util.h>
#include <arrayfire.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <af/image.h>
#include <af/util.h>
#include <limits>

using namespace af;

//Faces used for training the model
#define TRAINING_FACES 399

//Can be used to visualize the recomposizion of faces from summing eigenvectors and faces
#define GRAPHICS

int main(int, char **) {

   float retention = 0.9;
   //print system info
   af::info();

   try {
      //All images
      af::array images(TRAINING_FACES, 4096, af::dtype::f32); // Note the dimensions are transposed
      af::array average_face(1, 4096);

      //Load images into the array
      for (int i = 0; i < TRAINING_FACES; ++i) {
         std::string filename = "../images/image-" + std::to_string(i) + ".png";
         af::array img = af::loadImage(filename.c_str(), false); // Load the image without conversion
         af::array flat_img = af::flat(img); // Flatten the 64x64 image into a 4096-element vector
         images(i, span) = flat_img;
      }
      
      //Images to test the accuracy
      af::array test_image(1, 4096); 
      //test_image = images(399,span);
      //test_image(0, span) = af::flat(af::loadImage("/home/Nico/Documents/code/eigenfaces/images/image-401.png", false));
      //This test image is a screenshot of 142
      test_image(0, span)= af::flat(af::loadImage("../images/image-403.png", false));

      //Find average face
      average_face = af::mean(images, 0);

      //Sub avergae face from every face
      for (int i = 0; i < TRAINING_FACES; i++) {
         images(i, span) = images(i,span) - average_face;
      }

      //We use SVD instead of covariance matrix to cut down on time?
      af::array U, S, VT; 
      af::svd(U, S, VT, images.T());
      
      //Will both be used if graphics and if non graphics
      af::array image_reconstruction;

#ifdef GRAPHICS

      af::array csum = af::accum(S);

      af::array lambda = diag(S, 0, false).as(f32);

      int start = 0;
      int end = TRAINING_FACES;

      af::array new_csum(TRAINING_FACES);
      float sum_lambda = sum<float>(lambda); // Calculate sum of lambda outside the loop

      for (int i = 0; i < TRAINING_FACES; i ++) {
         new_csum(i) = csum(i) / sum_lambda;      
      }
      af::array arr = af::range(dim4(TRAINING_FACES));

      const static int width = 512, height = 512;
      
      af::Window window(width, height, "Eigenvalues");
      af::Window window2(width, height, "Cumulative variance proportion");
      af::Window window3(width, height, "Image recreated");
      af::Window window4(width, height, "Current eigenface");
      
      int i = 1;
      //This loop slowly reconstructs the image showing the eigenfaces used
      do{
         af::timer delay = timer::start();

         af::array U_pc = U.cols(0, i);
         array x = matmul(U_pc, test_image - average_face, AF_MAT_TRANS, AF_MAT_TRANS);

         image_reconstruction = matmul(U_pc, x) + transpose(average_face);

         //Draw functions
         window3.image(af::moddims(image_reconstruction/255.f, 64, 64));
         window2.plot(arr, new_csum);
         window.plot(arr, S);
         window4.image(af::moddims(U.col(i-1)*-15+average_face.T()/max(average_face),64,64));


         if (i < TRAINING_FACES - 1) {
            i++;
         }else {
            //Commenting out the break will keep the graphics
            break;
         }

         double fps = 60;
         while (timer::stop(delay) < (1 / fps)) {}
      } while(!window.close());
#endif
      /* This doesnt work lol
      for (int i = 0 ; i < 30; i++) {

         std::string filename = "../eigenvectors/eigenvector-" + std::to_string(i) + ".png";
         af::saveImage(filename.c_str(), af::moddims(U.col(i-1)*-6+average_face.T()/max(average_face),64,64));
      }
      */

      //Lets build the recontruction of the image (image_reconstruction)
      array x = matmul(U.cols(0,TRAINING_FACES-1), test_image - average_face, AF_MAT_TRANS, AF_MAT_TRANS);

      image_reconstruction = matmul(U.cols(0,TRAINING_FACES-1), x) + transpose(average_face);

      //Check if it's a face (distance between reconstructed and actual)
      printf("\nReconstructione error: %f\n", af::norm((test_image) - image_reconstruction.T()));

      //Facial recognition, we need some images of the person in the dataset
      array face_weights = matmul(test_image - average_face, U.cols(0, TRAINING_FACES-1));

      array temp = matmul(images, U.cols(0,TRAINING_FACES-1));
      array diff(TRAINING_FACES,TRAINING_FACES);

      for (int i = 0; i < TRAINING_FACES; i++) {
         diff(i, span) = temp(i, span) - face_weights;
      }

      array diff_square = pow(diff, 2);
      array match_index, min_value;

      af::min(min_value, match_index, sum(diff_square, 1));

      //match_index tells us which index x of image-x.png the given test_image is most close to 
      std::cout << "Match index type: " << match_index.type() << std::endl;
      std::cout << "Match index shape: " << match_index.dims() << std::endl;
      uint32_t* host_match_index = match_index.host<uint32_t>();
      std::cout << "index:" << host_match_index[0]<< std::endl;
      freeHost(host_match_index);

   } catch (af::exception &e) {
      fprintf(stderr, "%s\n", e.what());
      throw;
   }
   return 0;
}
