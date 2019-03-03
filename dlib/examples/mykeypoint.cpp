#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <iostream>

using namespace dlib;
using namespace std;

int main(int argc, char** argv)
{
	const std::string faces_directory = argv[1];
	const std::string model_directory = argv[2];
	
	dlib::array<array2d<unsigned char> > images_train, images_test;
	std::vector<std::vector<full_object_detection> > faces_train, faces_test;

	load_image_dataset(images_train, faces_train, faces_directory+"/training_with_face_landmarks.xml");
	//load_image_dataset(images_test, faces_test, faces_directory+"/testing_with_face_landmarks.xml");

	// Now make the object responsible for training the model.  
	shape_predictor_trainer trainer;
	trainer.set_oversampling_amount(100);
	trainer.set_nu(0.1);
	trainer.set_tree_depth(5);
	trainer.set_num_threads(100);
	trainer.be_verbose();

	shape_predictor sp = trainer.train(images_train, faces_train);
	serialize(model_directory + "/sp.dat") << sp;
	
	return 0;
}


