#include <vector>

#include "caffe/layers/data_rotate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



template <typename Dtype>
void Data_RotateLayer<Dtype>::Forward_gpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int count = top[0]->count();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	//caffe_abs(count, bottom[0]->cpu_data(), top_data);


	const int outputSize = bottom[0]->shape(0);
	const int inputSize = bottom[0]->shape(1);
	//const int axesSize = bottom[0]->shape(2);
	const int axesSize = 2;
	const int xSize = bottom[0]->shape(3); //2D only
	const int ySize = bottom[0]->shape(2); //2D only

	const float x0 = (float(xSize) - 1.0) / 2.0;
	const float y0 = (float(ySize) - 1.0) / 2.0;

	if (axesSize == 2){// Only circulize 2 dimensional filters
		for (int O = 0; O < outputSize; ++O){ //for each output
			const float radians = caffe_rng_rand() * M_PI * 2.0;
			//Initialise temp vector
			vector<vector<vector<float> > > temp(inputSize, vector<vector<float> >(ySize, vector <float>(xSize, 0.0)));
			for (int I = 0; I < inputSize; ++I){//for each i
				for (int Y = 0; Y < ySize; ++Y){//for each x
					for (int X = 0; X < xSize; ++X){//for each y
						temp[I][Y][X] = 0.0;
					}
				}
			}


			for (int Y = 0; Y < ySize; ++Y){//for each x
				for (int X = 0; X < xSize; ++X){//for each y

					int y1 = Y;
					int x1 = X;

					//convert to radians

					float x2 = cos(radians) * (x1 - x0) - sin(radians) * (y1 - y0) + x0;
					float y2 = sin(radians) * (x1 - x0) + cos(radians) * (y1 - y0) + y0;

					//Get the surrounding coordinates
					float left = floor(x2);
					float right = floor(x2) + 1.0;
					float down = floor(y2) + 1.0;
					float up = floor(y2);

					float lw = pow(fabs(right - x2), 2.0) / (pow(fabs(right - x2), 2.0) + pow(fabs(x2 - left), 2.0));
					float rw = pow(fabs(x2 - left), 2.0) / (pow(fabs(right - x2), 2.0) + pow(fabs(x2 - left), 2.0));
					float uw = pow(fabs(y2 - down), 2.0) / (pow(fabs(up - y2), 2.0) + pow(fabs(y2 - down), 2.0));
					float dw = pow(fabs(up - y2), 2.0) / (pow(fabs(up - y2), 2.0) + pow(fabs(y2 - down), 2.0));

					for (int I = 0; I < inputSize; ++I){//for each input

						int address = ((O * inputSize + I) * ySize + Y) * xSize + X;


						if ((left >= 0) && (up >= 0) && (left <= xSize - 1) && (up <= ySize - 1)){
							temp[I][up][left] = temp[I][up][left] + uw * lw * bottom_data[address];
						}
						if ((left >= 0) && (down >= 0) && (left <= xSize - 1) && (down <= ySize - 1)){
							temp[I][down][left] = temp[I][down][left] + dw * lw * bottom_data[address];
						}
						if ((right >= 0) && (up >= 0) && (right <= xSize - 1) && (up <= ySize - 1)){
							temp[I][up][right] = temp[I][up][right] + uw * rw * bottom_data[address];
						}
						if ((right >= 0) && (down >= 0) && (right <= xSize - 1) && (down <= ySize - 1)){
							temp[I][down][right] = temp[I][down][right] + dw * rw * bottom_data[address];
						}



					}
				}
			}

			for (int I = 0; I < inputSize; ++I){//for each input
				for (int Y = 0; Y < ySize; ++Y){//for each x
					for (int X = 0; X < xSize; ++X){//for each y

						int address = ((O * inputSize + I) * ySize + Y) * xSize + X;
						top_data[address] = temp[I][Y][X];
					}
				}
			}
		}
	}
}

template <typename Dtype>
void Data_RotateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const int count = top[0]->count();
	const Dtype* top_diff = top[0]->gpu_diff();
	if (propagate_down[0]) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		caffe_cpu_sign(count, bottom_data, bottom_diff);
		caffe_mul(count, bottom_diff, top_diff, bottom_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(Data_RotateLayer);

}  // namespace caffe
