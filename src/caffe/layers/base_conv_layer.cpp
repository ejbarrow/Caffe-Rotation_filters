#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"


#include "caffe/layers/conv_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::viewFilter(const int i, const int j, const bool showW, const Dtype* weights) {
		const int outputSize = weight_shape_[0];
		const int inputSize = weight_shape_[1];
		const int axesSize = weight_shape_[2];
		const int xSize = weight_shape_[4]; //2D only
		const int ySize = weight_shape_[3]; //2D only

		//const Dtype* weights = this->blobs_[0]->cpu_data();

		if (axesSize == 2){// Only circulize 2 dimensional filters
			for (int Y = 0; Y < ySize; ++Y){//for each x
				string text = "[";
				for (int X = 0; X < xSize; ++X){
					int address = ((i * inputSize + j) * ySize + Y) * xSize + X;
					//weights[address] ;
					string s = "";
					if (showW == 1){
						std::ostringstream ss;
						ss << weights[address];
						s = ss.str();
					}
					else{
						if (weights[address] == 0){
							s = "0";
						}
						else{
							s = "1";
						}

					}

					if (X == xSize - 1){
						text = text + s;
					}
					else{
						text = text + s + ", ";
					}

				}
				text = text + "]";
				LOG(INFO) << text;

			}
			LOG(INFO) << "----------------------------------";
		}

	}

	template <typename Dtype>
	float BaseConvolutionLayer<Dtype>::distance(float x1, float y1, float x2, float y2){

		return sqrt(((x1 - x2)*(x1 - x2)) + ((y1 - y2)*(y1 - y2)));

	}



	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::rotator(Dtype* weights, Dtype* bias, const int degrees, const int bckflag){

		const int outputSize = weight_shape_[0];
		const int inputSize = weight_shape_[1];
		const int axesSize = weight_shape_[2];
		const int xSize = weight_shape_[4]; //2D only
		const int ySize = weight_shape_[3]; //2D only
		int address = 0;

		//const float x0 = (float(xSize) - 1.0) / 2.0;
		//const float y0 = (float(ySize) - 1.0) / 2.0;

		const float rstride = float(degrees) * M_PI / float(180);
		float radians = float(0);
		const int nfilters = outputSize/this->num_layers_;


		if (enlarge_rotate_ == 0){
			if (axesSize == 2){// Only circulize 2 dimensional filters
				const float x0 = (float(xSize) - 1.0) / 2.0;
				const float y0 = (float(ySize) - 1.0) / 2.0;
				for (int n = 1; n<this->num_layers_; ++n) {
					radians=rstride*n;
					if (bckflag == 1) {
						radians=-1*radians;
					}

					for (int O = 0; O < nfilters; ++O){ //for each output

						if (bckflag == 1) {
							bias[O]=bias[O]+bias[n*nfilters+O];
						}
						else {
							bias[n*nfilters+O]=bias[O];
						}

						vector<vector<vector<float> > > temp(inputSize, vector<vector<float> >(ySize, vector <float>(xSize, 0.0)));

						//address = (((O+n*nfilters) * inputSize + I) * ySize + Y) * xSize + X;
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

								float x2 = cos(radians) * (float(x1) - x0) - sin(radians) * (float(y1) - y0) + x0;
								float y2 = sin(radians) * (float(x1) - x0) + cos(radians) * (float(y1) - y0) + y0;

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

									if (bckflag==1) {
										address = (((nfilters*n+O) * inputSize + I) * ySize + Y) * xSize + X;
									}
									else {
										address = ((O * inputSize + I) * ySize + Y) * xSize + X;
									}
									//Check new location is in range
									//split weight accross surrounding pixels


									if ((left >= 0) && (up >= 0) && (left <= xSize - 1) && (up <= ySize - 1)){
										temp[I][up][left] = temp[I][up][left] + uw * lw * weights[address];
									}
									if ((left >= 0) && (down >= 0) && (left <= xSize - 1) && (down <= ySize - 1)){
										temp[I][down][left] = temp[I][down][left] + dw * lw * weights[address];
									}
									if ((right >= 0) && (up >= 0) && (right <= xSize - 1) && (up <= ySize - 1)){
										temp[I][up][right] = temp[I][up][right] + uw * rw * weights[address];
									}
									if ((right >= 0) && (down >= 0) && (right <= xSize - 1) && (down <= ySize - 1)){
										temp[I][down][right] = temp[I][down][right] + dw * rw * weights[address];
									}



								}
							}
						}

						for (int I = 0; I < inputSize; ++I){//for each input
							for (int Y = 0; Y < ySize; ++Y){//for each x
								for (int X = 0; X < xSize; ++X){//for each y
									if (bckflag==1) {
										address = ((O * inputSize + I) * ySize + Y) * xSize + X;
										weights[address] = weights[address]+temp[I][Y][X];
									}
									else {
										address = (((O+n*nfilters) * inputSize + I) * ySize + Y) * xSize + X;
										weights[address] = temp[I][Y][X];
									}
								}
							}
						}
					}
				}
			}

		}
		else{ // This code will enlarge the filter, before rotating the filter, then useing average pooling to reduce the filter to the original size.
			
			const float x0 = (float(xSize*2) - 1.0) / 2.0;
			const float y0 = (float(ySize*2) - 1.0) / 2.0;
			
			for (int n = 1; n<this->num_layers_; ++n) {
				radians=rstride*n;
				if (bckflag == 1) {
					radians=-1*radians;
				}

				for (int O = 0; O < nfilters; ++O){ //for each output

					if (bckflag == 1) {
						bias[O]=bias[O]+bias[n*nfilters+O];
					}
					else {
						bias[n*nfilters+O]=bias[O];
					}

			//increase by a factor of 2

				vector<vector<vector<float> > > templarge(inputSize, vector<vector<float> >(ySize * 2, vector <float>(xSize * 2, 0.0)));
				for (int I = 0; I < inputSize; ++I){//for each i
					for (int Y = 0; Y < ySize; ++Y){//for each x
						for (int X = 0; X < xSize; ++X){//for each y
							int A = Y*2 ;//Y
							int B = X*2;//X

							//address = (((O + n*nfilters) * inputSize + I) * ySize + Y) * xSize + X;
							if (bckflag==1) {
								address = (((nfilters*n+O) * inputSize + I) * ySize + Y) * xSize + X;
							}
							else {
								address = ((O * inputSize + I) * ySize + Y) * xSize + X;
							}

							templarge[I][A][B] = weights[address];
							templarge[I][A + 1][B] = weights[address];
							templarge[I][A][B + 1] = weights[address];
							templarge[I][A + 1][B + 1] = weights[address];
						}
					}
				}

				// --------------------------------------------------------------------------------------------------------------------
				//rotate

				vector<vector<vector<float> > > temp(inputSize, vector<vector<float> >(ySize*2, vector <float>(xSize*2, 0.0)));

				for (int I = 0; I < inputSize; ++I){//for each i
					for (int Y = 0; Y < ySize*2; ++Y){//for each x
						for (int X = 0; X < xSize*2; ++X){//for each y
							temp[I][Y][X] = 0.0;
						}
					}
				}



				for (int Y = 0; Y < ySize*2; ++Y){//for each x
					for (int X = 0; X < xSize*2; ++X){//for each y

						int y1 = Y;
						int x1 = X;

						//convert to radians

						float x2 = cos(radians) * (float(x1) - x0) - sin(radians) * (float(y1) - y0) + x0;
						float y2 = sin(radians) * (float(x1) - x0) + cos(radians) * (float(y1) - y0) + y0;

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

							//int address = ((O * inputSize + I) * ySize + Y) * xSize + X;
							//Check new location is in range
							//split weight accross surrounding pixels


							if ((left >= 0) && (up >= 0) && (left <= xSize*2 - 1) && (up <= ySize*2 - 1)){
								temp[I][up][left] = temp[I][up][left] + uw * lw * templarge[I][Y][X];
							}
							if ((left >= 0) && (down >= 0) && (left <= xSize*2 - 1) && (down <= ySize*2 - 1)){
								temp[I][down][left] = temp[I][down][left] + dw * lw * templarge[I][Y][X];
							}
							if ((right >= 0) && (up >= 0) && (right <= xSize*2 - 1) && (up <= ySize*2 - 1)){
								temp[I][up][right] = temp[I][up][right] + uw * rw * templarge[I][Y][X];
							}
							if ((right >= 0) && (down >= 0) && (right <= xSize*2 - 1) && (down <= ySize*2 - 1)){
								temp[I][down][right] = temp[I][down][right] + dw * rw * templarge[I][Y][X];
							}



						}
					}
				}


				// --------------------------------------------------------------------------------------------------------------------




				//decrease by factor of 2
				for (int I = 0; I < inputSize; ++I){//for each i
					for (int Y = 0; Y < ySize; ++Y){//for each x
						for (int X = 0; X < xSize; ++X){//for each y
							int A = Y * 2 ;//Y
							int B = X * 2 ;//X

							//address = ((O * inputSize + I) * ySize + Y) * xSize + X;
							//float value = temp[I][A][B] + temp[I][A + 1][B] + temp[I][A][B + 1] + temp[I][A + 1][B + 1];
							//weights[address] = value /4 ;
							if (bckflag==1) {
								address = ((O * inputSize + I) * ySize + Y) * xSize + X;
								float value = temp[I][A][B] + temp[I][A + 1][B] + temp[I][A][B + 1] + temp[I][A + 1][B + 1];
								weights[address] = weights[address]+value/4;
							}
							else {
								address = (((O+n*nfilters) * inputSize + I) * ySize + Y) * xSize + X;
								float value = temp[I][A][B] + temp[I][A + 1][B] + temp[I][A][B + 1] + temp[I][A + 1][B + 1];
								weights[address] = value/4;
							}

						}
					}
				}


			}

}
		}
	}






	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::Circulise_weights(){ //for weight diffs
		circulize(this->blobs_[0]->mutable_cpu_data());
	}



	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::Circulise_weights_gpu(){ //for weight diffs
		circulize(this->blobs_[0]->mutable_cpu_data());
		//this->blobs_[0]->gpu_data();
	}



	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::circulize(Dtype* weights){
		// - blobs_[0] holds the filter weights
		// - blobs_[1] holds the biases (optional)
		// - weight_shape_  Holds the weight Shapes
		//LOG(INFO) << "Circulizing Weights... " ;

		const int outputSize = weight_shape_[0];
		const int inputSize = weight_shape_[1];
		const int axesSize = weight_shape_[2];
		const int xSize = weight_shape_[4]; //2D only
		const int ySize = weight_shape_[3]; //2D only
const int nfilters = outputSize/this->num_layers_;



		const float midpointx = (float(xSize)) / 2.0;
		const float midpointy = (float(ySize)) / 2.0;
		float distThreshold = NULL;

		if (xSize <= ySize){
			distThreshold = (float(xSize)-0.7) / 2.0;
		}
		else {
			distThreshold = (float(ySize)-0.7) / 2.0;
		}


		if (axesSize == 2){// Only circulize 2 dimensional filters
			for (int O = 0; O < nfilters; ++O){ //for each output
				for (int I = 0; I < inputSize; ++I){//for each input
					for (int Y = 0; Y < ySize; ++Y){//for each x
						for (int X = 0; X < xSize; ++X){//for each y
							float squareMidx = X + 0.5;
							float squareMidy = Y + 0.5;

							float distance = sqrt(((squareMidx - midpointx)*(squareMidx - midpointx)) + ((squareMidy - midpointy)*(squareMidy - midpointy)));

							if (distance > distThreshold){ //if over threshold set weight to zero
								int address = ((O * inputSize + I) * ySize + Y) * xSize + X;
								weights[address] = 0;
								//LOG(INFO) << "This is running";
							}
						}
					}
				}
			}
		}


	}




	template <typename Dtype>
	void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			// Configure the kernel size, padding, stride, and inputs.
			ConvolutionParameter conv_param = this->layer_param_.convolution_param();
			force_nd_im2col_ = conv_param.force_nd_im2col();
			channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
			const int first_spatial_axis = channel_axis_ + 1;
			const int num_axes = bottom[0]->num_axes();
			rotation_stride_ = conv_param.rotation_stride();
			enlarge_rotate_ = conv_param.enlarge_rotate();
			if (rotation_stride_ == 0) rotation_stride_ = 360;
			num_spatial_axes_ = num_axes - first_spatial_axis;
			CHECK_GE(num_spatial_axes_, 0);
			vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
			vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
			// Setup filter kernel dimensions (kernel_shape_).
			kernel_shape_.Reshape(spatial_dim_blob_shape);
			int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
			if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
				CHECK_EQ(num_spatial_axes_, 2)
				<< "kernel_h & kernel_w can only be used for 2D convolution.";
				CHECK_EQ(0, conv_param.kernel_size_size())
				<< "Either kernel_size or kernel_h/w should be specified; not both.";
				kernel_shape_data[0] = conv_param.kernel_h();
				kernel_shape_data[1] = conv_param.kernel_w();
			}
			else {
				const int num_kernel_dims = conv_param.kernel_size_size();
				CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
				<< "kernel_size must be specified once, or once per spatial dimension "
				<< "(kernel_size specified " << num_kernel_dims << " times; "
				<< num_spatial_axes_ << " spatial dims).";
				for (int i = 0; i < num_spatial_axes_; ++i) {
					kernel_shape_data[i] =
					conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
				}
			}
			for (int i = 0; i < num_spatial_axes_; ++i) {
				CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
			}
			// Setup stride dimensions (stride_).
			stride_.Reshape(spatial_dim_blob_shape);
			int* stride_data = stride_.mutable_cpu_data();
			if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
				CHECK_EQ(num_spatial_axes_, 2)
				<< "stride_h & stride_w can only be used for 2D convolution.";
				CHECK_EQ(0, conv_param.stride_size())
				<< "Either stride or stride_h/w should be specified; not both.";
				stride_data[0] = conv_param.stride_h();
				stride_data[1] = conv_param.stride_w();
			}
			else {
				const int num_stride_dims = conv_param.stride_size();
				CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
					num_stride_dims == num_spatial_axes_)
					<< "stride must be specified once, or once per spatial dimension "
					<< "(stride specified " << num_stride_dims << " times; "
					<< num_spatial_axes_ << " spatial dims).";
					const int kDefaultStride = 1;
					for (int i = 0; i < num_spatial_axes_; ++i) {
						stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
						conv_param.stride((num_stride_dims == 1) ? 0 : i);
						CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
					}
				}
				// Setup pad dimensions (pad_).
				pad_.Reshape(spatial_dim_blob_shape);
				int* pad_data = pad_.mutable_cpu_data();
				if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
					CHECK_EQ(num_spatial_axes_, 2)
					<< "pad_h & pad_w can only be used for 2D convolution.";
					CHECK_EQ(0, conv_param.pad_size())
					<< "Either pad or pad_h/w should be specified; not both.";
					pad_data[0] = conv_param.pad_h();
					pad_data[1] = conv_param.pad_w();
				}
				else {
					const int num_pad_dims = conv_param.pad_size();
					CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
						num_pad_dims == num_spatial_axes_)
						<< "pad must be specified once, or once per spatial dimension "
						<< "(pad specified " << num_pad_dims << " times; "
						<< num_spatial_axes_ << " spatial dims).";
						const int kDefaultPad = 0;
						for (int i = 0; i < num_spatial_axes_; ++i) {
							pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
							conv_param.pad((num_pad_dims == 1) ? 0 : i);
						}
					}
					// Setup dilation dimensions (dilation_).
					dilation_.Reshape(spatial_dim_blob_shape);
					int* dilation_data = dilation_.mutable_cpu_data();
					const int num_dilation_dims = conv_param.dilation_size();
					CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
						num_dilation_dims == num_spatial_axes_)
						<< "dilation must be specified once, or once per spatial dimension "
						<< "(dilation specified " << num_dilation_dims << " times; "
						<< num_spatial_axes_ << " spatial dims).";
						const int kDefaultDilation = 1;
						for (int i = 0; i < num_spatial_axes_; ++i) {
							dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
							conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
						}
						// Special case: im2col is the identity for 1x1 convolution with stride 1
						// and no padding, so flag for skipping the buffer and transformation.
						is_1x1_ = true;
						for (int i = 0; i < num_spatial_axes_; ++i) {
							is_1x1_ &=
							kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
							if (!is_1x1_) { break; }
						}
						// Configure output channels and groups.
						channels_ = bottom[0]->shape(channel_axis_);
						num_output_ = this->layer_param_.convolution_param().num_output();
						CHECK_GT(num_output_, 0);
						group_ = this->layer_param_.convolution_param().group();
						CHECK_EQ(channels_ % group_, 0);
						CHECK_EQ(num_output_ % group_, 0)
						<< "Number of output should be multiples of group.";
						if (reverse_dimensions()) {
							conv_out_channels_ = channels_;
							conv_in_channels_ = num_output_;
						}
						else {
							conv_out_channels_ = num_output_;
							conv_in_channels_ = channels_;
						}
						// Handle the parameters: weights and biases.
						// - blobs_[0] holds the filter weights
						// - blobs_[1] holds the biases (optional)
						vector<int> weight_shape(2);
						weight_shape[0] = conv_out_channels_;
						weight_shape[1] = conv_in_channels_ / group_;
						weight_shape_.push_back(weight_shape[0]);
						weight_shape_.push_back(weight_shape[1]);
						weight_shape_.push_back(num_spatial_axes_);
						LOG(INFO) << "num_spatial_axes_ : " << num_spatial_axes_;
						LOG(INFO) << "weight_shape[0] : " << weight_shape[0];
						LOG(INFO) << "weight_shape[1]: " << weight_shape[1];
						for (int i = 0; i < num_spatial_axes_; ++i) {
							weight_shape.push_back(kernel_shape_data[i]);
							weight_shape_.push_back(kernel_shape_data[i]);
							LOG(INFO) << "Pushing Axis";
						}

						bias_term_ = this->layer_param_.convolution_param().bias_term();
						vector<int> bias_shape(bias_term_, num_output_);
						if (this->blobs_.size() > 0) {

							CHECK_EQ(1 + bias_term_, this->blobs_.size())
							<< "Incorrect number of weight blobs.";
							if (weight_shape != this->blobs_[0]->shape()) {
								Blob<Dtype> weight_shaped_blob(weight_shape);
								LOG(FATAL) << "Incorrect weight shape: expected shape "
								<< weight_shaped_blob.shape_string() << "; instead, shape was "
								<< this->blobs_[0]->shape_string();
							}
							if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
								Blob<Dtype> bias_shaped_blob(bias_shape);
								LOG(FATAL) << "Incorrect bias shape: expected shape "
								<< bias_shaped_blob.shape_string() << "; instead, shape was "
								<< this->blobs_[1]->shape_string();
							}
							LOG(INFO) << "Skipping parameter initialization";
						}
						else {

							if (bias_term_) {
								this->blobs_.resize(2);
							}
							else {
								this->blobs_.resize(1);
							}

							// Initialize and fill the weights:
							// output channels x input channels per-group x kernel height x kernel width
							this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
							shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
								this->layer_param_.convolution_param().weight_filler()));
								weight_filler->Fill(this->blobs_[0].get());
								// If necessary, initialize and fill the biases.
								if (bias_term_) {
									this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
									shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
										this->layer_param_.convolution_param().bias_filler()));
										bias_filler->Fill(this->blobs_[1].get());

									}
								}
								kernel_dim_ = this->blobs_[0]->count(1);
								weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
								// Propagate gradients to the parameters (as directed by backward pass).
								this->param_propagate_down_.resize(this->blobs_.size(), true);

								//If Master Rotation Layer setup parallel layers
								if (rotation_stride_ != 360){
									LOG(INFO) << "Rotation Step Size: " << rotation_stride_;
									if (360 % rotation_stride_ != 0){
										LOG(FATAL) << "360 not directly divisable by rotation stride! ";
									}
									else{
										LOG(INFO) << "Rotation stride acceptable :)";
										num_layers_ = 360 / rotation_stride_;
									}

									// LOG(INFO) << "Setting up Paralel Layers";
									// for (int l = 0; l < num_layers_; ++l){
									// 	LOG(INFO) << "Setting up Paralel Layer " << l + 1;
									//
									// 	unsigned int Pstride = conv_param.stride(0);
									// 	unsigned int Pkern = conv_param.kernel_size(0);
									// 	unsigned int Pnumout = conv_param.num_output();
									//
									// 	LOG(INFO) << "Pnumout: " << Pnumout;
									// 	LOG(INFO) << "Pkern: " << Pkern;
									// 	LOG(INFO) << "Pstride: " << Pstride;
									//
									// 	LayerParameter layer_param;
									//
									// 	ConvolutionParameter* conv_param2 = layer_param.mutable_convolution_param();
									// 	conv_param2->add_kernel_size(Pkern);
									// 	conv_param2->add_stride(Pstride);
									// 	conv_param2->set_num_output(Pnumout);
									// 	conv_param2->mutable_weight_filler()->set_type("circ_xavier");
									// 	conv_param2->mutable_bias_filler()->set_type("constant");
									//
									// 	shared_ptr<Layer<Dtype> > layer(new ConvolutionLayer<Dtype>(layer_param));
									// 	layer->SetUp(bottom, top);
									// 	Conv_Layers_.push_back(layer);
									// }
								}
							}

							template <typename Dtype>
							void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top) {
									const int first_spatial_axis = channel_axis_ + 1;
									CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
									<< "bottom num_axes may not change.";
									num_ = bottom[0]->count(0, channel_axis_);
									CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
									<< "Input size incompatible with convolution kernel.";
									// TODO: generalize to handle inputs of different shapes.
									for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
										CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
										<< "All inputs must have the same shape.";
									}
									// Shape the tops.
									bottom_shape_ = &bottom[0]->shape();
									compute_output_shape();
									vector<int> top_shape(bottom[0]->shape().begin(),
									bottom[0]->shape().begin() + channel_axis_);
									top_shape.push_back(num_output_);
									for (int i = 0; i < num_spatial_axes_; ++i) {
										top_shape.push_back(output_shape_[i]);
									}
									for (int top_id = 0; top_id < top.size(); ++top_id) {
										top[top_id]->Reshape(top_shape);
									}
									if (reverse_dimensions()) {
										conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
									}
									else {
										conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
									}
									col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
									output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
									// Setup input dimensions (conv_input_shape_).
									vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
									conv_input_shape_.Reshape(bottom_dim_blob_shape);
									int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
									for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
										if (reverse_dimensions()) {
											conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
										}
										else {
											conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
										}
									}
									// The im2col result buffer will only hold one image at a time to avoid
									// overly large memory usage. In the special case of 1x1 convolution
									// it goes lazily unused to save memory.
									col_buffer_shape_.clear();
									col_buffer_shape_.push_back(kernel_dim_ * group_);
									for (int i = 0; i < num_spatial_axes_; ++i) {
										if (reverse_dimensions()) {
											col_buffer_shape_.push_back(input_shape(i + 1));
										}
										else {
											col_buffer_shape_.push_back(output_shape_[i]);
										}
									}
									col_buffer_.Reshape(col_buffer_shape_);
									bottom_dim_ = bottom[0]->count(channel_axis_);
									top_dim_ = top[0]->count(channel_axis_);
									num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
									num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
									// Set up the all ones "bias multiplier" for adding biases by BLAS
									out_spatial_dim_ = top[0]->count(first_spatial_axis);
									if (bias_term_) {
										vector<int> bias_multiplier_shape(1, out_spatial_dim_);
										bias_multiplier_.Reshape(bias_multiplier_shape);
										caffe_set(bias_multiplier_.count(), Dtype(1),
										bias_multiplier_.mutable_cpu_data());
									}
								}

								template <typename Dtype>
								void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
									const Dtype* weights, Dtype* output, bool skip_im2col) {
										const Dtype* col_buff = input;
										if (!is_1x1_) {
											if (!skip_im2col) {
												conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
											}
											col_buff = col_buffer_.cpu_data();
										}
										for (int g = 0; g < group_; ++g) {
											caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
												group_, conv_out_spatial_dim_, kernel_dim_,
												(Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
												(Dtype)0., output + output_offset_ * g);
											}
										}

										template <typename Dtype>
										void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
											const Dtype* bias) {
												caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
													out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
													(Dtype)1., output);
												}

												template <typename Dtype>
												void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
													const Dtype* weights, Dtype* input) {
														Dtype* col_buff = col_buffer_.mutable_cpu_data();
														if (is_1x1_) {
															col_buff = input;
														}
														for (int g = 0; g < group_; ++g) {
															caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
																conv_out_spatial_dim_, conv_out_channels_ / group_,
																(Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
																(Dtype)0., col_buff + col_offset_ * g);
															}
															if (!is_1x1_) {
																conv_col2im_cpu(col_buff, input);
															}
														}

														template <typename Dtype>
														void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
															const Dtype* output, Dtype* weights) {
																const Dtype* col_buff = input;
																if (!is_1x1_) {
																	conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
																	col_buff = col_buffer_.cpu_data();
																}
																for (int g = 0; g < group_; ++g) {
																	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
																		kernel_dim_, conv_out_spatial_dim_,
																		(Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
																		(Dtype)1., weights + weight_offset_ * g);
																	}
																}

																template <typename Dtype>
																void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
																	const Dtype* input) {
																		caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
																			input, bias_multiplier_.cpu_data(), 1., bias);
																		}

																		#ifndef CPU_ONLY

																		template <typename Dtype>
																		void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
																			const Dtype* weights, Dtype* output, bool skip_im2col) {
																				const Dtype* col_buff = input;
																				if (!is_1x1_) {
																					if (!skip_im2col) {
																						conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
																					}
																					col_buff = col_buffer_.gpu_data();
																				}
																				for (int g = 0; g < group_; ++g) {
																					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
																						group_, conv_out_spatial_dim_, kernel_dim_,
																						(Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
																						(Dtype)0., output + output_offset_ * g);
																					}
																				}

																				template <typename Dtype>
																				void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
																					const Dtype* bias) {
																						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
																							out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
																							(Dtype)1., output);
																						}

																						template <typename Dtype>
																						void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
																							const Dtype* weights, Dtype* input) {
																								Dtype* col_buff = col_buffer_.mutable_gpu_data();
																								if (is_1x1_) {
																									col_buff = input;
																								}
																								for (int g = 0; g < group_; ++g) {
																									caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
																										conv_out_spatial_dim_, conv_out_channels_ / group_,
																										(Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
																										(Dtype)0., col_buff + col_offset_ * g);
																									}
																									if (!is_1x1_) {
																										conv_col2im_gpu(col_buff, input);
																									}
																								}

																								template <typename Dtype>
																								void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
																									const Dtype* output, Dtype* weights) {
																										const Dtype* col_buff = input;
																										if (!is_1x1_) {
																											conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
																											col_buff = col_buffer_.gpu_data();
																										}
																										for (int g = 0; g < group_; ++g) {
																											caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
																												kernel_dim_, conv_out_spatial_dim_,
																												(Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
																												(Dtype)1., weights + weight_offset_ * g);
																											}
																										}

																										template <typename Dtype>
																										void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
																											const Dtype* input) {
																												caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
																													input, bias_multiplier_.gpu_data(), 1., bias);
																												}




																												#endif  // !CPU_ONLY

																												INSTANTIATE_CLASS(BaseConvolutionLayer);

																											}  // namespace caffe
