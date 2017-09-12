#include < vector >
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

  template < typename Dtype >
    void ConvolutionLayer < Dtype > ::Forward_gpu(const vector < Blob < Dtype > * > & bottom,
      const vector < Blob < Dtype > * > & top) {
      Dtype * weight;
      Dtype * bias;

      int size = this - > num_layers_;

      if (this - > rotation_stride_ != 360) {
        weight = this - > blobs_[0] - > mutable_cpu_data();
        bias = this - > blobs_[1] - > mutable_cpu_data();
        this - > Circulise_weights_gpu();

        //Rotate each Layer
        this - > rotator(weight, bias, this - > rotation_stride_, 0);

      }

      //Forward propoigate
      weight = this - > blobs_[0] - > mutable_gpu_data();
      bias = this - > blobs_[1] - > mutable_gpu_data();

      for (int i = 0; i < bottom.size(); ++i) {
        const Dtype * bottom_data = bottom[i] - > gpu_data();
        Dtype * top_data = top[i] - > mutable_gpu_data();
        for (int n = 0; n < this - > num_; ++n) {
          this - > forward_gpu_gemm(bottom_data + n * this - > bottom_dim_, weight,
            top_data + n * this - > top_dim_);
          if (this - > bias_term_) {

            this - > forward_gpu_bias(top_data + n * this - > top_dim_, bias);
          }
        }
      }

    }

  template < typename Dtype >
    void ConvolutionLayer < Dtype > ::Backward_gpu(const vector < Blob < Dtype > * > & top,
      const vector < bool > & propagate_down,
        const vector < Blob < Dtype > * > & bottom) {

      const Dtype * weight = this - > blobs_[0] - > gpu_data();
      Dtype * weight_diff = this - > blobs_[0] - > mutable_gpu_diff();
      Dtype * bias_diff;
      for (int i = 0; i < top.size(); ++i) {
        const Dtype * top_diff = top[i] - > gpu_diff();

        // Bias gradient, if necessary.
        if (this - > bias_term_ && this - > param_propagate_down_[1]) {
          bias_diff = this - > blobs_[1] - > mutable_gpu_diff();
          for (int n = 0; n < this - > num_; ++n) {
            this - > backward_gpu_bias(bias_diff, top_diff + n * this - > top_dim_);
          }
        }
        if (this - > param_propagate_down_[0] || propagate_down[i]) {
          Dtype * bottom_diff = bottom[i] - > mutable_gpu_diff();
          const Dtype * bottom_data = bottom[i] - > gpu_data();
          for (int n = 0; n < this - > num_; ++n) {
            // gradient w.r.t. weight. Note that we will accumulate diffs.
            if (this - > param_propagate_down_[0]) {
              this - > weight_gpu_gemm(bottom_data + n * this - > bottom_dim_,
                top_diff + n * this - > top_dim_, weight_diff);
            }
            // gradient w.r.t. bottom data, if necessary.
            if (propagate_down[i]) {
              this - > backward_gpu_gemm(top_diff + n * this - > top_dim_, weight,
                bottom_diff + n * this - > bottom_dim_);
            }
          }
        }
      }

      if (this - > rotation_stride_ != 360) {
        if (this - > param_propagate_down_[0]) {

          bias_diff = this - > blobs_[1] - > mutable_cpu_diff();

          weight_diff = this - > blobs_[0] - > mutable_cpu_diff();

          this - > rotator(weight_diff, bias_diff, this - > rotation_stride_, 1);

        }
      }

    }

  template < typename Dtype >
    void ConvolutionLayer < Dtype > ::GetWeight_diff(vector < Blob < Dtype > * > & weight_diff) {
      weight_diff[0] - > CopyFrom( * this - > blobs_[0], 1, 1);
      weight_diff[1] - > CopyFrom( * this - > blobs_[1], 1, 1);
    }

  INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

} // namespace caffe