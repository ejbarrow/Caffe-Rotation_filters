#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/rot_max_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Dtype>
void Rot_maxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
  // bottom[0] supplies the data
  // bottom[1] supplies the size
  //const RotMaxParameter& param = this->layer_param_.rot_max_param();
  CHECK_EQ(bottom.size(), 1) << "Wrong number of bottom blobs.";

}

template <typename Dtype>
void Rot_maxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const RotMaxParameter& param = this->layer_param_.rot_max_param();

  vector<int> new_shape(bottom[0]->shape());

    new_shape[1] = new_shape[1]/param.rot_num();

  top[0]->Reshape(new_shape);

  this->winnersaveBlob_->Reshape(new_shape);
}



template <typename Dtype>
void Rot_maxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

    const RotMaxParameter& param = this->layer_param_.rot_max_param();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int nfilters = bottom[0]->shape(1)/param.rot_num();

    const int shapeN = bottom[0]->shape(0);
    const int shapeC = bottom[0]->shape(1);
    const int shapeH = bottom[0]->shape(2);
    const int shapeW = bottom[0]->shape(3);

    Dtype* winners = winnersaveBlob_->mutable_cpu_data();

    for (int Fnum = 0; Fnum < shapeN; ++Fnum){  //for each num
      for (int Fchan = 0; Fchan < nfilters; ++Fchan){  //for each channel
        for (int Fheight = 0; Fheight < shapeH; ++Fheight){  //for each height
          for (int Fwidth = 0; Fwidth < shapeW; ++Fwidth){  //for each width
            int winner = 0;
            Dtype winnerval = -1000.0; //for max instead of absolute max
            //Dtype winnerval = 0.0;
            int address2 = ((Fnum * nfilters + Fchan) * shapeH + Fheight) * shapeW + Fwidth;
            for (int Flay = 0; Flay < param.rot_num(); ++Flay){ //for each layer
              int address = ((Fnum * shapeC + Fchan+Flay*nfilters) * shapeH + Fheight) * shapeW + Fwidth;

              if (winnerval < bottom_data[address]){   //use this instead for winner by max instead of absolute max
              //if (fabs(winnerval)<fabs(bottom_data[address])){
                winner = Flay;
                winnerval = bottom_data[address];
              }

            }
              top_data[address2] = winnerval;
              winners[address2] = winner;

          }
        }
      }
    }

  }

template <typename Dtype>
void Rot_maxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

      const RotMaxParameter& param = this->layer_param_.rot_max_param();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  const Dtype* winners = winnersaveBlob_->cpu_data();
  const int nfilters = bottom[0]->shape(1)/param.rot_num();

  const int shapeN = bottom[0]->shape(0);
  const int shapeC = bottom[0]->shape(1);
  const int shapeH = bottom[0]->shape(2);
  const int shapeW = bottom[0]->shape(3);

  CHECK_EQ(winnersaveBlob_->shape(0), top[0]->shape(0)) << "dont match: " << winnersaveBlob_->shape(0) << "and " << top[0]->shape(0);
  CHECK_EQ(winnersaveBlob_->shape(1), top[0]->shape(1)) << "dont match: " << winnersaveBlob_->shape(1) << "and " << top[0]->shape(1);
  CHECK_EQ(winnersaveBlob_->shape(2), top[0]->shape(2)) << "dont match: " << winnersaveBlob_->shape(2) << "and " << top[0]->shape(2);
  CHECK_EQ(winnersaveBlob_->shape(3), top[0]->shape(3)) << "dont match: " << winnersaveBlob_->shape(3) << "and " << top[0]->shape(3);

  if (propagate_down[0]) {
    //caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    //std::vector<int> indices(top[0]->num_axes(), 0);
    //crop_copy(bottom, top, offsets, indices, 0, top_diff, bottom_diff, false);

//zero out diff
    for (int Fnum = 0; Fnum < shapeN; ++Fnum){  //for each num
      for (int Fchan = 0; Fchan < shapeC; ++Fchan){  //for each channel
        for (int Fheight = 0; Fheight < shapeH; ++Fheight){  //for each height
          for (int Fwidth = 0; Fwidth < shapeW; ++Fwidth){  //for each width

            int address = ((Fnum * shapeC + Fchan) * shapeH + Fheight) * shapeW + Fwidth;
            //calculate address in array

            bottom_diff[address]=0;
          }
        }
      }
    }

    for (int Fnum = 0; Fnum < shapeN; ++Fnum){  //for each num
      for (int Fchan = 0; Fchan < nfilters; ++Fchan){  //for each channel
        for (int Fheight = 0; Fheight < shapeH; ++Fheight){  //for each height
          for (int Fwidth = 0; Fwidth < shapeW; ++Fwidth){  //for each width

            int address2 = ((Fnum * nfilters + Fchan) * shapeH + Fheight) * shapeW + Fwidth;
            //calculate address in array
            int address = ((Fnum * shapeC + (Fchan+winners[address2]*nfilters)) * shapeH + Fheight) * shapeW + Fwidth;
            bottom_diff[address]=top_diff[address2];
          }
        }
      }
    }

  }

}

#ifdef CPU_ONLY
STUB_GPU(Rot_maxLayer);
#endif

INSTANTIATE_CLASS(Rot_maxLayer);
REGISTER_LAYER_CLASS(Rot_max);

}  // namespace caffe
