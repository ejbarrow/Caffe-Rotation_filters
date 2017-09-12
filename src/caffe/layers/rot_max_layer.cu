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
void Rot_maxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

    const RotMaxParameter& param = this->layer_param_.rot_max_param();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int nfilters = bottom[0]->shape(1)/param.rot_num();
    int address;
    int address2;

    const int shapeN = bottom[0]->shape(0);
    const int shapeC = bottom[0]->shape(1);
    const int shapeH = bottom[0]->shape(2);
    const int shapeW = bottom[0]->shape(3);
    const int bsize = bottom[0]->count();
    const int tsize = top[0]->count();
    const int wsize = winnersaveBlob_->count();

    CHECK_EQ(winnersaveBlob_->shape(0), top[0]->shape(0)) << "dont match: " << winnersaveBlob_->shape(0) << "and " << top[0]->shape(0);
    CHECK_EQ(winnersaveBlob_->shape(1), top[0]->shape(1)) << "dont match: " << winnersaveBlob_->shape(1) << "and " << top[0]->shape(1);
    CHECK_EQ(winnersaveBlob_->shape(2), top[0]->shape(2)) << "dont match: " << winnersaveBlob_->shape(2) << "and " << top[0]->shape(2);
    CHECK_EQ(winnersaveBlob_->shape(3), top[0]->shape(3)) << "dont match: " << winnersaveBlob_->shape(3) << "and " << top[0]->shape(3);

    Dtype* winners = winnersaveBlob_->mutable_cpu_data();

    for (int Fnum = 0; Fnum < shapeN; ++Fnum){  //for each num
      for (int Fchan = 0; Fchan < nfilters; ++Fchan){  //for each channel
        for (int Fheight = 0; Fheight < shapeH; ++Fheight){  //for each height
          for (int Fwidth = 0; Fwidth < shapeW; ++Fwidth){  //for each width
            int winner = 0;
            Dtype winnerval = -1000.0;
            //Dtype winnerval = 0.0;
            address2 = ((Fnum * nfilters + Fchan) * shapeH + Fheight) * shapeW + Fwidth;
            for (int Flay = 0; Flay < param.rot_num(); ++Flay){ //for each layer
              address = ((Fnum * shapeC + Fchan+Flay*nfilters) * shapeH + Fheight) * shapeW + Fwidth;

              if (winnerval < bottom_data[address]){
              //if (fabs(winnerval) < fabs(bottom_data[address])) {
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
void Rot_maxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      const RotMaxParameter& param = this->layer_param_.rot_max_param();
  const Dtype* top_diff = top[0]->cpu_diff();

  const Dtype* winners = winnersaveBlob_->cpu_data();
  const int nfilters = bottom[0]->shape(1)/param.rot_num();
  int address;
  int address2;

  const int shapeN = bottom[0]->shape(0);
  const int shapeC = bottom[0]->shape(1);
  const int shapeH = bottom[0]->shape(2);
  const int shapeW = bottom[0]->shape(3);
  const int bsize = bottom[0]->count();
  const int tsize = top[0]->count();
  const int wsize = winnersaveBlob_->count();

  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

//zero out diff
  Dtype alpha = 0;
  caffe_set(bottom[0]->count(), alpha, bottom_diff);
  //caffe_set(bottom[0]->shape(0) * bottom[0]->shape(1) * bottom[0]->shape(2) * bottom[0]->shape(3), alpha, bottom_diff);
      // for (int Fnum = 0; Fnum < shapeN; ++Fnum){  //for each num
      //   for (int Fchan = 0; Fchan < shapeC; ++Fchan){  //for each channel
      //     for (int Fheight = 0; Fheight < shapeH; ++Fheight){  //for each height
      //       for (int Fwidth = 0; Fwidth < shapeW; ++Fwidth){  //for each width
      //
      //         address = ((Fnum * shapeC + Fchan) * shapeH + Fheight) * shapeW + Fwidth;
      //         //calculate address in array
      //
      //         bottom_diff[address]=0.0;
      //       }
      //     }
      //   }
      // }

CHECK_EQ(bottom[0]->asum_diff(),0) << "zeroing failed";

    for (int Fnum = 0; Fnum < shapeN; ++Fnum){  //for each num
      for (int Fchan = 0; Fchan < nfilters; ++Fchan){  //for each channel
        for (int Fheight = 0; Fheight < shapeH; ++Fheight){  //for each height
          for (int Fwidth = 0; Fwidth < shapeW; ++Fwidth){  //for each width

            address2 = ((Fnum * nfilters + Fchan) * shapeH + Fheight) * shapeW + Fwidth;
            //calculate address in array

            address = (((Fnum * shapeC) + (Fchan + nfilters * int(winners[address2])) ) * shapeH + Fheight) * shapeW + Fwidth;

            bottom_diff[address]=top_diff[address2];
          }
        }
      }
    }

}

}

INSTANTIATE_LAYER_GPU_FUNCS(Rot_maxLayer);

}  // namespace caffe
