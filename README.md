# MetaF2N: Blind Image Super-Resolution by Learning Efficient Model Adaptation from Faces

<a href="https://arxiv.org/pdf/2309.08113.pdf"><img src="https://img.shields.io/badge/arXiv-2309.08113-b31b1b.svg" height=22.5></a>

## Getting Started- EE655 Project

### Environment Setup

```shell
conda create -n metaf2n python=3.9
conda activate metaf2n
pip install -r requirements.txt
```

### Trained Models

We provided the trained checkpoints in [Google Drive](https://drive.google.com/drive/folders/1zTIHZDp0SUE-cuEvmPXj6vHd62cZLykg). One can download them and save to the directory `./weights/Model1`.

Updated trained model can be found in [Google Drive](https://drive.google.com/drive/folders/1mUIn_HdGMKDkcyYuaNocqU6qxaWiElKZ). One can download them and save to the directory `./weights/Model2`.

Moreover, the other models (pretrained GPEN, Real-ESRGAN and RetinaFace) we need for training and testing are also provided in [Google Drive](https://drive.google.com/drive/folders/1UyduarmLBkZ38NCRQSiuJSjrtPqWQXiX?usp=drive_link). One can download them and save to the directory `./weights`.

### Preparing Dataset

- Since we used tensoflow as our framework, we prepared our training data in the format of .tfrecord as [Google Drive](https://drive.google.com/drive/folders/1NGPghw74He0YF5ELNZOXGLwysFQL6UAQ) . One can download them and save to the directory `./datasets`.
- Updated Preprocessed triaining data in the format of .tfrecord can be found in [Google Drive](https://drive.google.com/drive/folders/1owEhKteloJMWkBYj1MyBycKPGRh0AeSX) . One can download them and save to the directory `./datasets`.

- If you want to prepare the training data yourself, you can use the generate_tfrecord.py and change the parameters.
    ```shell
    python scripts/generate_tfrecord.py
    ```

- All the test data are provided as  [Google Drive](https://drive.google.com/drive/folders/13aGnJXZiEKSRanu7bu6pJGutvMvKFeuV?usp=drive_link). Each synthsized dataset has two subfolders (GT, LQ)
  you can also generate Face_LQ and Face_HQ using this:
    ```shell
    python generate_test_faces.py --input_dir input_dir --output_dir output_dir
    ```

- The final test data structure is like this:
    ```shell
    ├── datasets
    │   ├── CelebA_iid
    │   │   ├── Face_HQ
    │   │   ├── Face_LQ
    │   │   ├── GT
    │   │   └── LQ
    │   ├── CelebA_ood
    │   │   ├── Face_HQ
    │   │   ├── Face_LQ
    │   │   ├── GT
    │   │   └── LQ
    │   ├── FFHQ_iid
    │   │   ├── Face_HQ
    │   │   ├── Face_LQ
    │   │   ├── GT
    │   │   └── LQ
    │   ├── FFHQ_Multi_iid
    │   │   ├── Face_HQ
    │   │   ├── Face_LQ
    │   │   ├── GT
    │   │   └── LQ
    │   ├── FFHQ_Multi_ood
    │   │   ├── Face_HQ
    │   │   ├── Face_LQ
    │   │   ├── GT
    │   │   └── LQ
    │   ├── FFHQ_ood
    │   │   ├── Face_HQ
    │   │   ├── Face_LQ
    │   │   ├── GT
    │   │   └── LQ
    ```
    
### Testing

To test the method, one can run,
```Shell
CUDA_VISIBLE_DEVICES=0 test.py --input_dir input_dir --output_dir output_dir --face_dir face_dir --patch_size patch_size --patch_num_per_img patch_num_per_img --fine_tune_num fine_tune_num
```

#### __Four parameters can be changed for flexible usage:__
```
--input_dir # test LQ image path
--output_dir # save the results path
--face_dir # the path that contains Face_LQ (Cropped LQ Face Area) and Face_HQ (Retored HQ Face Area)
--patch_size # the patch size for the cropping of HQ face area
--patch_num_per_img # the number of patches for the cropping of HQ face area
--fine_tune_num # the steps of the inner loop fine-tuning

```

### Training

To train MetaF2N, you can adjust the parameters in config.py and run,
To train my updated model with the one can delete the existing train python file and change the name of train1.py file to train.py. and then after adjusting the parameters in config.py one can run the command given below,

```Shell
python main.py --trial trial --step step --gpu gpu_id
```

### Calculate Metrics

To calculate metrics of the results, one can run,

```Shell
python calculate_metrics.py --result_dir result_dir --gt_dir gt_dir --fid_ref_dir fid_ref_dir
```

