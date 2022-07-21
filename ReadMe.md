# Download Model Checkpoints

+ Download model checkpoints from the following Google Drive Link:
  + https://drive.google.com/drive/folders/14F6gjjQ3Q7aHzOnXiW29jbPdme5Ercip?usp=sharing
+ There will be 11 model checkpoints. Drag them into "/Submit_LSTM_Ensemble_Final(Arontier_HYY)(200122)/code/checkpoints" folder.
  + 480_CV1_checkpoint.pth, 480_CV2_checkpoint.pth, 480_CV3_checkpoint.pth, 480_CV4_checkpoint.pth, 480_CV5_checkpoint.pth
  + 912_CV1_checkpoint.pth, 912_CV2_checkpoint.pth, 912_CV3_checkpoint.pth, 912_CV4_checkpoint.pth, 912_CV5_checkpoint.pth
  + background_remove_checkpoint.pth

# Requirements

+ python 3.6 with anaconda packages
+ openslide-python 1.1.1
+ imgaug 0.3.0 (pip install imgaug)
+ pytorch 1.3.1, torchvision 0.4.2
+ efficientnet-pytorch 0.5.1 (pip install --upgrade efficientnet-pytorch) https://github.com/lukemelas/EfficientNet-PyTorch
+ GPU Memory >= 10GB
+ run on linux OS
# Usage
## 1. Run script `Make_Test_Image_List_Dict_F.py`
+ Run the code to get grid points of tissue region of the WSIs

+ Run for tile size 480: 

    ```
    CUDA_VISIBLE_DEVICES=0 python Make_Test_Image_List_Dict_F.py --tile_size=480 --test_slides_dir=path/TestDataset/Slides --batch_size=64 --num_workers=8
    ```
    + `--test_slides_dir=path/TestDataset/Slides` Change the path to yours.

+ Run for tile size 912: 

    ```
    CUDA_VISIBLE_DEVICES=0 python Make_Test_Image_List_Dict_F.py --tile_size=912 --test_slides_dir=path/TestDataset/Slides --batch_size=64 --num_workers=8
    ```

    + `--test_slides_dir=path/TestDataset/Slides` Change the path to yours.

+ The program takes several hours with default settings for one tile size. After the program ends, it will create **tile_dict_480** and **tile_dict_912** folders under current directory. There will be **Check** folder and **Test_Dict.pth** file in each folder.

+ There are two kinds of images in **Check** folder, _Dict.jpg and _Ori.jpg.
    - _Dict.jpg is after removing background and re-stitched tile images
    - _Ori.jpg is direct downsampled the WSI image
    - You should check the _Dict.jpg images that whether the background removing and tiling working well

+ `Test_Dict.pth` is a dictionary, and contains grid points for each tissue tile in the WSIs.

    ``````
    Total_Dict = {
    	'slides': [], #slide name
        'grid': [],
        'level': Level,
        'dict_tile_size': Tile_Size,
        'dict_overlap': Overlap
    }
    ``````

## 2. Run Script `WSI_RNN_Test_Sampler_F.py`

+ Run the code will get a result csv file for a model. We trained 5 models for each tile size that is totally 10 models. So you should run this code for 10 times with different input parameters.

+ In the **checkpoints** folder, there are 11 model checkpoints. one is used to remove background, five for 480 and five for 912 tile size.

+ As for tile size 480, run the code with following command. 
    ````
    CUDA_VISIBLE_DEVICES=0 python WSI_RNN_Test_Sampler_F.py --tile_size=480 --test_lib=tile_dict_480/Test_Dict.pth --test_slides_dir=path/TestDataset/Slides --checkpoint=checkpoints/480_CV1_checkpoint.pth --batch_size=64
    ````

    + `--test_slides_dir=path/TestDataset/Slides` Change the path to yours.
    + `--checkpoint=checkpoints/480_CV1_checkpoint.pth` Change CV1 by CV2, CV3, CV4 and CV5. Totally run 5 times.

+ As for tile size 912, run the code with following command. 

    ````
    CUDA_VISIBLE_DEVICES=0 python WSI_RNN_Test_Sampler_F.py --tile_size=912 --test_lib=tile_dict_912/Test_Dict.pth --test_slides_dir=path/TestDataset/Slides --checkpoint=checkpoints/912_CV1_checkpoint.pth --batch_size=32
    ````

    + `--test_slides_dir=path/TestDataset/Slides` Change the path to yours.
    + `--checkpoint=checkpoints/912_CV1_checkpoint.pth` Change CV1 by CV2, CV3, CV4 and CV5. Totally run 5 times.

+ The run the program once takes one hour for 480 and several hours for 912. After the program ends, it will create **results** folder under current directory. There will be **_checkpoint.csv** and **_checkpoint_probs_features.pth** files

+ **_checkpoint.csv** is a result file for each model. **_checkpoint_probs_features.pth** contains probability and features for each tile image by the model checkpoint, and it is made just in case something goes wrong and need to re-run the program, it can save lots of time.

+ Check if there are 10 **_checkpoint.csv** files in the **results** folder.

## 3. Run Script `CSV_Results_Ensemble.py`

+ Run the code and it will create **Arontier_HYY.csv** that is final ensemble result of the ten **_checkpoint.csv** files in the **results** folder.
    ````
    python CSV_Results_Ensemble.py
    ````