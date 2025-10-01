
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
pretty_name: "CIFAR10 with leftover data for calibration"
license: mit
language: en
task_categories: ["image-classification"]
size_categories: 100K<n<1M
source_datasets: "https://www.cs.toronto.edu/~kriz/cifar.html"
dataset_info:
  features:
    - name: image
      dtype: image
    - name: label
      dtype: int64
    - name: classname
      dtype: string
configs:
- config_name: complete
  default: true
  data_files:
  - split: train
    path: "train/*.zip"
  - split: calibration
    path: "calibration/*.zip"
  - split: test
    path: "test/*.zip"
- config_name: no_airplane
  data_files:
  - split: train
    path:
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
  - split: calibration
    path:
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
  - split: test
    path:
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
  - split: left_out_train
    path: "train/airplane.zip"
  - split: left_out_calibration
    path: "calibration/airplane.zip"
  - split: left_out_test
    path: "test/airplane.zip"
  - split: left_out
    path: 
    - "train/airplane.zip"
    - "calibration/airplane.zip"
    - "test/airplane.zip"
- config_name: no_automobile
  data_files:
  - split: train
    path:
    - "train/airplane.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
  - split: calibration
    path:
    - "calibration/airplane.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
  - split: test
    path:
    - "test/airplane.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
  - split: left_out_train
    path: "train/automobile.zip"
  - split: left_out_calibration
    path: "calibration/automobile.zip"
  - split: left_out_test
    path: "test/automobile.zip"
  - split: left_out
    path: 
    - "train/automobile.zip"
    - "calibration/automobile.zip"
    - "test/automobile.zip"
- config_name: no_bird
  data_files:
  - split: train
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
  - split: calibration
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
  - split: test
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
  - split: left_out_train
    path: "train/bird.zip"
  - split: left_out_calibration
    path: "calibration/bird.zip"
  - split: left_out_test
    path: "test/bird.zip"
  - split: left_out
    path: 
    - "train/bird.zip"
    - "calibration/bird.zip"
    - "test/bird.zip"
- config_name: no_cat
  data_files:
  - split: train
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
  - split: calibration
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
  - split: test
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
  - split: left_out_train
    path: "train/cat.zip"
  - split: left_out_calibration
    path: "calibration/cat.zip"
  - split: left_out_test
    path: "test/cat.zip"
  - split: left_out
    path: 
    - "train/cat.zip"
    - "calibration/cat.zip"
    - "test/cat.zip"
- config_name: no_deer
  data_files:
  - split: train
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
  - split: calibration
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
  - split: test
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
  - split: left_out_train
    path: "train/deer.zip"
  - split: left_out_calibration
    path: "calibration/deer.zip"
  - split: left_out_test
    path: "test/deer.zip"
  - split: left_out
    path: 
    - "train/deer.zip"
    - "calibration/deer.zip"
    - "test/deer.zip"
- config_name: no_dog
  data_files:
  - split: train
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
  - split: calibration
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
  - split: test
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
  - split: left_out_train
    path: "train/dog.zip"
  - split: left_out_calibration
    path: "calibration/dog.zip"
  - split: left_out_test
    path: "test/dog.zip"
  - split: left_out
    path: 
    - "train/dog.zip"
    - "calibration/dog.zip"
    - "test/dog.zip"
- config_name: no_frog
  data_files:
  - split: train
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
  - split: calibration
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
  - split: test
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
  - split: left_out_train
    path: "train/frog.zip"
  - split: left_out_calibration
    path: "calibration/frog.zip"
  - split: left_out_test
    path: "test/frog.zip"
  - split: left_out
    path: 
    - "train/frog.zip"
    - "calibration/frog.zip"
    - "test/frog.zip"
- config_name: no_horse
  data_files:
  - split: train
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/ship.zip"
    - "train/truck.zip"
  - split: calibration
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
  - split: test
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/ship.zip"
    - "test/truck.zip"
  - split: left_out_train
    path: "train/horse.zip"
  - split: left_out_calibration
    path: "calibration/horse.zip"
  - split: left_out_test
    path: "test/horse.zip"
  - split: left_out
    path: 
    - "train/horse.zip"
    - "calibration/horse.zip"
    - "test/horse.zip"
- config_name: no_ship
  data_files:
  - split: train
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/truck.zip"
  - split: calibration
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/truck.zip"
  - split: test
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/truck.zip"
  - split: left_out_train
    path: "train/ship.zip"
  - split: left_out_calibration
    path: "calibration/ship.zip"
  - split: left_out_test
    path: "test/ship.zip"
  - split: left_out
    path: 
    - "train/ship.zip"
    - "calibration/ship.zip"
    - "test/ship.zip"
- config_name: no_truck
  data_files:
  - split: train
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
  - split: calibration
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
  - split: test
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
  - split: left_out_train
    path: "train/truck.zip"
  - split: left_out_calibration
    path: "calibration/truck.zip"
  - split: left_out_test
    path: "test/truck.zip"
  - split: left_out
    path: 
    - "train/truck.zip"
    - "calibration/truck.zip"
    - "test/truck.zip"
# From here, STUPID WORKAROUNDS for
# https://discuss.huggingface.co/t/download-only-requested-split/165968
- config_name: train
  data_files:
  - split: unique_split
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
- config_name: calibration
  data_files:
  - split: unique_split
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
- config_name: test
  data_files:
  - split: unique_split
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
- config_name: no_airplane_train
  data_files:
  - split: unique_split
    path:
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
- config_name: airplane_train
  data_files:
  - split: unique_split
    path: "train/airplane.zip"
- config_name: no_airplane_calibration
  data_files:
  - split: unique_split
    path:
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
- config_name: airplane_calibration
  data_files:
  - split: unique_split
    path: "calibration/airplane.zip"
- config_name: no_airplane_test
  data_files:
  - split: unique_split
    path:
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
- config_name: airplane_test
  data_files:
  - split: unique_split
    path: "test/airplane.zip"
- config_name: no_automobile_train
  data_files:
  - split: unique_split
    path:
    - "train/airplane.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
- config_name: automobile_train
  data_files:
  - split: unique_split
    path: "train/automobile.zip"
- config_name: no_automobile_calibration
  data_files:
  - split: unique_split
    path:
    - "calibration/airplane.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
- config_name: automobile_calibration
  data_files:
  - split: unique_split
    path: "calibration/automobile.zip"
- config_name: no_automobile_test
  data_files:
  - split: unique_split
    path:
    - "test/airplane.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
- config_name: automobile_test
  data_files:
  - split: unique_split
    path: "test/automobile.zip"
- config_name: no_bird_train
  data_files:
  - split: unique_split
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
- config_name: bird_train
  data_files:
  - split: unique_split
    path: "train/bird.zip"
- config_name: no_bird_calibration
  data_files:
  - split: unique_split
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
- config_name: bird_calibration
  data_files:
  - split: unique_split
    path: "calibration/bird.zip"
- config_name: no_bird_test
  data_files:
  - split: unique_split
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
- config_name: bird_test
  data_files:
  - split: unique_split
    path: "test/bird.zip"
- config_name: no_cat_train
  data_files:
  - split: unique_split
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
- config_name: cat_train
  data_files:
  - split: unique_split
    path: "train/cat.zip"
- config_name: no_cat_calibration
  data_files:
  - split: unique_split
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
- config_name: cat_calibration
  data_files:
  - split: unique_split
    path: "calibration/cat.zip"
- config_name: no_cat_test
  data_files:
  - split: unique_split
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
- config_name: cat_test
  data_files:
  - split: unique_split
    path: "test/cat.zip"
- config_name: no_deer_train
  data_files:
  - split: unique_split
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
- config_name: deer_train
  data_files:
  - split: unique_split
    path: "train/deer.zip"
- config_name: no_deer_calibration
  data_files:
  - split: unique_split
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
- config_name: deer_calibration
  data_files:
  - split: unique_split
    path: "calibration/deer.zip"
- config_name: no_deer_test
  data_files:
  - split: unique_split
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
- config_name: deer_test
  data_files:
  - split: unique_split
    path: "test/deer.zip"
- config_name: no_dog_train
  data_files:
  - split: unique_split
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
- config_name: dog_train
  data_files:
  - split: unique_split
    path: "train/dog.zip"
- config_name: no_dog_calibration
  data_files:
  - split: unique_split
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
- config_name: dog_calibration
  data_files:
  - split: unique_split
    path: "calibration/dog.zip"
- config_name: no_dog_test
  data_files:
  - split: unique_split
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
- config_name: dog_test
  data_files:
  - split: unique_split
    path: "test/dog.zip"
- config_name: no_frog_train
  data_files:
  - split: unique_split
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
    - "train/truck.zip"
- config_name: frog_train
  data_files:
  - split: unique_split
    path: "train/frog.zip"
- config_name: no_frog_calibration
  data_files:
  - split: unique_split
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
- config_name: frog_calibration
  data_files:
  - split: unique_split
    path: "calibration/frog.zip"
- config_name: no_frog_test
  data_files:
  - split: unique_split
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
    - "test/truck.zip"
- config_name: frog_test
  data_files:
  - split: unique_split
    path: "test/frog.zip"
- config_name: no_horse_train
  data_files:
  - split: unique_split
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/ship.zip"
    - "train/truck.zip"
- config_name: horse_train
  data_files:
  - split: unique_split
    path: "train/horse.zip"
- config_name: no_horse_calibration
  data_files:
  - split: unique_split
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/ship.zip"
    - "calibration/truck.zip"
- config_name: horse_calibration
  data_files:
  - split: unique_split
    path: "calibration/horse.zip"
- config_name: no_horse_test
  data_files:
  - split: unique_split
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/ship.zip"
    - "test/truck.zip"
- config_name: horse_test
  data_files:
  - split: unique_split
    path: "test/horse.zip"
- config_name: no_ship_train
  data_files:
  - split: unique_split
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/truck.zip"
- config_name: ship_train
  data_files:
  - split: unique_split
    path: "train/ship.zip"
- config_name: no_ship_calibration
  data_files:
  - split: unique_split
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/truck.zip"
- config_name: ship_calibration
  data_files:
  - split: unique_split
    path: "calibration/ship.zip"
- config_name: no_ship_test
  data_files:
  - split: unique_split
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/truck.zip"
- config_name: ship_test
  data_files:
  - split: unique_split
    path: "test/ship.zip"
- config_name: no_truck_train
  data_files:
  - split: unique_split
    path:
    - "train/airplane.zip"
    - "train/automobile.zip"
    - "train/bird.zip"
    - "train/cat.zip"
    - "train/deer.zip"
    - "train/dog.zip"
    - "train/frog.zip"
    - "train/horse.zip"
    - "train/ship.zip"
- config_name: truck_train
  data_files:
  - split: unique_split
    path: "train/truck.zip"
- config_name: no_truck_calibration
  data_files:
  - split: unique_split
    path:
    - "calibration/airplane.zip"
    - "calibration/automobile.zip"
    - "calibration/bird.zip"
    - "calibration/cat.zip"
    - "calibration/deer.zip"
    - "calibration/dog.zip"
    - "calibration/frog.zip"
    - "calibration/horse.zip"
    - "calibration/ship.zip"
- config_name: truck_calibration
  data_files:
  - split: unique_split
    path: "calibration/truck.zip"
- config_name: no_truck_test
  data_files:
  - split: unique_split
    path:
    - "test/airplane.zip"
    - "test/automobile.zip"
    - "test/bird.zip"
    - "test/cat.zip"
    - "test/deer.zip"
    - "test/dog.zip"
    - "test/frog.zip"
    - "test/horse.zip"
    - "test/ship.zip"
- config_name: truck_test
  data_files:
  - split: unique_split
    path: "test/truck.zip"
---

# Dataset Specifications

Contains the entire CIFAR10 dataset, downloaded via *PyTorch*, then split and saved as `.png` files representing 32x32 images.

There a three splits, perfectly balanced class-wise:

- `train`: 49,000 out of the original 50,000 samples from the training set of CIFAR10;
- `calibration`: 1,000 left-out samples from the training set;
- `test`: 10,000 samples, the entire original test set.

# File Structure

Files are archives `<split>/<classname>.zip`. Each `<classname>.zip` is a flat archive containing the associated images `XXXX.png`.

**For a given class**, every filename `XXXX.png` is unique, with `XXXX` ranging:

- from `"0000"` to `"0999"` for test samples,
- from `"1000"` to `"1099"` for calibration samples,
- from `"1100"` to `"5999"` for train samples.


# Use with PyTorch

As a helper, you can use the following snippet to iterate through a specific split:
```python
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("ego-thales/cifar10", name="no_bird_calibration")
dataset = dataset["unique_split"]  # Comment out if you're in case 1. below
dataloader = DataLoader(dataset.with_format("torch"), batch_size=300)
for batch in dataloader:
    # batch["images"]:    tensor with shape `(300, 3, 32, 32)` (`torch.uint8` values between 0 and 255)
    # batch["label"]:     tensor with shape `(300,)` (`torch.int64` values between 0 and 9)
    # batch["classname"]: list of length `300` with classnames as `str`
```

## Loading arguments

While [this question](https://discuss.huggingface.co/t/download-only-requested-split/165968/3?u=ego-thales) does not find a reasonable answer, there are two cases to consider.

#### 1. If you wish to download the full dataset

- `name` (Optional): Either `"complete"` or `"no_<classname>"`.
- `split` (Optional): One of `"train"`, `"calibration"` or `"test"`. If `name` is of the format `"no_<classname>"`, then the following values are also allowed: `"left_out_train"`, `"left_out_calibration"`, `"left_out_test"` and `"left_out"` (for all left-out samples).

#### 2. If you wish to only use a single split

Since apparently `streaming=True` still goes through every archive to find metadata no matter the setting, as a workaround, we defined many different configuration to act as splits. So use:

- `name`: One of `"<classname>_<split>"` or `"no_<classname>_<split>"` with `<split>` either `"train"`, `"calibration"` or `"test"`.
- `split`: Leave empty.

## Good to know

The `.zip` archives are unzipped at every iteration where they are needed. As such, we recommend using an adapted `batch_size`, accounting for the number of samples in each `data.zip`.

Output:
{
    "extracted_code": "from datasets import load_dataset\nfrom torch.utils.data import DataLoader\n\ndataset = load_dataset(\"ego-thales/cifar10\", name=\"no_bird_calibration\")\ndataset = dataset[\"unique_split\"]  # Comment out if you're in case 1. below\ndataloader = DataLoader(dataset.with_format(\"torch\"), batch_size=300)\nfor batch in dataloader:\n    # batch[\"images\"]:    tensor with shape `(300, 3, 32, 32)` (`torch.uint8` values between 0 and 255)\n    # batch[\"label\"]:     tensor with shape `(300,)` (`torch.int64` values between 0 and 9)\n    # batch[\"classname\"]: list of length `300` with classnames as `str`"
}
