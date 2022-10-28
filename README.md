# ___DeepLearning-Template-Code___

## ___Catalogue___
* Intro
* Diagram
* Module and function
* Getting started
* To Do List



## ___PART 1. Introduction___

Hi, there. This project's name is deep learning template code, 
which means it provides a model or a template for simply training your deep learning
algorithm. The purpose of this project is to simplify the process of
deep learning, __it aims at covering all of necessary steps during
an experiment__. It requires no extra movements, all you need to do
is just __preparing your dataset, focus on neural networks framework and tune the parameters to
get the best result__. More importantly, you can use this template in both windows and macos environment
(running on cpu).

In the following content of this __README__ file, I'll introduce every implement details of how this
template actually work to help you have a complete understanding of this project, so that you could 
modify it in the way that you want.

Part 2 will sketch an overall image of different modules in this project which offers you a quick 
understanding. Part 3 will explain the function for each directory and explain its API. Part 4 gives
you a brief introduction of how to use with your own models and dataset. Part 5 is the project's to
 do list. So, let's dive into it!
 
## ___PART 2. Diagram___
![](./readme_files/framwork.png)

## ___PART 3. Module and function___
###  ___Directory baseline___
Directory baseline provides two files, `base_data_load.py` and `base_trainer.py`. `base_data_load.py` provides basic
dataloader class, while `base_trainer.py` provide basic trainer class.

For `base_data_load.py`:
> __Class name__: BaseDataLoader
> 
> __Params__: \
> `dataset`: Warped data by Pytorch Dataset class. \
> `batch_size`: Batch size of data. \
> `shuffle`: Binary value, shuffle the dataset or not. \
> `num_workers`: 1 as default.
> 
> __Methods__: \
> `log_record`: Record logger information of dataloader on console and logger file. \
> `length`: Compute the number of dataset.

For `base_trainer.py`:
> __Class name__: BaseTrainer
> 
> __Params__: \
> `model`: The neural network model that you created. \
> `epoch`: The number of epoch. \
> `data_loader`: The data loader that you created. \
> `optimizer`: The optimizer that you choose. \
> `checkpoint_enable`: A binary value. Enable the checkpoint or not. \
> `device`: Running on GPU or CPU.
> 
> __Methods__: \
> `_epoch_train`: Abstract method. The logic of each epoch during training process. \
> `_epoch_val`: Abstract method. The logic of each epoch after training, to test model performance. \
> `_train`: Overall training procedure. Call `_epoch_train` and `_epoch_val` for each epoch. \
> `save_model`: Save the model for the specific epoch. \
> `checkpoint_generator`: Generate the necessary information for checkpoint. \
> `load_model`: Load the model from the checkpoint (Only used for checkpoint function).

### ___Directory data_loader___

This is the place where you need to define your self dataloader.
For instance, dataset loction or transform.

### ___Directory database___

It will restore your dataset through data_loader.

### ___Directory logger___

By changing the content or parameters in log_config.json, you
could change formatter, logger and handlers. This file will be
parsed by logger_parser.

### ___Directory model___

Backbone is for restoring your models.
Layers is for defining your own layers.

### ___Directory readme_files___

Nothing but imgs.

### ___Directory trainer___

Self-defined trainers based on basic class.

### ___Directory utils___

It mainly contains two things. First, you could write any function
here as tools. Second, you could define your own optimizers or loss
function here.

### config.json
Change and add your params here directly.


### 4. Getting started

So, how you gonna use it? All you need to do is add your creative
model to backbone, your dataset to database and go to train.py.
Click 'run' button, and wait till the end of learing!




## To Do List
- [X] Add base class for data-loading process and training process
- [x] Add data loaders module
- [x] Build a basic neural network for testing
- [x] Add trainers
- [x] Add config file and config file parser
- [x] Add log system
- [x] Add model save and reload function
- [x] Add checkpoint to base class
- [ ] Enhance code expandability and robustness
- [ ] Fix all bugs

