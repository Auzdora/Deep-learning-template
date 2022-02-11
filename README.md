# DeepLearning-Template-Code

## Catalogue
* Intro
* Diagram
* Module and function
* Getting started
* To Do List


## Content
### 1. Introduction
&ensp;&ensp;&ensp;&ensp;
Hi, there. This project's name is deep learning template code, 
which means it provides a model or a template or framework, if
you wanna call it, for deep learning experiments or deep learning
project. The purpose of this project is to simplify the process of
deep learning, __it aims at covering all of necessary steps during
an experiment__. It requires no extra movements, all you need to do
is just __focus on neural network's frame and tune the parameters to
get the best result__.

&ensp;&ensp;&ensp;&ensp;
Next, diagram part introduces the whole framework for your guys
to understand how this project been built.

&ensp;&ensp;&ensp;&ensp;
Module and function part describes every directory's function and
utility. So you can add more based on these.

&ensp;&ensp;&ensp;&ensp;
How to use part demonstrates how you can implement this during
a project.

### 2. Diagram (ps, recommande use light mode of web so that you can see it clearly, sorry for that..)

![](./readme_files/framwork.png)
&ensp;&ensp;&ensp;&ensp;
Logger directory provide simple files to control the whole project
log record. Many log commands have been written to the different
part of code.

&ensp;&ensp;&ensp;&ensp;
Database has two directories. One is train_data, other is test_data.
It depends on your original dataset.

&ensp;&ensp;&ensp;&ensp;
Utils provides many useful tools for coding. It also has optimizers, 
loss function and some json utils.

&ensp;&ensp;&ensp;&ensp;
Model has backbone which you could define your innovational model
here and layers which you could define your model's layer.

&ensp;&ensp;&ensp;&ensp;
Baseline provides basic class for data_loader and trainer.

&ensp;&ensp;&ensp;&ensp;
Data_loader will load data to database. Trainer provide a class
for training process.

### 3. Module and function
#### Baseline
&ensp;&ensp;&ensp;&ensp;
Baseline provides basic elements you need to implement a neural
network project for data loader and trainer. Specifically, for 
trainer, you need to rewrite _**_epoch_train**_ and **__epoch_vall_** methods.
These two files require no need to change if you just wanna use
this project, also you could change it based on your purpose.

#### data_loader
&ensp;&ensp;&ensp;&ensp;
This is the place where you need to define your self dataloader.
For instance, dataset loction or transform.

#### database
&ensp;&ensp;&ensp;&ensp;
It will restore your dataset through data_loader.

#### logger
&ensp;&ensp;&ensp;&ensp;
By changing the content or parameters in log_config.json, you
could change formatter, logger and handlers. This file will be
parsed by logger_parser.

#### model
&ensp;&ensp;&ensp;&ensp;
Backbone is for restoring your models.

&ensp;&ensp;&ensp;&ensp;
Layers is for defining your own layers.

#### readme_files
&ensp;&ensp;&ensp;&ensp;
Nothing but imgs.

#### trainer
&ensp;&ensp;&ensp;&ensp;
Self-defined trainers based on basic class.

#### utils
&ensp;&ensp;&ensp;&ensp;
It mainly contains two things. First, you could write any function
here as tools. Second, you could define your own optimizers or loss
function here.

#### config.json
Change and add your params here directly.


### 4. Getting started
&ensp;&ensp;&ensp;&ensp;
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

