# Preferred Networks 2019 Internship Machine Learning Task

These are the codes to solve the problems specified in Preferred Networks 2019 Internship's Machine Learning screening task. The codes consist of 3 main parts:
- ```gnn.py``` - The main file in which the Graph Neural Network (GNN) model classes and training/evaluation routines are located. 
- ```utils.py``` - A utility file for building training/evaluation data generator.
- ```test.py``` - A file consisting of several unit tests to test the correctness of model classes methods.
- ```make_graph.py``` - Visualize the changes of training and validation accuracy and losses using graphs, and save the corresponding figures.

## Requirements

- Python 3.7.2
- NumPy 1.16.2
- Matplotlib 3.0.3
- tqdm 4.28.0 - For visualizing progress bar of a loop routine (in this case, the progress of training/evaluation)

## Theme selected for 'Problem 4'

Implementing Adam[1] optimizer.

## Running

Go to ```/src/``` and run:

```
python gnn.py [-h] [-b BATCH_SIZE] [-e EPOCH] [-t TYPE] [-sml]
              [-sm SAVE_MODEL] [-lm LOAD_MODEL] [-po]
            
optional arguments:
  -h, --help            show help message and exit
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -e EPOCH, --epoch EPOCH
  -t TYPE, --type TYPE  Type of algorithm to run: sgd - Vanilla SGD, msgd -
                        Momentum SGD, adam - Adam
  -sml, --see_mb_loss   See per-minibatch average loss.
  -sm SAVE_MODEL, --save_model SAVE_MODEL
                        Save model every n epoch in directory called 'models'.
                        (n=0 means not saving)
  -lm LOAD_MODEL, --load_model LOAD_MODEL
                        Load model saved in a directory.
  -po, --predict_only   Predict test data only. (no training)
```

For example, running:

```
python gnn.py -b 8 -e 50 -t msgd -sm 5
```

will run a training routine with the following options:
- Batch size of 8.
- Train until 50th epoch.
- Use Momentum SGD to train model.
- Save model every 5 epochs.

To reproduce the results shown in the report, run the following commands sequentially:
```
python gnn.py -b 2 -e 50 -t sgd -sm 1
python gnn.py -b 4 -e 50 -t sgd -sm 1
python gnn.py -b 8 -e 50 -t sgd -sm 1
python gnn.py -b 16 -e 50 -t sgd -sm 1
python gnn.py -b 32 -e 50 -t sgd -sm 1
python gnn.py -b 64 -e 50 -t sgd -sm 1
python gnn.py -b 128 -e 50 -t sgd -sm 1
python gnn.py -b 2 -e 50 -t msgd -sm 1
python gnn.py -b 4 -e 50 -t msgd -sm 1
python gnn.py -b 8 -e 50 -t msgd -sm 1
python gnn.py -b 16 -e 50 -t msgd -sm 1
python gnn.py -b 32 -e 50 -t msgd -sm 1
python gnn.py -b 64 -e 50 -t msgd -sm 1
python gnn.py -b 128 -e 50 -t msgd -sm 1
python gnn.py -b 2 -e 50 -t adam -sm 1
python gnn.py -b 4 -e 50 -t adam -sm 1
python gnn.py -b 8 -e 50 -t adam -sm 1
python gnn.py -b 16 -e 50 -t adam -sm 1
python gnn.py -b 32 -e 50 -t adam -sm 1
python gnn.py -b 64 -e 50 -t adam -sm 1
python gnn.py -b 128 -e 50 -t adam -sm 1
```
and then use one of the models saved in ```/src/models/gnn_adam/``` by running:
```
python gnn.py -lm ./models/gnn_adam/[model_name].pkl -po
```
to produce the predictions for test data, which will be saved in ```/predictions.txt```. For ease of reproduction, the model used to predict the dataset is included (```e50_b2_va64.25.pkl```). To produce the graphs shown in the report, execute:
```
python make_graph.py
```
which will access saved losses and accuracy history, yielding 3 pngs which show graphs that correspond to each training algorithms.

Note:
- The code assumes that the dataset is located on ```/datasets/```
- Training hyperparameters are currently hard-coded to the model class itself, and changes to the hyperparameters require one to edit the code itself. The hyperparameters are currently set to the recommended setting specified in the task guidelines.
- Training and validation data are split to 80:20 ratio.
- When running the code for the first time, the code will produce 2 numpy files (```train_ids.npy``` and ```valid_ids.npy```) which specify the indices for training data and validation data. These files will then be loaded on the next run to ensure a fair comparison between different batch sizes and training algorithms. To produce a fresh batch of indices, one can delete these files and run the code again. Also, for easier reproduction, the indices used in the report has been included in ```/src/```.
- Perfect reproduction is impossible because of the inherent stochastic features of the algorithms (initialization of weights using Gaussian distribution, random sampling of the mini-batches), but repeated experiments will produce similar results.

## Testing

Running:
```
python test.py
```
will run each unit test inside the code to test the correctness of the methods specified in model classes. This also corresponds to the task described in 'Problem 1'.

## Author

Calvin Janitra Halim - <corronade@gmail.com>

## References

[1] Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." ICLR 2014.
