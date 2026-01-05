Digits Recognition

* Used MNIST standard dataset for training
* Some test images I used to check the performance of the model are also there
* Architecture of the model is:
  
  Conv layer(32)->MaxPooling->Conv layer(64)->MaxPooling->Flatten ((5,5,64) : 1600 neurons)->128 neurons hidden layer->10 neurons output layer
* Used Adam Optimizer along with sparse_categorical_crossentropy loss function
* Compared it with SGD(n=0.01 and 0.1), with 0.1 learning rate, it surpassed adam model by some margin
