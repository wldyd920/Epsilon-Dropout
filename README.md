2023/04/10  
Currently only Weight Dropout is available.  
With this code, we can easily drop a layer's weights.  
You can run simple implementation code here.  
  
https://colab.research.google.com/drive/1o739LKrmxg5pLC4kiXBKoQZQJF5eEHqw?usp=sharing  
  
2023/04/12  
Main update: Run function and Plot function.  
Experiments: Runed 1000 epochs for {(MNIST, CIFAR-10) X (NoDrop, Drop, WeightDrop)}   
Result: No Drop had highest accuracies for both datasets.  
   For CIFAR-10, Normal Dropout(Node drop) had serious problem with training.  
   And Weight Dropout had more unstable convergence compared to No Dropout.
   
2023/04/13
Main updates:   
1) Changed architecture to VGG-like architectures. (conv-conv-pool) - (conv-conv-pool) - (flatten-fc-fc-fc) - out  
2) Learning rate: 0.1 > 0.01 (last experiment had issue for training Dropout architecture. Assuming the reason was gradient explode)
3) Nomalization for dataset before train: 0.5 > reasonable values
4) Record not only plots but also loss and accuracies.  

Will be uploaded tomorrow after the training is done.
   
Thank you.
