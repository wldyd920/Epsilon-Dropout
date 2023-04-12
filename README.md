Currently only Weight Dropout is available.  
With this code, we can easily drop a layer's weights.  
You can run simple implementation code here.  
  
https://colab.research.google.com/drive/1o739LKrmxg5pLC4kiXBKoQZQJF5eEHqw?usp=sharing  
  
2023/04/12  
Update: Run function and Plot function.  
Experiments: Runed 1000 epochs for {(MNIST, CIFAR-10) X (NoDrop, Drop, WeightDrop)} has been done.  
Result: No Drop had highest accuracies for both datasets.  
   For CIFAR-10, Normal Dropout(Node drop) had serious problem with training.  
   And Weight Dropout had more unstable convergence compared to No Dropout.
  
   
Thank you.
