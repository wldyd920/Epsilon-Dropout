# Update Note
2023/04/10  
Weight_Dropout 1.ipynb  
Currently only Weight Dropout is available.  
With this code, we can easily drop a layer's weights.  
You can run simple implementation code here.  
  
https://colab.research.google.com/drive/1o739LKrmxg5pLC4kiXBKoQZQJF5eEHqw?usp=sharing  
  
2023/04/12  
Weight_Dropout 2.ipynb  
Main update: Run function and Plot function.  
Experiments: Runed 1000 epochs for {(MNIST, CIFAR-10) X (NoDrop, Drop, WeightDrop)}   
Result: No Drop had highest accuracies for both datasets.  
   For CIFAR-10, Normal Dropout(Node drop) had serious problem with training.  
   And Weight Dropout had more unstable convergence compared to No Dropout.
   
2023/04/13  
Weight_Dropout(VGG-like) 3.ipynb  
Main updates:   
   Changed architecture to VGG-like architectures. (conv-conv-pool) - (conv-conv-pool) - (flatten-fc-fc-fc) - out  
   Learning rate: 0.1 > 0.01 (last experiment had issue for training Dropout architecture. Assuming the reason was gradient explode)  
   Nomalization for dataset before train: 0.5 > reasonable values  
   Record not only plots but also loss and accuracies. (exp_num = -1 to not record)  

2023/04/14  
Weight_Dropout(more dataset) 4.ipynb  
Main updates:  
   Added 2 more datasets to compare. (Fashion_MNIST, CIFAR-100)  
   Results of the experiments are in Result4 folder
  
2023/04/18  
SMD1.ipynb  
Implementation of Shape-Memory Dropout. (Drops node)  
experiment has been done on FMNIST.  
save: (mask during training, accuracy of training batch)  
use : top_k mask (k=1) after half of the total_epoch  
Result: Fashion_MNIST: 92.01%  
  
2023/04/18  
SMD2.ipynb  
Changed which accuracy to save, where to use.  
save: (mask during training, loss when test)  
use : top_k mask (k=10), Use the saved masks based on the test accs.  
Result: Fashion_MNIST: 90.69%  
  
2023/04/18  
Problem: OOM(cannot save all the masks)  

2023/04/18   
Totally new idea : Control regularization by adjust the dropout ratio with momentum.
  
2023/04/18  
Debatable ideas  
Firstly, there can be two different ways of evaluate the masks:  
1) Train accuracy: Evaluate with pure training performance.  
2) Test accuracy : Unlearned objective evaluation.  
  
Secondly, there can be two different ways of use the saved masks:  
1) Epoch      : Possible to update continuously with Epsilon.  
2) Validation : Masks can be used purely for train purposes. (Possible to open all the nodes when test)  
  
For Last, there can be two different ways of choose masks when test:  
1) Open all the nodes : This is the same as original Dropout. (Regularization)  
2) Random choose : This is similar to Evolutionary approach. (Search)  


  
Thank you.  


# TODO  
1. How to Drop Weights with knowledge?  
 : By saving the masks of Dropout  
2. More Datasets?  
 : Fasion-MNIST, CIFAR-100, ImageNET(ILSVRC2012 task 1&2)  
3. Adaptable to other architectures?  
 : Try other famous architectures (VGG, GoogleNet, etc.)   
4. Effective also in Attention Machanism?  
 : After showing the effectiveness  
     
       

