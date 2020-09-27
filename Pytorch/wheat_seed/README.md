# Wheat Seed Dataset
--- 
Dataset: http://archive.ics.uci.edu/ml/datasets/seeds#  

Features of three different types of wheat: Kama, Rosa, and Canadian. 70 elements each. 
Xrays were taken of each seed's kernel and the following features were recorded from the xrays:
1. area A,  
2. perimeter P,  
3. compactness C = 4*pi*A/P^2,  
4. length of kernel,  
5. width of kernel,  
6. asymmetry coefficient  
7. length of kernel groove.  
All of these parameters were real-valued continuous.

The dataset can be downloaded as a .txt file where each row contains the 7 feature values, with
the 8th value is either 1, 2, or 3 indicating its class membership. Feature are separated by white space.

TODO:
1. Read data from text file. 
2. Partition data into training and validation sets. 
3. Create model(s), train. 
4. Run on validation set. 
5. Plot results.


