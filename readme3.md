# logo generation using LoGANv2 (conditional styleGAN) & conditional MSG-styleGAN

The source code of LoGANv2 is in the ConditionalStyleGAN folder and the source code of MSG-styleGAN is in the msg-stylegan-tf folder.

# Clustering logo images using kmeans and creating pickle file for training
run preprocessing.py

# Training ConditionalStyleGAN
1. run python dataset_tool.py create_from_images dataset/logos 'path-to-logo-dataset' 1
2. change hyperparameters in networkstylegan.py if neccessary
3. run python train.py

# Training MSG-styleGAN
1. prepare tfrecord dataset if not done yet (would be the same tfrecord dataset as ConditionalStyleGAN)
2. change directory of label_file in line 48 of dataset.py to labels tfrecord dataset
3. change hyperparameters in networkstylegan.py if neccessary
4. run python train.py

# Evaluation (same for both models)
change network_pkl to path to your evaluation network in line 80 of run_metrics.py
run python run_metrics.py

future direction: 
1. two step clustering 1. cluster using text embedding 2. cluster using image feature
2. use style-mixing or other techniques to integrate real-world objects 