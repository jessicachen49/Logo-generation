# Logo generation using conditional styleGAN and MSG-styleGAN

The source code of LoGANv2 is in the ConditionalStyleGAN folder (code from https://github.com/cedricoeldorf/ConditionalStyleGAN) and the source code of MSG-styleGAN is in the msg-stylegan-tf folder (code from https://github.com/akanimax/msg-stylegan-tf).

# To run the code

1. Clustering logo images using kmeans and creating pickle file for training:

run preprocessing.py

2. Training ConditionalStyleGAN:

run python dataset_tool.py create_from_images dataset/logos 'path-to-logo-dataset' 1
change hyperparameters in networkstylegan.py if neccessary
run python train.py

3. Training MSG-styleGAN:

prepare tfrecord dataset if not done yet (would be the same tfrecord dataset as ConditionalStyleGAN)
change directory of label_file in line 48 of dataset.py to labels tfrecord dataset
change hyperparameters in networkstylegan.py if neccessary
run python train.py

4. Evaluation:

change network_pkl to path to your evaluation network in line 80 of run_metrics.py run python run_metrics.py

# Future Directions
1. Two step clustering (1) cluster using text embedding (2) cluster using image feature
2. Use style-mixing or other techniques to integrate real-world objects
