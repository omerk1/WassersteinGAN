# General
All the results mentioned in the paper can be reproduced by running the notebooks in the main repository.

# Python files
The main relevant python files are:
1. generator.py- defining the generator architecture. generator_type='MLP' defines a MLP generator, and generator_type='DCGAN' defines a deep-convolutinal generator.
2. discriminator.py- defining the discriminator architecture. model_type='DCGAN' defines a regular gan discriminator, and model_type='WGAN' defines a WGAN critic.
3. gan.py- defines the whole training procedure of regular GAN.
4. wgan.py- defines the whole training procedure of WGAN.
