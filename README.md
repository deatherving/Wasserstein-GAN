# Wasserstein-GAN
My implementation of Wasserstein Generative Network

# Notice
a. In the experiments, if we apply any batch-norm in Generator or Discriminator, then we can't train the Generator well after several times of training under discriminator.
The train steps must be:

weight_clip -> train Discriminator -> train Generator

rather than

    for t in n_critic:

      weight_clip
  
      train Discriminator
  
    train Generator

as shown in the paper.

b. If we remove the batch-norms then the generator couldn't generate reasonable images. I fonud that if we use a tanh activation on the logits of generator, then the model becames stabilized.

c. I'm working on the math of these two weired phenomenons. Paper is proposed.

f. For any questions, please send email to ervingzhou@gmail.com.
