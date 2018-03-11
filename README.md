# Single image depth estimation by dilated deep residual convolutional neural network and soft-weight-sum inference

This repo is an implementation of this paper: https://arxiv.org/pdf/1705.00534.pdf

Explained better here: https://arxiv.org/pdf/1708.02287.pdf

Runs: 
 - 2018-03-08--16-41-31 initial with correct implementation, with batch normalization, no weights decay, with weights regularization
 - 2018-03-09--17-23-46 hopefully correct learning rate decay, xavier initialization, with weights regularization
 - 2018-03-10--01-02-25 xavier, no weights regularization
 - 2018-03-10--11-03-09 same as above, 40 images, trying to overfit
 - 2018-03-11--02-04-09 only 100 depth bins, using softmax also for inference, learning rate 1e-4 (as above), decay is *0.9
 - 2018-03-11--14-40-26 same as above, but initial learning rate is 1e-5
 - 2018-03-11--15-30-10 same as above, but initial learning rate is 1e-4, decay is correct, *0.1 (dividing by 10)
 - 2018-03-11--16-59-14 same as above, but loss from the other paper, Estimating Depth from Monocular Images... tried