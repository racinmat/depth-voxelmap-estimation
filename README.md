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
 - 2018-03-13--03-52-42 whole nyu dataset with the above mentioned dataset
 - 2018-03-19--04-14-04 whole GTA dataset, but with incorrect depth loaded
 - 2018-03-26--19-25-51 whole GTA dataset, with fixed depth loading and preprocessing
 - 2018-03-28--13-55-48 trying to overfit on 40 GTA images
 - 2018-03-28--22-24-23 again, overfitting, with correct metrics and ground truth depth dumping
 - 2018-03-29--00-14-08 again, overfitting, with correct metrics, ground truth depth dumping, and hopefully synced input and output images in tensorboard
 - 2018-03-29--12-41-37 whole GTA dataset, now with correct metrics and tensorboard visualization, and correctly split training and testing set

dgs s momentem a nesteroff momentem je lepší než adam
 In train-nyu and test-nyu, data from train are split in ratio 80:20
 
 
 report of accuracies:
 +--------------------+----------------------+----------------------+---------------------+----------------------+----------------------+
|   treshold_1.25    |     mean_rel_err     |         rms          |       rms_log       |      log10_err       |         name         |
+--------------------+----------------------+----------------------+---------------------+----------------------+----------------------+
|     0.98328125     | 0.044797583685980906 | 0.025788567528874328 | 0.03040171146706071 | 0.018311002519395617 | 2018-03-11--23-23-32 |
| 0.9387152777777777 |  2.1360649956597224  |  1.6739856387785428  |  0.3718028906000468 | 0.10152558220757378  | 2018-03-11--15-30-10 |
| 0.9305729166666666 |  2.229178466796875   |  1.7460586196130414  | 0.37347099263959904 | 0.10319221496582032  | 2018-03-11--14-40-26 |
+--------------------+----------------------+----------------------+---------------------+----------------------+----------------------+


Tensorboard:
inspecting it single run:
tensorboard --inspect --logdir=logs/2018-03-28--10-46-41 

inspecting single value in run:
tensorboard --inspect --logdir=logs/2018-03-28--10-46-41 --tag=cost
