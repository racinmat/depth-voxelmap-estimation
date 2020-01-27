# Single image depth estimation by dilated deep residual convolutional neural network and soft-weight-sum inference

This repository contains source code for trainingf neural networks in Matěj Račinský's master thesis, available [here](https://dspace.cvut.cz/handle/10467/76430).

Also, this repo is an implementation of this paper: https://arxiv.org/pdf/1705.00534.pdf

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
 - 2018-03-13--03-52-42 whole nyu dataset with the above mentioned setup
 - 2018-03-19--04-14-04 whole GTA dataset, but with incorrect depth loaded
 - 2018-03-26--19-25-51 whole GTA dataset, with fixed depth loading and preprocessing
 - 2018-03-28--13-55-48 trying to overfit on 40 GTA images, batch size = 4
 - 2018-03-28--22-24-23 again, overfitting, with correct metrics and ground truth depth dumping
 - 2018-03-29--00-14-08 again, overfitting, with correct metrics, ground truth depth dumping, and hopefully synced input and output images in tensorboard
 - 2018-03-29--12-41-37 whole GTA dataset, now with correct metrics and tensorboard visualization, and correctly split training and testing set
 - 2018-04-01--00-25-06 learning rate decay after 30k iterations, Adam optimizer (as in all previous cases), on 1080 Ti
 - 2018-04-01--00-26-49 learning rate decay after 30k iterations, Nadam optimizer, on Titan X
 - 2018-04-01--00-32-39 Adam optimizer, epsilon=1e-5, on titan Xp
 - 2018-04-02--02-51-28 Nadam, epsilon=1e-5
 - 2018-04-02--02-52-07 Nadam, epsilon=1e-2
 - 2018-04-02--02-59-31 Nadam, epsilon=1e-8 (default), no decaying learning rate, still 1e-4
 - 2018-04-05--09-15-19 momentum optimizer with nesterov, momentum=0.999
 - 2018-04-05--09-22-22 momentum optimizer with nesterov, momentum=0.9
 - 2018-04-22--21-01-54 training voxelmaps, momentum optimizer with nesterov, momentum=0.9
 - 2018-04-23--08-15-23 training voxelmaps, Nadam, e=1e-8
 - 2018-04-29--22-35-13 training voxelmaps, l2 loss, Nadam, e=1e-8, voxelmaps in view, depths linear in view (voxelmaps and depths linear in view, and forever onwards)
 - 2018-04-30--10-46-45 training voxelmaps, l2 loss, momentum with nesterov, momentum=0.9
 - 2018-05-01--00-20-51 training voxelmaps, logistic loss from paper, nadam
 - 2018-05-01--01-03-01 training voxelmaps, logistic loss from paper, nadam, new deconv layer, accidentally batchsize=1
 - 2018-05-04--22-57-49 training voxelmaps, logistic loss from paper correctly (with weights), nadam, new metrics
 - 2018-05-04--23-03-46 training voxelmaps, logistic loss from paper correctly (with weights), SGD with nesterov, new metrics           (from now on, logistic loss is correct)
 - 2018-05-06--00-03-04 training voxelmaps, softmax loss, SGC with nesterov, new metrics - not good
 - 2018-05-06--00-05-58 training voxelmaps, softmax loss, nadam, new metrics - not good
 - 2018-05-06--10-47-19 training voxelmaps, logistic loss, with new deconv(kernel=5,stride=1), nadam, new metrics - no activation, useless
 - 2018-05-06--10-48-08 training voxelmaps, logistic loss, with new deconv, nadam, new metrics - no activation, useless
 - 2018-05-07--17-22-10 training voxelmaps, logistic loss, with new deconv(kernel=5,stride=1,num_out=50,activation=lrelu), nadam, new metrics
 - 2018-05-08--23-37-07 training voxelmaps, logistic loss, with new deconv(kernel=2,stride=2,num_out=50,activation=lrelu), nadam, new metrics
 - 2018-05-11--00-10-54 training voxelmaps, logistic loss, with new deconv(kernel=2,stride=2,num_out=200,activation=lrelu), nadam, new metrics


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


some insights: titan X is slower than 1080 ti, roughly 2 times, Titan Xp is only slightly faster than 1080 Ti.


by htp -u racinmat I find my processes
F5 - tree visualization
by arrow I select process and kill it by F9

filter for cost and metrics: cost|positive_rate|iou|dist_on
