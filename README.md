# 1D-Temporal-Convlution-with-Pytorch
A simple project that creates a model for temporal convolution.

In order to train the 1D CNN on data, you need to first create the files of signals by running the following code:

python gen_time_serie_for_cnn.py

This code generates two types of simple signals.
Class 0: One type consists of random noise and one sharp dent heading up and one soft dent heading down in random places.
Class 1: And the other type, consists of two sharp dents next to each other.

The two files will be created:
time_series_x_100x.pt
time_series_y_100x.pt

The file time_series_x_100x.pt includes 1000 sequences of size 1000. Half of which is class 0 and the other half is class 1.
The file time_series_y_100y.pt contains 1000 numbers as class ofthe corresponding time series of the file mentioned above.

Now you can simply run the following code and see the performance:

python sin_wave_cnn.py
