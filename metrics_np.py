import numpy as np


def accuracy_under_treshold(truth_img, predicted_img, treshold=1.25):
    n_pixels = np.prod(truth_img.shape)
    delta = np.maximum(predicted_img / truth_img, truth_img / predicted_img)
    lower = delta < treshold
    ratio = np.sum(np.cast(lower, np.float32)) / n_pixels
    return ratio


def mean_relative_error(truth_img, predicted_img):
    n_pixels = np.prod(truth_img.shape)
    rel = np.abs(predicted_img - truth_img) / truth_img
    rel = np.sum(rel) / n_pixels
    return rel


def root_mean_squared_log_error(truth_img, predicted_img):
    n_pixels = np.prod(truth_img.shape)
    rms = (np.log10(predicted_img) - np.log10(truth_img)) ** 2
    rms = np.sqrt(np.sum(rms) / n_pixels)
    return rms


def root_mean_squared_error(truth_img, predicted_img):
    n_pixels = np.prod(truth_img.shape)
    rms = (predicted_img - truth_img) ** 2
    rms = np.sqrt(np.sum(rms) / n_pixels)
    return rms


def log10_error(truth_img, predicted_img):
    n_pixels = np.prod(truth_img.shape)
    lg10 = np.abs(np.log10(predicted_img) - np.log10(truth_img))
    lg10 = np.sum(lg10) / n_pixels
    return lg10
