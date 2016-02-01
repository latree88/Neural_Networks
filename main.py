import random
from string import ascii_uppercase
import numpy as np
from random import randint
import copy
import math
import matplotlib.pyplot as plt
import time


source = []
training_set = []
testing_set = []
ipt_to_hn_weight = []
hn_to_opt_weight = []
shuffled_training_set = []
shuffled_testing_set = []
acc_training_set_list = []
acc_testing_set_list = []
pre_delta_ipt_to_hn = []
pre_delta_hn_to_opt = []
plot_x = []

hidden_units = 4
num_inputs = 16
num_outputs = 26
learning_rate = 0.3
momentum = 0.3
total_epoch = 10


alphabet = 0
# declare all the variables I need
for alphabet in range(num_outputs):
    source.append([])
    training_set.append([])
    testing_set.append([])


# read data into three dimensional lists
def read_data(the_list):
    with open('letter-recognition.data', 'r') as f:
        while True:
            one_line = f.readline()
            if not one_line:
                break
            one_data_set = one_line.split(',', 16)

            # remove the letter from the list
            letter = one_data_set.pop(0)

            # charIntValue is the ascii integer value for the letter
            letter_int_value = ord(letter)

            # remove the new line element from last element in list
            one_data_set[-1] = one_data_set[-1].strip()

            # add this set value to the list
            the_list[letter_int_value-65].append(one_data_set)


# split data in two parts one is training set, another is testing set
# size try to be half and half
def split_data(the_list, split_list1, split_list2):
    for i in range(num_outputs):
        half_size = len(the_list[i])/2
        size = len(the_list[i])
        for j in range(size):
            if j < half_size:
                split_list1[i].append(the_list[i][j])

            else:
                split_list2[i].append(the_list[i][j])


# convert the data from string to integer
def convert_to_int(to_convert_list):
    for i in range(num_outputs):
        size = len(to_convert_list[i])
        for j in range(size):
            length = len(to_convert_list[i][j])
            for k in range(length):
                to_convert_list[i][j][k] = int(to_convert_list[i][j][k])


# two list for storing all the data that need to calculate mean and std
temp_mean_list = []
temp_std_list = []
temp_mean = []
temp_std = []
for feature_index in range(num_inputs):
    temp_mean_list.append([])
    temp_std_list.append([])


def get_mean_and_std_list(to_scale, the_mean_list, the_std_list):
    for i in range(len(to_scale)):
        for j in range(len(to_scale[i])):
            for k in range(len(to_scale[i][j])):
                temp_mean_list[k].append(to_scale[i][j][k])
                temp_std_list[k].append(to_scale[i][j][k])


def get_mean_and_std(the_mean_list, the_std_list, the_mean, the_std):
    for i in range(num_inputs):
        the_mean.append(float(np.mean(the_mean_list[i])))
        the_std.append(float(np.std(the_std_list[i])))


# scale the data by using the equation of (x-mean)/std
def preprocess_data(to_scale, the_mean, the_std):
    for i in range(len(to_scale)):
        for j in range(len(to_scale[i])):
            for k in range(len(to_scale[i][j])):
                to_scale[i][j][k] = (to_scale[i][j][k] - the_mean[k]) / the_std[k]


def init_weight(to_init1, to_init2):
    # declare the weights from input to hidden units
    for i in range(num_inputs + 1):
        to_init1.append([])
    for i in range(num_inputs + 1):
        for j in range(hidden_units):
            to_init1[i].append([])

    # init the weights value, last column of elements is bias' weight!!!!!!!!!!!!!!!!!!!!!!!!
    for i in range(num_inputs + 1):
        for j in range(hidden_units):
            to_init1[i][j] = random.uniform(-0.25, 0.25)

    # declare the weights from hidden units to output
    for i in range(hidden_units + 1):
        to_init2.append([])
    for i in range(hidden_units + 1):
        for j in range(num_outputs):
            to_init2[i].append([])

    # init the weights value, last column of elements is bias' weight!!!!!!!!!!!!!!!!!!!
    for i in range(hidden_units + 1):
        for j in range(num_outputs):
            to_init2[i][j] = random.uniform(-0.25, 0.25)


# shuffle the sorted data set
# last element of shuffled_training_set will be the key for that data set
# eg. if it is 0 it will be an 'A' data set
# notice the length of the set will change!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def shuffle_data_set(the_data_set, the_shuffled_set):
    temp = copy.deepcopy(the_data_set)
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            temp[i][j].append(i)

    for i in range(len(temp)):
        for j in range(len(temp[i])):
            the_shuffled_set.append(temp[i][j])
    random.shuffle(the_shuffled_set)


def sigmoid(the_input):
    return 1/(1 + math.exp(-the_input))


# get the hidden layer value
def calculate_h(one_set, weight):
    temp_h_list = []
    temp_h = 0
    for i in range(hidden_units):
        for j in range(num_inputs):
                temp_h += weight[j][i] * one_set[j] + weight[num_inputs][i]
        temp_h = sigmoid(temp_h)
        temp_h_list.append(temp_h)
    return temp_h_list


# get the output layer value
def calculate_o(the_h_list, weight):
    temp_o_list = []
    temp_o = 0
    for i in range(num_outputs):
        for j in range(len(the_h_list)):
            temp_o += weight[j][i] * the_h_list[j] + weight[len(the_h_list) - 1][i]
        temp_o = sigmoid(temp_o)
        temp_o_list.append(temp_o)
    return temp_o_list


def find_max_index(the_list):
    max_num = max(the_list)
    max_list = []
    for i in range(len(the_list)):
        if the_list[i] == max_num:
            max_list.append(i)
    if len(max_list) == 1:
        return max_list[0]
    else:
        return max_list[randint(0, (len(max_list)-1))]


def get_acc(first_weight, second_weight, the_shuffled_set):
    count = 0
    for i in range(len(the_shuffled_set)):
        temp_h_list = calculate_h(the_shuffled_set[i], first_weight)
        temp_o_list = calculate_o(temp_h_list, second_weight)
        max_index = find_max_index(temp_o_list)
        if max_index == the_shuffled_set[i][num_inputs]:
            count += 1
    # print count
    # print float(len(the_shuffled_set))
    return count/float(len(the_shuffled_set))


def zero_list_ipt_to_hn(the_pre_delta):
    for i in range(num_inputs + 1):
        the_pre_delta.append([])
        for j in range(hidden_units):
            the_pre_delta[i].append([])

    for i in range(num_inputs + 1):
        for j in range(hidden_units):
            the_pre_delta[i][j] = 0.0


def zero_list_hn_to_opt(the_pre_delta):
    for i in range(hidden_units + 1):
        the_pre_delta.append([])
        for j in range(num_outputs):
            the_pre_delta[i].append([])

    for i in range(hidden_units + 1):
        for j in range(num_outputs):
            the_pre_delta[i][j] = 0.0


# pass in the shuffled_training_set !!!!!!!!!!!!!!!
def train(first_weight, second_weight, the_shuffled_training_set, the_shuffled_testing_set,
          the_training_acc_list, the_testing_acc_list, the_pre_delta_ipt_hn, the_pre_delta_hn_opt):
    acc_training_set = 0
    # acc_testing_set = 0
    num_epoch = 0
    while acc_training_set != 1 and num_epoch <= total_epoch:
        for i in range(len(the_shuffled_training_set)):
            # propagate the input forward
            temp_h_list = calculate_h(the_shuffled_training_set[i], first_weight)            # hj
            temp_o_list = calculate_o(temp_h_list, second_weight)                   # ok

            # calculate the error terms
            temp_error_opt_list = []
            # target = 0
            for j in range(num_outputs):
                if the_shuffled_training_set[i][num_inputs] == j:
                    target = 0.9
                else:
                    target = 0.1
                temp_error_opt = temp_o_list[j] * (1 - temp_o_list[j]) * (target - temp_o_list[j])
                temp_error_opt_list.append(temp_error_opt)

            temp_error_hidden_list = []
            weight_opt_sum = 0
            for k in range(hidden_units):
                for m in range(num_outputs):
                    weight_opt_sum += second_weight[k][m] * temp_error_opt_list[m]
                temp_error_hidden = temp_h_list[k] * (1 - temp_h_list[k]) * weight_opt_sum
                temp_error_hidden_list.append(temp_error_hidden)

            # update the weights
            for n in range(hidden_units + 1):
                for o in range(num_outputs):
                    if n == hidden_units:
                        delta_weight_hd_opt = learning_rate * temp_error_opt_list[o] * 1 + momentum * the_pre_delta_hn_opt[n][o]
                    else:
                        delta_weight_hd_opt = learning_rate * temp_error_opt_list[o] * temp_h_list[n] + momentum * the_pre_delta_hn_opt[n][o]
                    the_pre_delta_hn_opt[n][o] = delta_weight_hd_opt
                    second_weight[n][o] += delta_weight_hd_opt

            for n in range(num_inputs + 1):
                for o in range(hidden_units):
                    if n == num_inputs:
                        delta_weight_ipt_hd = learning_rate * temp_error_hidden_list[o] * 1 + momentum * the_pre_delta_ipt_hn[n][o]
                    else:
                        delta_weight_ipt_hd = learning_rate * temp_error_hidden_list[o] * the_shuffled_training_set[i][n] + momentum * the_pre_delta_ipt_hn[n][o]
                    the_pre_delta_ipt_hn[n][o] = delta_weight_ipt_hd
                    first_weight[n][o] += delta_weight_ipt_hd

        # get acc on both training set and testing set
        acc_training_set = get_acc(first_weight, second_weight, the_shuffled_training_set)
        acc_testing_set = get_acc(first_weight, second_weight, the_shuffled_testing_set)
        the_training_acc_list.append(acc_training_set)
        the_testing_acc_list.append(acc_testing_set)
        num_epoch += 1
        print num_epoch


def get_plot_x(the_x):
    for i in range(total_epoch + 1):
        the_x.append(i)

read_data(source)
split_data(source, training_set, testing_set)
convert_to_int(training_set)
convert_to_int(testing_set)

get_mean_and_std_list(training_set, temp_mean_list, temp_std_list)
get_mean_and_std(temp_mean_list, temp_std_list, temp_mean, temp_std)
preprocess_data(training_set, temp_mean, temp_std)
preprocess_data(testing_set, temp_mean, temp_std)

init_weight(ipt_to_hn_weight, hn_to_opt_weight)

zero_list_ipt_to_hn(pre_delta_ipt_to_hn)
zero_list_hn_to_opt(pre_delta_hn_to_opt)

shuffle_data_set(training_set, shuffled_training_set)
shuffle_data_set(testing_set, shuffled_testing_set)

train(ipt_to_hn_weight, hn_to_opt_weight, shuffled_training_set, shuffled_testing_set,
      acc_training_set_list, acc_testing_set_list, pre_delta_ipt_to_hn, pre_delta_hn_to_opt)

get_plot_x(plot_x)


print acc_training_set_list

plt.plot(plot_x, acc_training_set_list)
plt.show()
plt.plot(plot_x, acc_testing_set_list)
plt.show()
