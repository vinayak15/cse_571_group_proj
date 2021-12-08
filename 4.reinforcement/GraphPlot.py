import matplotlib.pyplot as plt
from os.path import exists

import json
def plot(rewards, string = None ):
    array = []
    sum = 0
    # print(len(rewards))

    # for i, reward in enumerate(rewards):
    #     if (i+1)%10==0:
    #         sum=sum+reward
    #         sum=sum/20
    #         array.append(sum)
    #         sum=0
    #     else:
    #         sum =sum+reward
    plt.title(string)
    string = '/home/vinayak/AI/anova powerClassic ' + string + '.json'
    # print(string)
    # file_exists = exists(string)
    # if file_exists:
    #     f = open(string, 'r')
    #     array = f.read()
    #     f.close()
    #     array =array.split('[')[1]
    #     array = array.split(']')[0]
    #     array = array.split(',')
    #     for i, value in enumerate(rewards):
    #         rewards[i] = rewards[i] +float(array[i])
    # f = open(string,'a')
    # json.dump(rewards, f)
    # f.close()
    # plt.plot(rewards)  # plotting by columns
    # print(array)
    # if len(array) > 0:
    #     ax.set_ylim([0, max(array) * 1.2])
    # plt.title(string)
    # plt.show()