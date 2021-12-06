import matplotlib.pyplot as plt

def plot(rewards, string = None ):
    array = []
    sum = 0
    # print(len(rewards))

    for i, reward in enumerate(rewards):
        if (i+1)%20==0:
            sum=sum+reward
            sum=sum/20
            array.append(sum)
            sum=0
        else:
            sum =sum+reward


    plt.plot(array)  # plotting by columns
    ax = plt.gca()
    # print(array)
    if len(array) > 0:
        ax.set_ylim([0, max(array) * 1.2])
    plt.title(string)
    plt.show()