from matplotlib import pyplot as plt

def plot_all_acc(run_no, ifRand = True):
    with open("AL_Results/run" + str(run_no) + "/k_vals.txt", "r") as f:
        X = f.read()[1:-1].split(',')
    X.insert(0,0)
    X = [int(i) for i in X]
    for i in range(len(X)):
        if i!=0:
            X[i] = int(X[i-1]) + int(X[i])
    with open("AL_Results/run" + str(run_no) + "/accuracy.txt", "r") as f:
        y_algo = f.read()[1:-1].split(',')
    y_algo = [float(i) for i in y_algo]
    y_algo = [round(i, 3) for i in y_algo]
    plt.title(f"Results of trial {run_no}")
    plt.plot(X, y_algo, label="Algorithm Accuracy")
    plt.ylim(0.84,0.89)
    plt.xlabel("Number of added training instances")
    plt.ylabel("Accuracy")
    if ifRand:
        with open("AL_Results/run" + str(run_no) + "/rand_accuracy.txt", "r") as f:
            y_rand = f.read()[1:-1].split(',')
        y_rand = [float(i) for i in y_rand]
        y_rand = [round(i, 3) for i in y_rand]
        plt.plot(X, y_rand, label="Random Accuracy")
    plt.legend()
    plt.savefig(f'for research paper/run{run_no}_acc.jpg')
    plt.close()

plot_all_acc(1)
plot_all_acc(2)
plot_all_acc(3, ifRand=False)