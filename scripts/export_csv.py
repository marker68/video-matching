import argparse
# import matplotlib.pyplot as plt
import re


def analyze(lines):
    train_losses = []
    val_losses = []
    for line in lines:
        if 'loss' in line:
            loss = line.split(' - ')[2].split(': ')[1][0:10]
            lf = re.sub('[^0-9a-zA-Z\\/@+\-:,.|#]+', '', loss)
            train_losses.append(lf)
        if 'val_loss' in line:
            loss = line.split(' - ')[3].split(': ')[1][0:10]
            lf = re.sub('[^0-9a-zA-Z\\/@+\-:,.|#]+', '', loss)
            val_losses.append(lf)
    return train_losses, val_losses


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Creating CSV")
    argparser.add_argument("log", type=str, help="log file")
    argparser.add_argument("csv_train", type=str, help="output csv file")
    argparser.add_argument("csv_val", type=str, help="output csv file")
    args = argparser.parse_args()

    with open(args.log, 'r') as f:
        lines = f.readlines()
    train_losses, val_losses = analyze(lines)
    nsteps = len(train_losses)
    # x = list(range(0,nsteps))
    # plt.plot(x, train_losses, linewidth=2.0)
    # plt.show()

    with open(args.csv_train, 'w') as f:
        for i in range(0, nsteps):
            f.write(str(i+1) + ',' + train_losses[i] + '\n')

    nsteps = len(val_losses)
    niter = len(train_losses)/nsteps
    with open(args.csv_val, 'w') as f:
        for i in range(0, nsteps):
            f.write(str(i*niter+1) + ',' + val_losses[i] + '\n')
