import matplotlib.pyplot as plt
import numpy as np

learningRate = 0.225
numItr = 20


class LVQ:
    def __init__(self, learningRate, numItr, dataset, codebook):
        self.learningRate = learningRate
        self.numItr = numItr
        self.codebook = codebook
        self.dataset = dataset

    def euclideanDistance(self, copyCodebook, copyDataset):
        distance = []
        for i in range(len(copyDataset)):
            temp = []
            for j in range(len(copyCodebook)):
                tmp = copyDataset[i] - copyCodebook[j]
                tmp = tmp[0] ** 2 + tmp[1] ** 2
                temp.append(tmp)
            distance.append(temp)
        return distance

    def priority(self, dis):
        winningClass = []
        winningPos = []
        for i in range(len(dis)):
            temp = dis[i].index(min(dis[i]))
            winningPos.append(temp)
            if temp == 0:
                winningClass.append('+')
            if temp == 1:
                winningClass.append('^')
            if temp == 2:
                winningClass.append('*')
            if temp == 3:
                winningClass.append('o')

        return winningClass, winningPos

    def target(self, copyDataset, dataset):
        targetClass = []
        for i in range(len(copyDataset)):
            for j in range(len(dataset)):
                if copyDataset[i][0] == dataset[j][0][0] and copyDataset[i][1] == dataset[j][0][1]:
                    targetClass.append(dataset[j][1][0])

        return targetClass

    def accuracy(self, targetClass, winningClass):
        correct = 0
        wrong = 0
        for i in range(len(targetClass)):
            if targetClass[i] == winningClass[i]:
                correct += 1
            if targetClass[i] != winningClass[i]:
                wrong += 1

        return (correct / (correct + wrong)) * 100

    def plotting(self, winningClass, acc, itr):
        y = 0
        for i in range(9):
            for j in range(9):
                if winningClass[y] == '+':
                    plt.plot(i, j, 'r+')

                if winningClass[y] == '^':
                    plt.plot(i, j, 'k^')

                if winningClass[y] == '*':
                    plt.plot(i, j, 'b*')

                if winningClass[y] == 'o':
                    plt.plot(i, j, 'go')
                y = y + 1
        plt.title(
            f"LVQ datapoints in 9x9 square.\nIteration: {itr + 1}\nAccuracy: {acc:.3f}%, LearningRate: {self.learningRate:.3f}",
            fontsize=9)
        plt.text(0, 0.4, 'Class 1', fontsize=7, bbox=dict(facecolor='red', alpha=0.5))
        plt.text(0, 7.4, 'Class 2', fontsize=7, bbox=dict(facecolor='k', alpha=0.5))
        plt.text(7, 0.4, 'Class 3', fontsize=7, bbox=dict(facecolor='b', alpha=0.5))
        plt.text(7, 7.4, 'Class 4', fontsize=7, bbox=dict(facecolor='g', alpha=0.5))
        plt.xlabel(f"X-axis", fontsize=10)
        plt.ylabel('Y-axis', fontsize=10)
        # plt.legend('+' = 'class1',bbox_to_anchor =(0.75, 1.15), ncol = 2)
        # if itr%5 == 0:
        plt.show()

    def updateWeight(self, copyCodebook, copyDataset):
        for itr in range(self.numItr):
            self.learningRate = self.learningRate * (1.0 - (itr / float(self.numItr)))

            dis = self.euclideanDistance(copyCodebook, copyDataset)
            winningClass, winningPos = self.priority(dis)
            targetClass = self.target(copyDataset, self.dataset)
            # print(f"targetClass: {targetClass}")

            for i in range(len(targetClass)):
                if targetClass[i] == winningClass[i]:
                    copyCodebook[winningPos[i]] = copyCodebook[winningPos[i]] + self.learningRate * (
                            copyDataset[i] - copyCodebook[winningPos[i]])
                if targetClass[i] != winningClass[i]:
                    copyCodebook[winningPos[i]] = copyCodebook[winningPos[i]] - self.learningRate * (
                            copyDataset[i] - copyCodebook[winningPos[i]])
            acc = self.accuracy(targetClass, winningClass)
            print(f"#---------------------------------------#")
            print(f"#---------------------------------------#")
            print(f"> Iteration: {itr}")
            print(f"> Accuracy: {acc}")
            print(f"> LearningRate: {self.learningRate}")
            print(f"> Updated Weight Matrix: ")
            for i in range(len(self.codebook)):
                print(f"  {copyCodebook[i]} : {self.codebook[i][1][0]}")
            # print(f"Updated Weight Matrix: \n{copyCodebook}")

            self.plotting(winningClass, acc, itr)

        return copyCodebook


classes = ['+', '^', 'o', '*']
x, y = range(0, 9), range(0, 9)
x, y = np.array(x), np.array(y)

ds = []
file = open(r"trainingPattern.txt", "r")
for i in range((9)):
    l = file.readline()
    ds.append(l)
    print(l, end='')

datasetr = ds
datasetr.reverse()

dataset11 = []
# print(len(datasetr[0]))
for i in range(len(datasetr)):
    for j in range(len(datasetr[i])):
        if datasetr[i][j] != '\n':
            dataset11.append([[j, i], [datasetr[i][j]]])

dataset11.sort(key=lambda var: var[0])
dataset = dataset11
codebook = [dataset[0], dataset[8], dataset[72], dataset[80]]

# dataset.sort(key=lambda var: var[0])
codebook.sort(key=lambda var: var[0])

for i in range(len(dataset)):
    if dataset[i][1][0] == '+':
        plt.plot(dataset[i][0][0], dataset[i][0][1], 'r+')
    if dataset[i][1][0] == 'o':
        plt.plot(dataset[i][0][0], dataset[i][0][1], 'go')
    if dataset[i][1][0] == '*':
        plt.plot(dataset[i][0][0], dataset[i][0][1], 'b*')
    if dataset[i][1][0] == '^':
        plt.plot(dataset[i][0][0], dataset[i][0][1], 'k^')
plt.title(f"LVQ datapoints in 9x9 square.\nTraining Set", fontsize=10)
plt.text(0, 0.4, 'Class 1', bbox=dict(facecolor='red', alpha=0.5))
plt.text(0, 7.4, 'Class 2', bbox=dict(facecolor='k', alpha=0.5))
plt.text(7, 0.4, 'Class 3', bbox=dict(facecolor='b', alpha=0.5))
plt.text(7, 7.4, 'Class 4', bbox=dict(facecolor='g', alpha=0.5))
plt.xlabel(f"X-axis", fontsize=10)
plt.ylabel('Y-axis', fontsize=10)
plt.show()

copyDataset = []
for i in range(len(dataset)):
    copyDataset.append(dataset[i][0])

copyCodebook = []
for i in range(len(codebook)):
    copyCodebook.append(codebook[i][0])
copyDataset = np.array(copyDataset, dtype=float)
copyCodebook = np.array(copyCodebook, dtype=float)

network = LVQ(learningRate, numItr, dataset, codebook)

uw = network.updateWeight(copyCodebook, copyDataset)

t = network.target(copyDataset, dataset)
print(t)

print()
print()
print(f"dataset = \n{dataset}")
print(dataset11)
print()
print(codebook)
