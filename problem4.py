import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def load_iris_data(url):
    data_frame = pd.read_csv(url)
    data_frame['Species'] = data_frame['Species'].map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})
    features = data_frame.drop('Species', axis=1).values
    labels = data_frame['Species'].values
    return train_test_split(features, labels, test_size=0.2, random_state=42)

class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.hidden1 = nn.Linear(4, 10)
        self.hidden2 = nn.Linear(10, 6)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(6,3)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = F.relu(self.hidden2(x))
        return self.output(x)

def train_model(model, criterion, optimizer, train_data, epochs):
    model.train()
    losses = []
    for epoch in range(epochs):
        inputs, targets = train_data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')
    return losses

def evaluate_model(model, test_data):
    model.eval()
    with torch.no_grad():
        inputs, targets = test_data
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == targets).float().mean().item()
        print(f'Accuracy: {accuracy * 100:.2f}%')

def plot_decision_boundaries(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),np.arange(y_min, y_max, 0.01))
    Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())]))
    _, Z = torch.max(Z, 1)
    Z = Z.numpy().reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary of the Iris Classification')
    plt.show()

if __name__ == "__main__":
    url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
    X_train, X_test, y_train, y_test = load_iris_data(url)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    model = CreativeIrisModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    train_model(model, criterion, optimizer, (X_train_tensor, y_train_tensor), epochs)

    evaluate_model(model, (X_test_tensor, y_test_tensor))

    plot_decision_boundaries(model, X_test, y_test)