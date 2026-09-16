# Salary Prediction with Linear Regression in PyTorch

Predicting salary from years of experience with a **simple linear regression model written from scratch in PyTorch**. The gradients and gradient-descent updates are hand-written; no `nn.Module` or `optim`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sedwna/salary-linear-regression-pytorch/blob/main/src/Simple_linear_regression.ipynb)

## Dataset

A two-column dataset (`Experience Years`, `Salary`) with 40 rows. The notebook loads it from [ybifoundation/Dataset](https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv), and a copy is in `data/Salary Data.csv`.

| Experience Years | Salary |
|-----------------:|-------:|
| 1.1 | 39343 |
| 1.2 | 42774 |
| 1.3 | 46205 |
| 1.5 | 37731 |
| 2.0 | 43525 |

## Approach

1. **Split:** 70% train / 30% test (`train_test_split(train_size=0.7)`), which gives 28 train and 12 test samples.
2. **Scaling:** separate `StandardScaler`s for X and y, fitted on the training data only.
3. **Model:** a custom `LinearRegression` class with two parameters, `theta0` (weight) and `theta1` (bias):
   - prediction: `y_hat = X * theta0 + theta1`
   - loss: Mean Squared Error
   - gradients computed by hand, parameters updated by gradient descent
4. **Training:** `model.fit(X_train, y_train, n=20, eta=0.1)`.
5. **Evaluation:** MSE on the scaled test set; predictions are inverse-transformed back to salaries.

```python
class LinearRegression():
    def linear_regression(self):
        self.y_hat = self.X * self.theta0 + self.theta1

    def calc_gradient(self):
        error = self.y_hat - self.y
        self.grad_theta0 = 2 * torch.mean(self.X * error)
        self.grad_theta1 = 2 * torch.mean(error)

    def update(self):
        self.theta0 = self.theta0 - self.eta * self.grad_theta0
        self.theta1 = self.theta1 - self.eta * self.grad_theta1
```

## Results

From the saved notebook run (the split has no fixed seed, so the numbers change between runs):

- **Training loss (scaled MSE):** `0.5696` at the first iteration, `0.0392` after 20 iterations
- **Test loss (scaled MSE):** `0.0403`
- **Learned parameters:** `theta0 = 0.9851`, `theta1 = -0.0069`

Sample test predictions after inverse scaling:

| Experience Years | Actual Salary | Predicted Salary |
|-----------------:|--------------:|-----------------:|
| 7.1 | 98273 | 93108.62 |
| 5.5 | 82200 | 77795.38 |
| 1.3 | 46205 | 37598.11 |

## Visualizations

| Experience vs Salary | Train vs Test split | Real vs Predicted |
|:---:|:---:|:---:|
| ![Experience vs Salary](photo/p1.png) | ![Train vs Test](photo/p2.png) | ![Real vs Predicted](photo/p3.png) |

## Tech Stack

Python, PyTorch, NumPy, pandas, scikit-learn (split and scaling), Matplotlib, Jupyter / Google Colab

## Project Structure

```
salary-linear-regression-pytorch/
├── data/
│   └── Salary Data.csv
├── photo/                          # plots used in this README
│   ├── p1.png
│   ├── p2.png
│   └── p3.png
├── src/
│   └── Simple_linear_regression.ipynb
└── README.md
```

## How to Run

Open the notebook in Google Colab with the badge above, or run it locally:

```bash
git clone https://github.com/sedwna/salary-linear-regression-pytorch.git
cd salary-linear-regression-pytorch
pip install torch numpy pandas matplotlib scikit-learn notebook
jupyter notebook src/Simple_linear_regression.ipynb
```

The notebook downloads the dataset from the URL above, so it needs internet access.

## Contact

Email: sajaddehqan2002@gmail.com
