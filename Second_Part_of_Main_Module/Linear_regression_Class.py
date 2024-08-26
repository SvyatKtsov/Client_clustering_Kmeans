import sys, os
# add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from First_Part_of_Main_Module.Data_exploration import *

X_linreg = df_copy['total_eve_minutes']  #- скільки хвилин клієнт розмовляв увечері, згідно з тарифом
y_linreg = df_copy['total_eve_charge'] #-скільки клієнту коштували вечірні дзвінки, згідно з тарифом

# initializing weights:
k, b = 0, 0
from numpy import ndarray


class LinReg():
    def __init__(self, lr=0.2, iter_num=800, needed_error=0.001):
        self.lr = lr
        self.iter_num = iter_num
        self.needed_error = needed_error
        self.k = None
        self.b = None

    def learn_patterns(self, X ,y):  # fit
        self.k = np.zeros(len([1]))  # as many w's as the number of X's (features)
        self.b = 0
        ##
        er = 0
        for i in range(X.shape[0]):
            y_predicted = np.dot(X[i], self.k) + self.b
            er = (y[i] - y_predicted) ** 2
            if er <= self.needed_error:
                print('Training has been finished successfully...')
                break
            print(f'k:{self.k}, b:{self.b}')
            dk = 2 * X[i] * (y_predicted - y[i]) / X.shape[0]
            db = 2 * (y_predicted - y[i]) / X.shape[0]

            self.k = self.k - self.lr * dk
            self.b = self.b - self.lr * db

    def predict_user_input(self, X: ndarray) -> ndarray:
        predicted_res_patternsLearnt = 0
        s_without_intercept = 0
        for j in range(len([1])):
            s_without_intercept += self.k[j] * X  # X[:, j] y=kx
        predicted_res_patternsLearnt = s_without_intercept + self.b
        print(f'Your prediction is: {predicted_res_patternsLearnt}')
        return np.array(predicted_res_patternsLearnt)


################################################################################predicting
linreg = LinReg(lr=0.004,iter_num=1000, needed_error=0.1)
linreg.learn_patterns(X_linreg, y_linreg)
linreg.predict_user_input(np.array([100]))

plt.plot(df_copy['total_eve_minutes'], df_copy['total_eve_charge'],marker='*',color='green')
plt.xlabel('Скільки хв. клієнт розмовляв по телефону')
plt.ylabel('Скільки клієнт за це заплатив'); plt.show()

print(f".....Predicted value via substituting y=kx+b with resulted coefficients: {100*0.08296809+0.0003949} \n")


