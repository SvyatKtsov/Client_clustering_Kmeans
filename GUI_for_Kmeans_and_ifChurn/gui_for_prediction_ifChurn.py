#Graphical User Interface using PyQT6

import sys, os
# add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Second_Part_of_Main_Module.Kmeans_Class import *
cluster_num = 4
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox

class GUI_ChurnPredictor(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Churn Prediction')

        layout = QVBoxLayout()

        self.num_of_callCenter_calls_label = QLabel('Введіть загальну к-ість дзвінків до кол-центру:')
        self.num_of_callCenter_calls_combobox = QComboBox()
        self.num_of_callCenter_calls_combobox.addItems([str(i) for i in range(0, 10)])
        layout.addWidget(self.num_of_callCenter_calls_label)
        layout.addWidget(self.num_of_callCenter_calls_combobox)

        self.total_minutes_label = QLabel('Введіть тривалість усіх розмов:')
        self.total_minutes_input = QLineEdit()
        layout.addWidget(self.total_minutes_label)
        layout.addWidget(self.total_minutes_input)

        self.total_charge_label = QLabel('Введіть вартість усіх дзвінків:')
        self.total_charge_input = QLineEdit()
        layout.addWidget(self.total_charge_label)
        layout.addWidget(self.total_charge_input)

        self.intern_plan_label = QLabel('Чи є в клієнта тарифний план:')
        self.intern_plan_combobox = QComboBox()
        self.intern_plan_combobox.addItems(['так', 'ні'])
        layout.addWidget(self.intern_plan_label)
        layout.addWidget(self.intern_plan_combobox)

        self.predict_button = QPushButton('Спрогнозувати')
        self.predict_button.clicked.connect(self.predict_churn)
        layout.addWidget(self.predict_button)

        self.result_label = QLabel('')
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def predict_churn(self):
        try:
            total_minutes = float(self.total_minutes_input.text())
            total_charge = float(self.total_charge_input.text())
            num_of_callCenter_calls = int(self.num_of_callCenter_calls_combobox.currentText())
            intern_plan = 1 if self.intern_plan_combobox.currentText() == 'так' else 0

            # перевірка на від'ємні значення та нульові значення
            if total_minutes <= 0 or total_charge <= 0:
                raise ValueError("Всі значення повинні бути більше нуля")

            data_point = [total_minutes, total_charge, num_of_callCenter_calls, intern_plan]

            # іде ствоерення об'єкта Kmeans і виконання прогнозування
            kmeans = Kmeans(cluster_num, centroids=last_centroids, if_loopStopCond_is_numOfIters=True,
                            if_loopStopCond_is_error_between_centroids=False, iter_num=5, needed_error=0.1)
            result = kmeans.predict_if_churn(data_point, last_centroids, Xnp)

            if "Prediction: Client has stopped using this company's services" in result:
                self.result_label.setText('Клієнт завершить користування послугами компанії.') # так
            else:
                self.result_label.setText('Клієнт не завершить користування послугами компанії.') # ні

        except ValueError:
            QMessageBox.warning(self, 'Input Error', 'Будь ласка, введіть коректні дані.')
        except Exception as e:
            QMessageBox.critical(self, f'Критична помилка:{e}')
            raise

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI_ChurnPredictor()
    ex.show()
    sys.exit(app.exec())
