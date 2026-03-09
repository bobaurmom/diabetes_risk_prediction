import pandas as pd
import numpy as np
from sklean.model_selection import train_test_split #for splitting data trainig and testing
from sklearn.logistic import LogisticRegression #logistic regression model
from sklearn.metrics import accuracy_score, classification_report #for evaluating model performance
from sklearn.preprocessing import StandardScaler #for feature scaling
from abc import ABC, abstractmethod #abstraction class for model
