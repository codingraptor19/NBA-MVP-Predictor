import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from scipy.stats import norm
from operator import itemgetter
from sklearn.model_selection import cross_val_score
from scipy.stats import linregress
from sklearn import neighbors


pastMVPs_df = pd.read_csv('historical-mvps.csv')
currMVPs_df = pd.read_csv('current-mvp-ladder.csv')
allMVPs_df = pd.read_csv('combined-mvps.csv')

plt.style.use('fivethirtyeight')
teamWins, graph = plt.subplots()
color_list = []
for i, j in allMVPs_df.iterrows():
    if j['Rank'] == 1:
        color_list.append('red')
    else:
        color_list.append('green')

graph.scatter(allMVPs_df['Team Wins'], allMVPs_df['Share'], color=color_list, label="MVP winners")
teamWins.suptitle("Correlation between wins and MVP votes", weight='bold', size=18, y=1.055)
graph.set_xlabel("Team wins")
graph.set_ylabel("Vote share")

graph.plot(np.unique(allMVPs_df['Team Wins']), np.poly1d(np.polyfit(allMVPs_df['Team Wins'], allMVPs_df['Share'], 1))(np.unique(allMVPs_df['Team Wins'])))

graph.legend(loc='best', prop={'size': 12, "family": "DejaVu Sans"})

slope, intercept, r_val, p_value, std_err = linregress(allMVPs_df['Team Wins'], allMVPs_df['Share'])

rsquared_val = r_val ** 2
graph_vals = "r = " + str(round(r_val, 3)) + ", p = " + str(round(p_value, 15)) + ", r-squared = " + str(round(rsquared_val, 3))

graph.set_title(graph_vals, size=12, fontname='DejaVu Sans')

teamWins.savefig('Team Wins-Correlation.png', dpi=400, bbox_inches='tight')


#-------------------------------------------
#corr b/n points and MVP vote share
plt.style.use('fivethirtyeight')
points_perGame, graph = plt.subplots()
color_list = []
for i, j in allMVPs_df.iterrows():
    if j['Rank'] == 1:
        color_list.append('red')
    else:
        color_list.append('green')

graph.scatter(allMVPs_df['PTS'], allMVPs_df['Share'], color=color_list, label="MVP winners")
points_perGame.suptitle("Correlation between Points per Game and MVP votes", weight='bold', size=18, y=1.055)
graph.set_xlabel("Points per Game")
graph.set_ylabel("Vote share")

graph.plot(np.unique(allMVPs_df['PTS']), np.poly1d(np.polyfit(allMVPs_df['PTS'], allMVPs_df['Share'], 1))(np.unique(allMVPs_df['PTS'])))

graph.legend(loc='best', prop={'size': 12, "family": "DejaVu Sans"})

slope, intercept, r_val, p_value, std_err = linregress(allMVPs_df['PTS'], allMVPs_df['Share'])

rsquared_val = r_val ** 2
graph_vals = "r = " + str(round(r_val, 3)) + ", p = " + str(round(p_value, 15)) + ", r-squared = " + str(round(rsquared_val, 3))

graph.set_title(graph_vals, size=12, fontname='DejaVu Sans')

points_perGame.savefig('PPG-Correlation.png', dpi=400, bbox_inches='tight')

#--------------------------------------------------------------------
#VORP
plt.style.use('fivethirtyeight')
vorp, graph = plt.subplots()
color_list = []
for i, j in allMVPs_df.iterrows():
    if j['Rank'] == 1:
        color_list.append('red')
    else:
        color_list.append('green')

graph.scatter(allMVPs_df['VORP'], allMVPs_df['Share'], color=color_list, label="MVP winners")
vorp.suptitle("Correlation between VORP and MVP votes", weight='bold', size=18, y=1.055)
graph.set_xlabel("VORP")
graph.set_ylabel("Vote share")

graph.plot(np.unique(allMVPs_df['VORP']), np.poly1d(np.polyfit(allMVPs_df['VORP'], allMVPs_df['Share'], 1))(np.unique(allMVPs_df['VORP'])))

graph.legend(loc='best', prop={'size': 12, "family": "DejaVu Sans"})

slope, intercept, r_val, p_value, std_err = linregress(allMVPs_df['VORP'], allMVPs_df['Share'])

rsquared_val = r_val ** 2
graph_vals = "r = " + str(round(r_val, 3)) + ", p = " + str(round(p_value, 15)) + ", r-squared = " + str(round(rsquared_val, 3))

graph.set_title(graph_vals, size=12, fontname='DejaVu Sans')

vorp.savefig('VORP-Correlation.png', dpi=400, bbox_inches='tight')

#-------------------------------
#BPM
plt.style.use('fivethirtyeight')
bpm, graph = plt.subplots()
color_list = []
for i, j in allMVPs_df.iterrows():
    if j['Rank'] == 1:
        color_list.append('red')
    else:
        color_list.append('green')

graph.scatter(allMVPs_df['BPM'], allMVPs_df['Share'], color=color_list, label="MVP winners")
bpm.suptitle("Correlation between BPM and MVP votes", weight='bold', size=18, y=1.055)
graph.set_xlabel("BPM")
graph.set_ylabel("Vote share")
graph.plot(np.unique(allMVPs_df['BPM']), np.poly1d(np.polyfit(allMVPs_df['BPM'], allMVPs_df['Share'], 1))(np.unique(allMVPs_df['BPM'])))
graph.legend(loc='best', prop={'size': 12, "family": "DejaVu Sans"})
slope, intercept, r_val, p_value, std_err = linregress(allMVPs_df['BPM'], allMVPs_df['Share'])
rsquared_val = r_val ** 2
graph_vals = "r = " + str(round(r_val, 3)) + ", p = " + str(round(p_value, 15)) + ", r-squared = " + str(round(rsquared_val, 3))
graph.set_title(graph_vals, size=12, fontname='DejaVu Sans')
bpm.savefig('BPM-Correlation.png', dpi=400, bbox_inches='tight')


#actual work

training_set, test_set = train_test_split(pastMVPs_df, test_size=0.25, random_state=10)
xval_train = training_set[['Team Wins', 'Overall Seed', 'PTS', 'TRB', 'AST', 'FG%', 'VORP', 'WS']]
yval_train = training_set[['Share']]
xval_test = test_set[['Team Wins', 'Overall Seed', 'PTS', 'TRB', 'AST', 'FG%', 'VORP', 'WS']]
yval_test = test_set[['Share']]

#accuracy scores for the models
def accuracy(model_type, vals):
    model_type.fit(xval_train, yval_train.values.ravel())
    yPredictions = model_type.predict(xval_test)
    print("Mean squared error: %3f" % mean_squared_error(yval_test, yPredictions))
    print("R^2 score: %.3f" % r2_score(yval_test, yPredictions))
    #score = cross_val_score(model_type, xval_test, yval_test.values.ravel(), cv=3, scoring='r2')
    for i in yPredictions:
        vals.append(i)

svr = SVR(kernel='rbf', gamma=1e-4, C=100, epsilon=.1)
svrmodel = []
print("The accuracy of the SVM model is : ")
accuracy(svr, svrmodel)

randForest = RandomForestRegressor(random_state=9, n_estimators=100, criterion='mse')
randForest_model = []
print("The accuracy of the Random Forest model is : ")
accuracy(randForest, randForest_model)

knn_prediction = neighbors.KNeighborsRegressor(n_neighbors=7, weights='uniform')
knn_model = []
print("The accuracy of the KNN model is : ")
accuracy(knn_prediction, knn_model)


deepNet = MLPRegressor(solver='lbfgs', hidden_layer_sizes=100, max_iter=1000, random_state=987654321, activation='identity', learning_rate='invscaling')
deepNet_model = []
print("The accuracy of the Deep Neural Network model is :")
accuracy(deepNet, deepNet_model)

#Predictions
mvp_players = currMVPs_df.iloc[:, 1]
mvp_players_predict = currMVPs_df[['Team Wins', 'Overall Seed', 'PTS', 'TRB', 'AST', 'FG%', 'VORP', 'WS']]
print(currMVPs_df.head())


#knn predictions
knn_predictor = knn_prediction.predict(mvp_players_predict)
knn_predictor = knn_predictor.tolist()
print("KNN preidictions")
for i, j in zip(mvp_players, knn_predictor):
    print(i, j)

#sort KNN results
unsorted_knn_list = [[i, j] for i, j in zip(mvp_players, knn_predictor)]
unsorted_knn_data = [data[1] for data in unsorted_knn_list]
knn_list_sorted = sorted(unsorted_knn_list, key=itemgetter(1), reverse=True)

knn_data = [data[1] for data in knn_list_sorted]
knn_mvp_players = [data[0] for data in knn_list_sorted]
print(knn_list_sorted)

#knn plot
plt.style.use('fivethirtyeight')
knn_prediction, axe = plt.subplots()
knn_xvals = np.arange(len(knn_data))
axe.bar(knn_xvals, knn_data, width=0.7, edgecolor='white', color='lightgreen', linewidth=4, label='Predicted')
player_label = knn_mvp_players
r = axe.patches
for i, j in zip(r, player_label):
    if i.get_x() > 2:
        axe.text(i.get_x() + i.get_width() / 1.75, i.get_height() + 0.02, j, ha='center', va='bottom', rotation='vertical', color='black')
    elif i.get_x() <= 2:
        height = 0.03
        axe.text(i.get_x() + i.get_width() / 1.75, height, j, ha='center', va='bottom', rotation='vertical', color='black')

knn_prediction.suptitle("KNN MVP Share Prediction:", weight='bold', size=18, y=1.0005)
axe.set_title("NBA.com MVP ladder rank", size=14, fontname='DejaVu Sans')
axe.xaxis.set_visible(False)
axe.set_ylabel("Vote Share")

knn_prediction.text(x=-0.02, y=0.03, s = '________________________________________________', fontsize=14, color='grey', horizontalalignment='left')
knn_prediction.savefig('KNN.png', dpi=400, bbox_inches='tight')


svrPrediction = svr.predict(mvp_players_predict)
svrPrediction = svrPrediction.tolist()

for (i, j) in zip(mvp_players, svrPrediction):
    print(i, j)

#sorted svm results:
unsorted_svr_list = [[i, j] for i, j in zip(mvp_players, svrPrediction)]
unsorted_svr_data = [data[1] for data in unsorted_svr_list]
svm_list_sorted = sorted(unsorted_svr_list, key=itemgetter(1), reverse=True)

svm_data = [data[1] for data in svm_list_sorted]
svm_mvp_players = [data[0] for data in svm_list_sorted]
print(svm_list_sorted)

#svr plot
plt.style.use('fivethirtyeight')
svr, axe = plt.subplots()
svm_xvals = np.arange(len(svm_data))
axe.bar(svm_xvals, svm_data, width=0.7, edgecolor='white', color='lightgreen', linewidth=4, label='Predicted')
player_label = svm_mvp_players
r = axe.patches
for i, j in zip(r, player_label):
    if i.get_x() > 4:
        axe.text(i.get_x() + i.get_width() / 1.75, i.get_height() + 0.02, j, ha='center', va='bottom', rotation='vertical', color='black')
    elif i.get_x() <= 4:
        height = 0.03
        axe.text(i.get_x() + i.get_width() / 1.75, height, j, ha='center', va='bottom', rotation='vertical', color='black')

svr.suptitle("SVR MVP Share Prediction:", weight='bold', size=18, y=1.0005)
axe.set_title("NBA.com MVP ladder rank", size=14, fontname='DejaVu Sans')
axe.xaxis.set_visible(False)
axe.set_ylabel("Vote Share")

svr.text(x=-0.02, y=0.03, s = '________________________________________________', fontsize=14, color='grey', horizontalalignment='left')
svr.savefig('SVM.png', dpi=400, bbox_inches='tight')



#rand forest prediction:
randForest_prediction = randForest.predict(mvp_players_predict)
randForest_prediction = randForest_prediction.tolist()
print("RANDFOREST")
for (i, j) in zip(mvp_players, randForest_prediction):
    print(i, j)

#sorting rf results
unsorted_randForest_list = [[i, j] for i, j in zip(mvp_players, randForest_prediction)]
unsorted_randForest_data = [data[1] for data in unsorted_randForest_list]
randomForest_list_sorted = sorted(unsorted_randForest_list, key=itemgetter(1), reverse=True)
randForest_data = [data[1] for data in randomForest_list_sorted]
randForest_mvp_players = [data[0] for data in randomForest_list_sorted]
print(randomForest_list_sorted)

#random forest plot
plt.style.use('fivethirtyeight')
randForest, axe = plt.subplots()
randForest_xvals = np.arange(len(randForest_data))
axe.bar(randForest_xvals, randForest_data, width=0.7, edgecolor='white', color='lightgreen', linewidth=4, label='Predicted')
player_label = randForest_mvp_players
r = axe.patches
for i, j in zip(r, player_label):
    if i.get_x() > 3:
        axe.text(i.get_x() + i.get_width() / 1.75, i.get_height() + 0.02, j, ha='center', va='bottom', rotation='vertical', color='black')
    elif i.get_x() <= 3:
        height = 0.03
        axe.text(i.get_x() + i.get_width() / 1.75, height, j, ha='center', va='bottom', rotation='vertical', color='black')

randForest.suptitle("Random Forest MVP Share Prediction:", weight='bold', size=18, y=1.0005)
axe.set_title("NBA.com MVP ladder rank", size=14, fontname='DejaVu Sans')
axe.xaxis.set_visible(False)
axe.set_ylabel("Vote Share")

randForest.text(x=-0.02, y=0.03, s = '________________________________________________', fontsize=14, color='grey', horizontalalignment='left')
randForest.savefig('RandomForest.png', dpi=400, bbox_inches='tight')

#DeepNet predict
deepNetPrediction = deepNet.predict(mvp_players_predict)
deepNetPrediction = deepNetPrediction.tolist()
print("NET")
for (i, j) in zip(mvp_players, deepNetPrediction):
    print(i, j)

#sorted deep neural network results
unsorted_deepNet_list = [[i, j] for i, j in zip(mvp_players, deepNetPrediction)]
unsorted_deepNet_data = [data[1] for data in unsorted_deepNet_list]
deepNet_list_sorted = sorted(unsorted_deepNet_list, key=itemgetter(1), reverse=True)

deepNet_data = [data[1] for data in deepNet_list_sorted]
deepNet_mvp_players = [data[0] for data in deepNet_list_sorted]
print(deepNet_list_sorted)

#network plot
plt.style.use('fivethirtyeight')
deepNet, axe = plt.subplots()
deepNet_xvals = np.arange(len(deepNet_data))
axe.bar(deepNet_xvals, deepNet_data, width=0.7, edgecolor='white', color='lightgreen', linewidth=4, label='Predicted')
player_label = deepNet_mvp_players
r = axe.patches
for i, j in zip(r, player_label):
    if i.get_x() > 8:
        axe.text(i.get_x() + i.get_width() / 1.75, i.get_height() + 0.05, j, ha='center', va='bottom', rotation='vertical', color='black')
    elif i.get_x() > 7:
        axe.text(i.get_x() + i.get_width() / 1.75, i.get_height() + 0.02, j, ha='center', va='bottom', rotation='vertical', color='black')
    elif i.get_x() <= 7:
        height = 0.03
        axe.text(i.get_x() + i.get_width() / 1.75, height, j, ha='center', va='bottom', rotation='vertical', color='black')

deepNet.suptitle("Deep Neural Network MVP Share Prediction:", weight='bold', size=18, y=1.0005)
axe.set_title("NBA.com MVP ladder rank", size=14, fontname='DejaVu Sans')
axe.xaxis.set_visible(False)
axe.set_ylabel("Vote Share")

deepNet.text(x=-0.02, y=0.03, s = '________________________________________________', fontsize=14, color='grey', horizontalalignment='left')
deepNet.savefig('DeepNetFig.png', dpi=400, bbox_inches='tight')

#average of all predictions
average_modelPredictions = []
for a, b, c, d in zip(unsorted_knn_data, unsorted_svr_data, unsorted_randForest_data, unsorted_deepNet_data):
    average_modelPredictions.append((a + b + c + d) / 4)

average_pred_list = [[i, j] for i, j in zip(mvp_players, average_modelPredictions)]
average_pred_list = sorted(average_pred_list, key=itemgetter(1), reverse=True)

average_data = [i[1] for i in average_pred_list]
average_playerMVPs = [i[0] for i in average_pred_list]
print(average_pred_list)

#average model plot
x_val_avg = np.arange(len(average_data))
plt.style.use('fivethirtyeight')
average_models, avg_graph = plt.subplots()
avg_graph.bar(x_val_avg, average_data, width=0.7, edgecolor='white', color='lightgreen', linewidth=4, label='Predicted')
player_label = average_playerMVPs
r = avg_graph.patches
for i, j in zip(r, player_label):
    if i.get_x() > 5:
        avg_graph.text(i.get_x() + i.get_width() / 1.75, i.get_height() + 0.02, j, ha='center', va='bottom', rotation='vertical', color='black')
    elif i.get_x() <= 5:
        height = 0.03
        avg_graph.text(i.get_x() + i.get_width() / 1.75, height, j, ha='center', va='bottom', rotation='vertical', color='black')

average_models.suptitle("Average predicted MVP Share Prediction:", weight='bold', size=18, y=1.0005)
avg_graph.set_title("NBA.com MVP ladder rank", size=14, fontname='DejaVu Sans')
avg_graph.xaxis.set_visible(False)
avg_graph.set_ylabel("Vote Share")

average_models.text(x=-0.02, y=0.03, s='________________________________________________', fontsize=14, color='grey', horizontalalignment='left')
average_models.savefig('Average_Prediction.png', dpi=400, bbox_inches='tight')