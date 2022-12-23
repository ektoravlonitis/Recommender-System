import pandas as pd
import numpy as np

data3 = pd.read_csv ('BX-Book-Ratings_NOZERO_more_than_10.csv', names=['uid','bid','rating'],sep=';', skiprows=1, encoding='latin-1')

data2 = pd.read_csv ('BX-Users_NONULL.csv', names=['uid','Location','Age'],sep=';', skiprows=1, encoding='latin-1',on_bad_lines = 'skip')

output = pd.merge(data3, data2, 
                   on='uid', 
                   how='left')

R_df = data3.pivot_table(index='uid', columns='bid', values='rating', margins=False).fillna(0)

R = R_df.to_numpy()

R2_df = output.pivot_table(index='uid', columns='bid', values='Age', margins=False).fillna(0)

R2 = R2_df.to_numpy()


#R3_df = output.pivot_table(index='uid', columns='bid', values='Location',aggfunc=lambda x: ' '.join(x), margins=False).fillna(0)

#R3 = R3_df.to_numpy()

data = pd.read_csv ('BX-Books.csv', names=['book_id','title','author','year','publisher','urls','urlm','urll'],sep=';', skiprows=1, encoding='latin-1', on_bad_lines = 'skip')                    

titles = {}
for value in data.values:
    titles[value[0]] = value[1]
    

# FIND SIMILARITIES

from sklearn.metrics.pairwise import cosine_similarity

# Find the k-most similar users for each user
#
# r is the ratings matrix
# k is the number of most similar users
#
# returns: '
#similarUsers: contains the indices of the most similar users to each user
# similarities: is the pairwise similarities, i.e. similarities between users
def findKSimilar (r1,r2, k):
    
    # similarUsers is 2-D matrix
    similarUsers=-1*np.ones((nUsers,k))
    
    similarities1=cosine_similarity(r1)
    similarities2=cosine_similarity(r2)
    #similarities3=cosine_similarity(r3)
    
    # for each user
    for i in range(0, nUsers):
        simUsersIdxs= np.argsort(similarities1[:,i])
        
        l=0
        #find its most similar users    
        for j in range(simUsersIdxs.size-2, simUsersIdxs.size-k-2,-1):
            simUsersIdxs[-k+1:]
            similarUsers[i,l]=simUsersIdxs[j]
            l=l+1
            
    return similarUsers, similarities1, similarities2
    

nUsers=R.shape[0]
nItems=R.shape[1]
nNeighbours=3

similarUsers, similarities1, similarities2=findKSimilar (R, R2, nNeighbours)

ind = list(R_df.index.values)
# Similarity between all of pairs users based on the books they have read
sim = pd.DataFrame(similarities1, columns=ind, index=ind)
# The 
simUs = pd.DataFrame(similarUsers, columns=range(1,nNeighbours+1), index=ind)

# TURN SIM FROM A DATABASE TO A CSV FILE
sim.to_csv('user-pairs-books.data', sep ='\t')
# TURN SIMUS TO A JSON FILE
simUs.to_json(r'neighbors-k-books.data', orient='index')


# RECOMMEND

# Predict for 'userId', the rating of 'itemId'. 
# A trivial implementation of a collaboarative system
#
#'r': is the ratings matrix
#'userId': is the userId, and 
#'itemID': is the item id    
#'similarUsers': contains for each user his most similar users
#'similarities': are th pairwise cosine similarities between the users
# returns the prediction.     
def predict(userId, itemId, r,similarUsers,similarities1,similarities2):

    # number of neighbours to consider
    nCols=similarUsers.shape[1]
    
    sum=0.0;
    simSum=0.0;
    for l in range(0,nCols):    
        neighbor=int(similarUsers[userId, l])
        if r[neighbor,itemId] == 0:
            continue
        #weighted sum
        sum= sum+ (r[neighbor,itemId]*similarities1[neighbor,userId] +r[neighbor,itemId]*similarities2[neighbor,userId])/2
        simSum = simSum + (similarities1[neighbor,userId]+similarities2[neighbor,userId])/2
    return  sum/simSum
    
# Insert the active user
#user = 641
user = 200
#user = 200

# EVALUATE THE RECOMMENDATION
threshold = 8
dic_predictions = {}
if user<= nUsers:
    sum = 0.0
    sum_real = 0
    sum_pred = 0
    sum_realpred = 0
    sum_real2 = 0
    sum_pred2 = 0
    tp=fn=fp=fn=0
    for z in range(0, nItems):
        try:
            prediction=predict(user,z,R, similarUsers, similarities1,similarities2)
            # Insert them in a dictionary and sort it
            dic_predictions[titles[R_df.columns[z]]] = prediction
            dic = sorted(dic_predictions.items(), key=lambda item: -item[1])
            # Calculate the actual rating
            real = R[user,z]
            squared = (prediction - real)**2
            sum = sum + squared
            # Calculate for for precision and recall:
            # For true positives:
            if prediction>=threshold and real>=threshold:
                tp=tp+1
            # For false positives:
            elif prediction>=threshold and real<threshold:
                fp=fp+1
            # For false negatives:
            elif prediction<threshold and real>=threshold:
                fn=fn+1
            sum_real = sum_real + real
            sum_pred = sum_pred + prediction
            sum_realpred = sum_realpred + real*prediction
            sum_real2 = sum_real2 + real**2
            sum_pred2 = sum_pred2 + prediction**2
        except:
            continue
    
    # Statistical measures
    pearson_r = (nItems*sum_realpred - sum_real*sum_pred)/((nItems*sum_real2 - sum_real**2)*(nItems*sum_pred2 - sum_pred**2))**(1/2)
    rmse = (sum/nItems)**(1/2)
    precision=tp/(tp+fp)
    recall = tp/(tp+fn)
    if precision !=0 and recall !=0:
        f1=2*precision*recall/(precision+recall)
    
    print('Pearson Correlation Coefficient: ', "{:.3f}".format(pearson_r))
    print ('F1 : ', "{:.3f}".format(f1))
    print ('Precision : ', "{:.3f}".format(precision))
    print ('Recall : ', recall)
    print("Root mean squared error: ", "{:.3f}".format(rmse))
    # Recommendation
    print("For user: ", user, "the 10 best reccomendations are: ", '\n', dic[0:10])
else:
    print("Not that many users")
