import pandas as pd
import numpy as np
import math
import time
import pickle

_DEBUG = False
# _DEBUG = True

np.random.seed(42)

class RatingPredictor(object):
    def __init__(self, trainingFile):
        self.minRating = 1
        self.maxRating = 5
        self.similarity = {} 
        self.sType = 'AdjCos'
        # load similarity matrix
        # self.loadIBParas()
        self.loadTrainingFile(trainingFile)


    def loadTrainingFile(self, trainingFile, K=400):
        """ 
        load training file 
        """
        self.trainingFile = trainingFile 
        self.ratingData = pd.read_csv(self.trainingFile,  sep='\t', names=['userID', 'movieID', 'rating', 'timestamp'])
        self.numOfTrain = self.ratingData.shape[0]
        self.avgRating = self.ratingData['rating'].mean()
        self.numOfUsers = 943
        self.numOfMovies = 1682
        self.numOfFactors = K

        self.ratingMatrix = np.zeros((self.numOfUsers, self.numOfMovies))
        self.p = np.random.random((self.numOfUsers, self.numOfFactors)) / 10
        self.q = np.random.random((self.numOfMovies, self.numOfFactors)) / 10
        self.bu = np.zeros((self.numOfUsers, 1))
        self.bi = np.zeros((self.numOfMovies, 1))
        self.uAvg = np.zeros((self.numOfUsers, 1))
        self.mAvg = np.zeros((self.numOfMovies, 1))

        for index, record in self.ratingData.iterrows():
            uID = record['userID'] - 1
            mID = record['movieID'] - 1
            self.ratingMatrix[uID][mID] = record['rating']

        for uID in self.ratingData['userID'].unique():
            # self.bu[uID - 1] = self.ratingData[self.ratingData['userID'] == uID]['rating'].mean() - self.avgRating
            self.uAvg[uID - 1] = self.ratingData[self.ratingData['userID'] == uID]['rating'].mean()

        for mID in self.ratingData['movieID'].unique():
            # self.bi[mID - 1] = self.ratingData[self.ratingData['movieID'] == mID]['rating'].mean() - self.avgRating
            self.mAvg[mID - 1] = self.ratingData[self.ratingData['movieID'] == mID]['rating'].mean()


    def loadTestFile(self, testFile):
        """ 
        load test file 
        """
        self.tRatingData = pd.read_csv(testFile, sep='\t', names=['userID', 'movieID', 'rating', 'timestamp'])
        self.tRatingMatrix = np.zeros((self.numOfUsers, self.numOfMovies))
        self.numOfTests = self.tRatingData.shape[0]
        for index, record in self.tRatingData.iterrows():
            uID = record['userID'] - 1
            mID = record['movieID'] - 1
            self.tRatingMatrix[uID][mID] = record['rating']



    #####################################################################
    # Latent Factor Model
    #####################################################################
    def SGDTraining(self, nIter=200, alpha=0.01, lamda=0.01):
        """
        training RSVD model

        Args:
            nIter:  maximum number of iterator
            alpha:  learning rate
            lambda: regularizing rate
        """
        if _DEBUG:
            print('---------- SGDTraining start ----------')
        lastRMSE = 9999999
        for n in range(nIter):
            rmse = 0
            for index, record in self.ratingData.iterrows():
                uID = record['userID'] - 1
                mID = record['movieID'] - 1
                rating = record['rating']

                # Funk-SVD
                rui = self.avgRating + self.bu[uID] + self.bi[mID] + np.dot(self.p[uID], self.q[mID])

                # Baseline
                # rui = self.avgRating + self.bu[uID] + self.bi[mID]

                rui = min(self.maxRating, rui)
                rui = max(self.minRating, rui) 
                e = rating - rui

                # update bu[i] and bi[j]        regularizing terms
                self.bu[uID] += alpha * (e - lamda * self.bu[uID])
                self.bi[mID] += alpha * (e - lamda * self.bi[mID])

                # update p[i] and q[j]          regularizing terms
                temp = self.q[mID]
                self.q[mID] += alpha * (e * self.p[uID] - lamda * self.q[mID])
                self.p[uID] += alpha * (e * temp - lamda * self.p[uID])
                rmse += e * e
            rmse = math.sqrt(rmse / self.numOfTrain)
            if _DEBUG:
                print('n = ', n, '; RMSE = ', rmse)

            if lastRMSE - rmse < 0.0015:
                if _DEBUG:
                    print('!!! Convergence !!!')
                break
            lastRMSE = rmse
            alpha *= 0.9
        if _DEBUG:
            print('---------- SGDTraining end ----------')


    def predictTestFileByRSVD(self):
        """
        predict the test file using RSVD method
        """
        self.pRatingMatrix = np.zeros((self.numOfUsers, self.numOfMovies))
        self.predictCount = 0
        for index, record in self.tRatingData.iterrows():
            uID = record['userID'] - 1
            mID = record['movieID'] - 1
            rui = self.avgRating + self.bu[uID] + self.bi[mID] + np.inner(self.p[uID], self.q[mID])
            rui = min(rui, self.maxRating)
            rui = max(rui, self.minRating)
            self.pRatingMatrix[uID][mID] = rui
            self.predictCount += 1


    def predictByRSVD(self, uID, mID):
        """
        predict the rating of mID the user most likely to give

        Args:
            uID: user ID
            mID: movie ID

        Return:
            predicting rating of mID
        """
        rui = self.avgRating + self.bu[uID] + self.bi[mID] + np.inner(self.p[uID], self.q[mID])
        return rui


    def calRSVDParameters(self, alpha=0.01, lamda=0.01, K=400):
        """
        calculate all parameters for RSVD and save that

        Args:
            Args:
            nIter:  maximum number of iterator
            alpha:  learning rate
            lambda: regularizing rate
        """
        self.loadTrainingFile('ml-100k/u.data')
        self.SGDTraining()
        prefixName = 'RSVD'
        with open(prefixName + '_bu.dat', 'wb') as F1:
            pickle.dump(self.bu, F1)
        with open(prefixName + '_bi.dat', 'wb') as F2:
            pickle.dump(self.bi, F2)
        with open(prefixName + '_p.dat', 'wb') as F3:
            pickle.dump(self.p, F3)
        with open(prefixName + '_q.dat', 'wb') as F4:
            pickle.dump(self.q, F4)


    def loadSVDParas(self):
        """
        load all parameters for SVD model

        Args:
            prefixPath: the prefixPath of parameters files
        """
        # prefixName = self.trainingFile + '_' + str(self.numOfFactors)
        prefixName = 'RSVD'
        with open(prefixName + '_bu.dat', 'rb') as F1:
            self.bu = pickle.load(F1)
        with open(prefixName + '_bi.dat', 'rb') as F2:
            self.bi = pickle.load(F2)
        with open(prefixName + '_p.dat', 'rb') as F3:
            self.p = pickle.load(F3)
        with open(prefixName + '_q.dat', 'rb') as F4:
            self.q = pickle.load(F4)



    #####################################################################
    # Item-Based Collaborative Filtering
    #####################################################################
    def simCalculate(self, m1, m2):
        """
        calculate similarity between m1 and m2

        Args:
            m1: movie 1 ID
            m2: movie 2 ID

        Return:
            similarity of m1 and m2
        """
        self.similarity.setdefault(m1,{})
        self.similarity.setdefault(m2,{})
        self.similarity[m1].setdefault(m2, -1)
        self.similarity[m2].setdefault(m1, -1)
        if self.similarity[m1][m2] != -1:
            return self.similarity[m1][m2]

        usersRatedm1 = self.ratingData.loc[self.ratingData['movieID'] == m1]['userID']
        usersRatedm2 = self.ratingData.loc[self.ratingData['movieID'] == m2]['userID']
        usersRatedm1m2 = pd.Series(list(set(usersRatedm1) & set(usersRatedm2)))
        # find co-rated users list
        comUsersData = self.ratingData.loc[self.ratingData['userID'].isin(usersRatedm1m2)]

        if (len(comUsersData) == 0):
            self.similarity[m1][m2] = 1
            self.similarity[m2][m1] = 1
            return 1

        RVm1 = np.array(comUsersData.loc[comUsersData['movieID'] == m1]['rating'])
        RVm2 = np.array(comUsersData.loc[comUsersData['movieID'] == m2]['rating'])

        if self.sType == 'AdjCos':
            avgRatingVector = self.getAvgRatingVector(usersRatedm1m2)
            # Adjusted Cosine
            STDm1 = RVm1 - avgRatingVector
            STDm2 = RVm2 - avgRatingVector
        elif self.sType in ['Pearson', 'Hybrid']:
            # Pearson Correlation Coefficient or Hybrid
            STDm1 = RVm1 - np.mean(RVm1)
            STDm2 = RVm2 - np.mean(RVm2)

        den = np.sqrt(np.sum(STDm1**2)) * np.sqrt(np.sum(STDm2**2))

        if den == 0:
            if self.sType == 'Hybrid':
                avgRatingVector = self.getAvgRatingVector(usersRatedm1m2)
                # Adjusted Cosine
                STDm1 = RVm1 - avgRatingVector
                STDm2 = RVm2 - avgRatingVector
                den = np.sqrt(np.sum(STDm1**2)) * np.sqrt(np.sum(STDm2**2))
                if den == 0:
                    self.similarity[m1][m2] = 0
                    self.similarity[m2][m1] = 0
                    return 0
            else:
                self.similarity[m1][m2] = 0
                self.similarity[m2][m1] = 0
                return 0
        sim = np.sum(STDm1 * STDm2) / den
        self.similarity[m1][m2] = sim
        self.similarity[m2][m1] = sim
        return sim


    def getAvgRatingVector(self, comUsers):
        """
        get the average rating vector of co-rated users

        Args:
            co-rated users list

        Return:
            list of average rating of co-rated users
        """
        avgRatingVector = np.zeros((comUsers.size, 1))
        for idx in list(range(comUsers.size)):
            uID = comUsers[idx]
            avgRatingVector[idx] = self.uAvg[uID - 1]
        return avgRatingVector


    def predictByIB(self, uID, mID, K):
        """
        predict the rating of mID the user most likely to give

        Args:
            uID: user ID
            mID: movie ID
            K:   top-K similarity neighbours

        Return:
            predicting rating of mID
        """
        topK = {}
        for idx, record in self.ratingData.loc[self.ratingData['userID'] == uID].iterrows():
            if mID == record['movieID']:
                continue
            sim = self.simCalculate(mID, record['movieID'])
            if sim <= 0:
                continue
            elif sim > 0:
                topK.setdefault(sim, [])
                topK[sim].append(record['rating'])

        accSim = 0
        accRat = 0
        desSimlarity = list(reversed(sorted(topK.keys())))

        for sim in desSimlarity[:K]:
            accRat += sum(topK[sim]) * sim
            accSim += len(topK[sim]) * sim
        if accSim == 0:
            return self.mAvg[mID - 1][0]
        return accRat / accSim


    def predictTestFileByIB(self, K=400):
        """
        predict the test file using item-based method

        Args:
            K: top-K similarity neighbours
        """
        self.pRatingMatrix = np.zeros((self.numOfUsers, self.numOfMovies))
        self.predictCount = 0
        for index, record in self.tRatingData.iterrows():
            uID = record['userID'] - 1
            mID = record['movieID'] - 1
            rui = self.predictByIB(uID + 1, mID + 1, K)
            self.pRatingMatrix[uID][mID] = rui
            self.predictCount += 1


    def calSimilarityMatrix(self, t='AdjCos'):
        """
        calculate similarity matrix for testing adn save that

        Args:
            t: similarity measure (AdjCos, Pearson or Hybrid)
        """
        self.sType = t
        all_movies = sorted(list(self.ratingData['movieID'].unique()))
        for m1 in all_movies:
            print('m1 = ', m1)
            for m2 in all_movies:
                simi = self.simCalculate(m1, m2)
        with open('similarityMatrix_' + self.sType + '.mat', 'wb') as F:
            pickle.dump(self.similarity, F)


    def loadIBParas(self):
        """
        load all parameters for item-based model
        """
        if self.sType == 'AdjCos':
            print('open AdjCos')
            with open('similarityMatrixAdjCos.mat', 'rb') as F:
                self.similarity = pickle.load(F)
        elif self.sType == 'Pearson':
            print('open Pearson')
            with open('similarityMatrixPearson.mat', 'rb') as F:
                self.similarity = pickle.load(F)
        elif self.sType == 'Hybrid':
            print('open hybrid')
            with open('similarityMatrixHybrid.mat', 'rb') as F:
                self.similarity = pickle.load(F)


    #####################################################################
    # Cross Validation
    #####################################################################
    def crossValidationIB(self, t='AdjCos'):
        """
        cross validation for K-neighbour selection in item-based model

        Args:
            t: similarity measure (AdjCos, Pearson or Hybrid)
        """
        self.sType = t
        print('---------- CV of IB ({0})----------'.format(self.sType))
        files = ['u1', 'u2', 'u3', 'u4', 'u5']
        # files = ['ua', 'ub']
        # files = ['u1','u3', 'u5']
        test  = '.test'
        for k in list(range(5, 51, 5)):
            rmse = 0
            mae  = 0
            for file in files:
                self.loadTestFile(basePath + file + test)
                self.loadIBParas()
                self.predictTestFileByIB(K=k)
                rmse += self.RMSE()
                mae  += self.MAE()
                # print('   c_file = ', file)
                # print('   c_rmse = ', self.RMSE())
                # print('   c_mae  = ', self.MAE())
            print('SimMeasure = {0}, K = {1}'.format(self.sType, k))
            print('The Avg RMSE of CV = ', rmse / len(files))
            print('The Avg MAE  of CV = ', mae  / len(files))
            print()


    def crossValidationSVDFactor(self):
        """
        cross validation for K-factor selection in latent factor model
        """
        print('---------- CV of SVD Factor ----------')
        files = ['u1', 'u2', 'u3', 'u4', 'u5']
        # files = ['ua', 'ub']
        # files = ['u1','u3', 'u5']
        train = '.base'
        test  = '.test'
        for k in list(range(50, 501, 50)):
            rmse = 0
            mae  = 0
            t1 = time.time()
            for file in files:
                self.loadTrainingFile(basePath + file + train, K=k)
                self.SGDTraining(lamda=0.01)
                self.loadTestFile(basePath + file + test)
                self.predictTestFileByRSVD()
                rmse += self.RMSE()
                mae  += self.MAE()
                print('c_rmse = ', self.RMSE())
                print('c_mae  = ', self.MAE())
            t2 = time.time() - t1
            print('alpha = 0.01, lambda = 0.01, K = ', k)
            print('The Avg RMSE of CV = ', rmse / len(files))
            print('The Avg MAE  of CV = ', mae  / len(files))
            print('Tot Avg Time Of CV = ', t2   / len(files))
            print()


    def crossValidationSVDLambda(self):
        """
        cross validation for lambda selection in latent factor model
        """
        print('---------- CV of SVD Lambda ----------')
        # files = ['u1', 'u2', 'u3', 'u4', 'u5']
        # files = ['ua', 'ub']
        files = ['u1','u3', 'u5']
        train = '.base'
        test  = '.test'
        L = [0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01, 0.008, 0.005, 0.001, 0]
        for lamd in L:
            rmse = 0
            mae  = 0
            for file in files:
                self.loadTrainingFile(basePath + file + train, K=400)
                self.SGDTraining(lamda=lamd)
                self.loadTestFile(basePath + file + test)
                self.predictTestFileByRSVD()
                rmse += self.RMSE()
                mae  += self.MAE()
                # print('c_rmse = ', self.RMSE())
                # print('c_mae  = ', self.MAE())
            print('alpha = 0.01, K = 400, lambda = ', lamd)
            print('The Avg RMSE of CV = ', rmse / len(files))
            print('The Avg MAE  of CV = ', mae  / len(files))
            print()


    def RMSE(self):
        """
        calculate RMSE of prediction rating matrix with test rating
        """
        return np.sqrt(((self.tRatingMatrix - self.pRatingMatrix) ** 2).sum() / self.numOfTests)


    def MAE(self):
        """
        calculate MAE of prediction rating matrix with test rating
        """
        return np.abs(self.tRatingMatrix - self.pRatingMatrix).sum() / self.numOfTests


    def ratingPrediction(self):
        """
        prediction rating by given a userID, movieID and model type

        Args:
            model: prediction model (SVD, AdjCos, Pearson, Hybrid)
        """
        while(1):
            model = input("Please select one of model from [ SVD, AdjCos, Pearson, Hybrid ]\nYour Answer is ")
            if model not in ['SVD', 'AdjCos', 'Pearson', 'Hybrid']:
                print('Entry wrong, try again!')
            else:
                break

        print('Hint: User ID from 1 - 943')
        print('Hint: Movie ID from 1 - 1682')
        if model == 'SVD':
            # self.loadSVDParas()
            # self.loadTrainingFile('ml-100k/u.data')
            print('\nSGDTraining start, that may take few minutes, please wait a moment, thanks!')
            self.SGDTraining()
            print('SDGTraining finished.\n')
            while(1):
                uID = int(input('Enter user ID: '))
                mID = int(input('Enter movie ID: '))
                if uID < 1 or uID > 943 or mID < 1 or mID > 1682:
                    print('\nSomething wrong, tryagin!')
                    print('Hint: User ID from 1 - 943')
                    print('Hint: Movie ID from 1 - 1682\n')
                    continue
                rui = self.predictByRSVD(uID - 1, mID - 1)
                r = self.ratingMatrix[uID-1][mID-1]
                print('\nPredicting rating  =', rui[0])
                print('User actual rating =', r)
                print("Current model is ", model)
                print("Hint: 0 means that the user didn't rate the movie yet\n")

        elif model in ['AdjCos', 'Pearson', 'Hybrid']:
            self.sType = model
            # self.loadIBParas()
            while (1):
                uID = int(input('Enter user ID: '))
                mID = int(input('Enter movie ID: '))
                if uID < 1 or uID > 943 or mID < 1 or mID > 1682:
                    print('\nSomething wrong, tryagin!')
                    print('Hint: User ID from 1 - 943')
                    print('Hint: Movie ID from 1 - 1682\n')
                    continue
                rui = self.predictByIB(uID - 1, mID - 1, 400)
                r = self.ratingMatrix[uID-1][mID-1]
                print('\nPredicting rating  =', rui)
                print('User actual rating =', r)
                print("Current model is", model)
                print("Hint: 0 means that the user didn't rate the movie yet\n")
        else:
            print('\nSomething wrong, tryagin! \n')


if __name__ == '__main__':
    basePath = 'ml-100k/'

    predictor = RatingPredictor(basePath + 'u.data')
    print('The end of loading training file')

    predictor.ratingPrediction()

    # cross validation for parameters selection
    # predictor.calSimilarityMatrix(t='Hybrid')
    # predictor.crossValidationSVDLambda()
    # predictor.crossValidationSVDFactor()
    # predictor.crossValidationIB(t='AdjCos')
    # predictor.calRSVDParameters()

    # Latent Factor Modle
    # predictor.SGDTraining()

    # Item-Based Method
    # predictor.loadTestFile(basePath + 'u1.test')
    # predictor.predictTestFileByIB(K=400)
    # predictor.RMSE(pType=2)