## COMP9417 TASK9 - Rating Predicting in RecommenderSystem

The program implement four models, including `Item-based(adjusted cosine)`, `Item-based(Pearson)`, `Item-based(hybrid of adjusted cosine and Pearson)` and `LFM(RSVD)`, to predict the movie rating besed on the MovieLens 100k dataset. 

In these models, `LFM(RSVD)` has the best performance, whereas `Item-based(Pearson)` has the least performance. The performance were measured using `RMSE` and `MAE` using on 5-fold cross validation.



### Getting Started

#### Requirements

- Python3

- pandas >= 0.19.2

- numpy >= 1.12.1

  ​



#### Usage

- `pip install -r requirements.txt`

- Download and extract [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip)

  ```
  9417-ratingPrediction/
  ├── CrossValidationData
  │   ├── CV-IB-AdjCos-KSelection.txt
  │   ├── CV-IB-Hybrid-KSelection.txt
  │   ├── CV-IB-Pearson-KSelection.txt
  │   ├── CV-SVD-Kfactor.txt
  │   └── CV-SVD-Lambda.txt
  ├── README.md
  ├── RatingPredictor.py
  ├── Screenshot
  │   ├── Result-IB-AdjCos.jpeg
  │   ├── Result-IB-Hybrid.jpeg
  │   ├── Result-IB-Pearson.jpeg
  │   └── Result-LFM-RSVD.jpeg
  ├── ml-100k
  │   ├── README
  │   ├── allbut.pl
  │   ├── mku.sh
  │   ├── u.data
  │   ├── u.genre
  │   ├── u.info
  │   ├── u.item
  │   ├── u.occupation
  │   ├── u.user
  │   ├── u1.base
  │   ├── u1.test
  │   ├── u2.base
  │   ├── u2.test
  │   ├── u3.base
  │   ├── u3.test
  │   ├── u4.base
  │   ├── u4.test
  │   ├── u5.base
  │   ├── u5.test
  │   ├── ua.base
  │   ├── ua.test
  │   ├── ub.base
  │   └── ub.test
  ├── report.pdf
  └── requirements.txt
  ```

- `python RatingPredictor.py`

  You will see

  ```python
  The end of loading training file
  Please select one of model from [ SVD, AdjCos, Pearson, Hybrid ]
  Your Answer is <Enter your select in there>
  ```

  Enter your answer, like SVD. If you select SVD, it may take 10 minutes to training model.

  After training that,  you will see

  ```python
  Hint: User ID from 1 - 943
  Hint: Movie ID from 1 - 1682
  Enter user ID: <Enter userID>
  Enter movie ID: <Enter movieID>
  ```

  Enter userID = 1, movieID = 2, you will see

  ```python
  Predicting rating  = 3.28266668378
  User actual rating = 3.0
  Current model is  SVD
  Hint: 0 means that the user didn't rate the movie yet
  ```




### LICENSE

Copyright © 2017 by Changxun Fan

Under Apache license : [http://www.apache.org/licenses/](http://www.apache.org/licenses/)