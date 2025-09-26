import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
videos = [
    {"id": 0, "title": "finance basics", "category": "finance", "views": 200, "likes": 50},
    {"id": 1, "title": "gym workout tips", "category": "gym", "views": 150, "likes": 40},
    {"id": 2, "title": "ronaldo motivation", "category": "sports", "views": 300, "likes": 100},
    {"id": 3, "title": "stock market edit", "category": "finance", "views": 220, "likes": 70},
    {"id": 4, "title": "chest exercises", "category": "gym", "views": 180, "likes": 60},
    {"id": 5, "title": "messi hard work", "category": "sports", "views": 250, "likes": 80},
    {"id": 6, "title": "crypto explained", "category": "finance", "views": 120, "likes": 30},
    {"id": 7, "title": "gym transformation", "category": "gym", "views": 90, "likes": 20},
    {"id": 8, "title": "basketball grind", "category": "sports", "views": 140, "likes": 35},
    {"id": 9, "title": "finance success story", "category": "finance", "views": 300, "likes": 90},
    {"id": 10, "title": "back workout", "category": "gym", "views": 170, "likes": 55},
    {"id": 11, "title": "ronaldo edit", "category": "sports", "views": 280, "likes": 95},
    {"id": 12, "title": "trading tips", "category": "finance", "views": 190, "likes": 40},
    {"id": 13, "title": "arm exercises", "category": "gym", "views": 130, "likes": 25},
    {"id": 14, "title": "sports grind", "category": "sports", "views": 160, "likes": 45},
    {"id": 15, "title": "finance edit", "category": "finance", "views": 210, "likes": 60},
]
#preprocess
df_videos = pd.DataFrame(videos)
# videos_encoded.drop(["title"])

user_data = [
    {
        "id": 1,
        "username": "alice",
        "viewed_videos": [0, 1, 3, 5, 7],
        "liked": [1, 5],
        "commented": [3, 7],
        "shared": [0, 7]
    },
    {
        "id": 2,
        "username": "bob",
        "viewed_videos": [2, 4, 6, 8, 10],
        "liked": [2, 6],
        "commented": [4, 10],
        "shared": [6, 8]
    },
    {
        "id": 3,
        "username": "charlie",
        "viewed_videos": [0, 2, 4, 6, 8, 10],
        "liked": [0, 4, 8],
        "commented": [2, 6],
        "shared": [0, 10]
    },
    {
        "id": 4,
        "username": "david",
        "viewed_videos": [1, 3, 5, 7, 9, 11],
        "liked": [3, 7],
        "commented": [5, 11],
        "shared": [1, 9]
    },
    {
        "id": 5,
        "username": "emma",
        "viewed_videos": [0, 2, 4, 6, 8, 10, 12],
        "liked": [0, 6, 12],
        "commented": [2, 8],
        "shared": [4, 10]
    }
]
df_user = pd.DataFrame(user_data)

num_videos = 16 # max video id + 1
def one_hot_encode(ids, size):
    vec = np.zeros(size)
    vec[ids] = 1
    return vec
X = []
for _, row in df_user.iterrows():
    viewed = one_hot_encode(row["viewed_videos"], num_videos)
    liked = one_hot_encode(row["liked"], num_videos)
    commented = one_hot_encode(row["commented"], num_videos)
    shared = one_hot_encode(row["shared"], num_videos)

    # concatenate all
    features = np.concatenate([viewed, liked, commented, shared])
    X.append(features)
X = np.array(X)
print("Input shape:", X.shape)
print(X)

# One-hot encode category
encoder = OneHotEncoder()
Y = encoder.fit_transform(df_videos[["category"]])
print("Categories:", encoder.categories_)
print("Y shape:", Y.shape)
print(Y[:5])  # first 5 rows
Y= np.array(X)
print("Input shape:", Y.shape)
print(Y)