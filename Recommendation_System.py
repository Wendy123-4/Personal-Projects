import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# the MovieStore is the file containing all
# information about the movies

dataset = pd.read_csv("MovieStore.csv")

# This is assuming we have a collection of users and movies they have watched
User_profile = {"user1": ( "Wendy", "285" ), "user2": ( "Sam", "58" ), "user3": ( "David", "1452" )}

#  Movies are stored in the csv file
# As a dictionary this is how it will appear
# Movie = {id:(title, keywords)


# Using the Term frequency–Inverse document frequency
#  which is a weighing scheme on the words of the keywords


tf = TfidfVectorizer( stop_words='english')

#matrix after applying the tfidf
tfidf_matrix = tf.fit_transform(dataset['Overview'])

# Now we calculate the cosine similarities to tell which movie is more similar
# using the tfdidf matrix
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# empty dictionary to store a number of recommended movies
recommended_movies = {}

for idx, row in dataset.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], dataset['id'][i]) for i in similar_indices]

    recommended_movies[row['id']] = similar_items[1:]
    


def add_user(name, movie_id):
    User_profile.update([('user' + str(len(User_profile)), (name,movie_id))])
    print(User_profile)

def remove_user(user_id):
    del User_profile[user_id]
    print(User_profile)

def movie(id):
    return dataset.loc[dataset['id'] == id]['Overview'].tolist()[0].split('-')[0]

# Just reads the results out of the dictionary.
def recommend(movie_id, num=1):
    print('\n************Hello!***************\n')

    print("This is recommended  since you watched  " + movie(movie_id) + "...\n")
    print("--------------------------------")
    final_set = recommended_movies[movie_id][:num]
    for final in final_set:
        print("Recommended: " + movie(final[1]) + " (score:" + str(final[0]) + ")")


recommend(movie_id=10)
# add_user("EPhraime", 55)
# remove_user("user2")