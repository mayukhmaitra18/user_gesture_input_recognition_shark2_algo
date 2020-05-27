'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''

from flask import Flask, request
from flask import render_template
import time
import json
import numpy as np
import scipy
from scipy import spatial
import random
app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])

def intermediates(p1, p2, nb_points):
    return zip(np.linspace(p1[0], p2[0], nb_points + 1), np.linspace(p1[1], p2[1], nb_points+1))

class BoundingBox(object):
    """
    A 2D bounding box
    """
    def __init__(self, points):
        if len(points) == 0:
            raise ValueError("Can't compute bounding box of empty list")
        self.minx, self.miny = float("inf"), float("inf")
        self.maxx, self.maxy = float("-inf"), float("-inf")
        for x, y in points:
            # Set min coords
            if x < self.minx:
                self.minx = x
            if y < self.miny:
                self.miny = y
            # Set max coords
            if x > self.maxx:
                self.maxx = x
            elif y > self.maxy:
                self.maxy = y
    @property
    def width(self):
        return self.maxx - self.minx
    @property
    def height(self):
        return self.maxy - self.miny
    def __repr__(self):
        return "BoundingBox({}, {}, {}, {})".format(
            self.minx, self.maxx, self.miny, self.maxy)


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    sample_points_X, sample_points_Y = [], []
    # TODO: Start sampling (12 points)

    sample_points_X.append(points_X[0])
    sample_points_Y.append(points_Y[0])

    nb_points = 100 // (len(points_X) - 1)
    # print('nb_points',nb_points)

    for x, y in zip(range(1, len(points_X)), range(1, len(points_Y))):
        p1 = []
        p2 = []
        p1.append(points_X[x - 1])
        p1.append(points_Y[y - 1])
        p2.append(points_X[x])
        p2.append(points_Y[y])

        # print('p1',p1,'p2',p2)

        res = intermediates(p1, p2, nb_points)
        # print('res',res)
        for a in res:
            if len(sample_points_X) < 99:
                sample_points_X.append(a[0])
                sample_points_Y.append(a[1])

    sample_points_X.append(points_X[len(points_X) - 1])
    sample_points_Y.append(points_Y[len(points_Y) - 1])


    return sample_points_X, sample_points_Y


# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    threshold = 30
    # TODO: Do pruning (12 points)

    g_start,g_end = [],[]
    g_start.append(gesture_points_X[0])
    g_start.append(gesture_points_Y[0])
    g_end.append(gesture_points_X[len(gesture_points_X)-1])
    g_end.append(gesture_points_Y[len(gesture_points_Y)-1])

    for i , j  in zip(range(len(template_sample_points_X)),range(len(template_sample_points_Y))):
        t_start = []
        t_end = []
        t_start.append(template_sample_points_X[i][0])
        t_start.append(template_sample_points_Y[j][0])
        t_end.append(template_sample_points_X[i][len(template_sample_points_X[i])-1])
        t_end.append(template_sample_points_Y[j][len(template_sample_points_Y[j])-1])
        d1 = scipy.spatial.distance.euclidean(g_start,t_start)
        d2 = scipy.spatial.distance.euclidean(g_end,t_end)
        if d1 <= threshold and d2 <= threshold:
            valid_words.append(words[i])
            valid_template_sample_points_X.append(template_sample_points_X[i])
            valid_template_sample_points_Y.append(template_sample_points_Y[j])


    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    shape_scores = []
    # TODO: Set your own L
    L = 1
    #scaling
    coords = []
    for i,j in zip(gesture_sample_points_X,gesture_sample_points_Y):
        coords.append(tuple((i,j)))
    bbox = BoundingBox(coords)
    W = bbox.width
    H = bbox.height
    num1 = float(L)/float(max(W,H))

    for i,j in zip(range(len(gesture_sample_points_X)),range(len(gesture_sample_points_Y))):
        gesture_sample_points_X[i] = gesture_sample_points_X[i]*num1
        gesture_sample_points_Y[j] = gesture_sample_points_Y[j]*num1


    u = []
    for i, j in zip(gesture_sample_points_X, gesture_sample_points_Y):
        x = []
        x.append(i)
        x.append(j)
        u.append(x)
    t = []
    for i, j in zip(valid_template_sample_points_X, valid_template_sample_points_Y):
        q = []
        for x, y in zip(i, j):
            z = []
            z.append(x)
            z.append(y)
            q.append(z)
        t.append(q)


    # TODO: Calculate shape scores (12 points)
    for i in t:
        summation = 0
        for x,y in zip(u,i):
            #print('x',x)
            #print('y',y)
            summation += scipy.spatial.distance.euclidean(x,y)

        shape_scores.append(float(summation)/float(100))


    '''
    for i, j in zip(valid_template_sample_points_X,valid_template_sample_points_Y):
        g_pt,t_pt = [],[]
        summation = 0
        for x in range(100):
            g_pt.append(gesture_sample_points_X[x])
            g_pt.append(gesture_sample_points_Y[x])
            t_pt.append(i[x])
            t_pt.append(j[x])
            print('g_pt',g_pt)
            print('t_pt',t_pt)
            summation += scipy.spatial.distance.euclidean(g_pt,t_pt)
        shape_scores.append(float(summation)/float(100))
    '''
    return shape_scores


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    radius = 12
    location_scores = []

    u = []
    for i, j in zip(gesture_sample_points_X,gesture_sample_points_Y):
        x = []
        x.append(i)
        x.append(j)
        u.append(x)
    t = []
    for i, j in zip(valid_template_sample_points_X,valid_template_sample_points_Y):
        q = []
        for x,y in zip(i,j):
            z = []
            z.append(x)
            z.append(y)
            q.append(z)
        t.append(q)

    def D(arr1,arr2):
        D = 0
        for i in range(100):
            D += np.max(d(arr1[i],arr2) - radius,0)

    def d(i,arr2):
        D_val = []
        for j in arr2:
            dist = scipy.spatial.distance.euclidean(i,j)
            D_val.append(dist)
        return np.min(D_val)

    alpha_val = np.random.dirichlet(np.ones(len(t)) * 1000, size=1)
    alpha_val.sort()

    score_i = 0
    for i in t:
        summation = 0
        if D(u,i) == 0 and D(i,u) == 0:
            delta_val = 0
            score_i += 0
        else:
            for x,y in zip(u,i):
                summation += scipy.spatial.distance.euclidean(x,y)
                delta_val = summation
                score_i += random.choice(alpha_val)*delta_val
        location_scores.append(score_i)

    # TODO: Calculate location scores (12 points)

    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.4
    # TODO: Set your own location weight
    location_coef = 0.6
    #print('shape_score',len(shape_scores))
    for i in range(len(shape_scores)):
        #print(shape_coef * shape_scores[i] + location_coef * location_scores[i])

        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])

    #print('integ_score',integration_scores)
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = 'the'
    # TODO: Set your own range.
    n = 3
    # TODO: Get the best word (12 points)
    best_score = -99999999
    #print('valid_words', valid_words)
    #print('int_score',integration_scores)
    indexes = np.argsort(integration_scores[len(integration_scores)-1])[::-1][:n]
    #(sorted(range(len(integration_scores)), key=lambda i: integration_scores[i])[-n:])
    best_words = []

    #print('integration',integration_scores)
    #print('index',indexes)
    for i in indexes:
        i_score = integration_scores[len(integration_scores)-1][i]
        word = valid_words[i]
        probability = probabilities[word]
        score = i_score*probability
        #print('score',score)
        if(score > best_score):
            best_word = word
            best_score = score
    #print('bestword',best_word)
    return best_word


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    #gesture_points_X = [gesture_points_X]
    #gesture_points_Y = [gesture_points_Y]


    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    #print(len(gesture_sample_points_Y),len(gesture_sample_points_X))

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    #print(len(template_sample_points_X), len(template_sample_points_Y))
    #print(len(valid_template_sample_points_X), len(valid_template_sample_points_Y))

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    #print('len_integ',len(integration_scores))

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'


if __name__ == "__main__":
    app.run()
