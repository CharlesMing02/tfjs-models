import numpy as np
import json
import sys

# Input: normalized arrays of the same length
def findCosineSimilarity(source_representation, test_representation):
    # a = np.matmul(np.transpose(source_representation), test_representation)
    # b = np.sum(np.multiply(source_representation, source_representation))
    # c = np.sum(np.multiply(test_representation, test_representation))
    # return (a / (np.sqrt(b) * np.sqrt(c)))

    # print(source_representation)
    # print(test_representation)
    return (np.dot(source_representation, test_representation) / (np.linalg.norm(source_representation) * np.linalg.norm(test_representation)))

# create a new one for each row
def bounding_box(points):
    # points is a list of points
    # returns a bounding box of the points
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    for x, y in points:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    return min_x, min_y, max_x, max_y

# Input: Array of keypoints
def process(arr):
    coords = [[kp['x'], kp['y']] for kp in arr]
    min_x, min_y, __, ___ = bounding_box(coords)

    flattened = []
    for x, y in coords:
        flattened.append(x - min_x)
        flattened.append(y - min_y)
    f_sum = sum(flattened)
    # idea: weight differently (elbow, wrists, ankles > head, shoulders, hips)
    return [float(i) / f_sum for i in flattened] # normalize

    

def calculate_similarity(base, comp):
    # base and comp are two arrays of the same length
    # returns a similarity score
    score = []
    print(len(base), len(comp))
    for i in range(min(len(base), len(comp))):
        b_keypoints = base[i]
        c_keypoints = comp[i]
        # hard-coding 0.5 as threshold
        if b_keypoints['score'] > 0.5 and c_keypoints['score'] > 0.5:
            b_vector = process(b_keypoints['keypoints'])
            c_vector = process(c_keypoints['keypoints'])
            score.append(findCosineSimilarity(b_vector, c_vector))
    print(score)
    return sum(score) / len(score)


def calculate_similarities(data):
    base = data[0]
    similarities = []
    for i in range(1, len(data)):
        if data[i]:
            similarity = calculate_similarity(base, data[i])
            similarities.append(similarity)
    return similarities

# run with python3 main.py same "path/to/file.json"
# OR python3 main.py sep "path/to/ref.json" "path/to/comp.json"
def main():
    path = sys.argv[2]
    with open(path) as json_file:
        data = json.load(json_file)
        
        if sys.argv[1] == "same":
            # do something
            similarities = calculate_similarities(data)
            print(similarities)
        elif sys.argv[1] == "sep":
            print("TODO")
        elif sys.argv[1] == "test":
            # should be low
            print(findCosineSimilarity([1, 0, 0], [0, 1, 0]))


if __name__ == "__main__":
    main()