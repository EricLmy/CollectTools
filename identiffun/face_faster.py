import math
from sklearn import neighbors
import os
import os.path
import pickle
# from PIL import Image, ImageDraw
import cv2
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class GenerateClass(object):
    """docstring for GenerateClass"""
    def __init__(self, path1):
        super(GenerateClass, self).__init__()
        self.path1 = path1
        self.face_data_path = os.path.join(self.path1, "identiffun/face_data")
        # self.knn_clf = get_knn_clf(os.path.join(self.path1, "identiffun/trained_knn_model1.clf"))

    def get_knn_clf(self, clf_file_path):
        # if clf:load clf   not get clf form face_data(long time add jindutiao)
        if os.path.exists(clf_file_path):
            with open(clf_file_path, 'rb') as f:
                knn_clf = pickle.load(f)
        else:
            self.data_default(self.face_data_path)
            knn_clf = self.create_KNN_classifier()

        return knn_clf

        # update clf : parmas: name and img
    def data_default(self, train_dir):
        self.X = []
        self.y = [] 
        for class_dir in os.listdir(train_dir):
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            # Loop through each training image for the current person
            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    self.X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    self.y.append(class_dir)

    def data_add(self, img, name):
        image = face_recognition.load_image_file(img)
        face_bounding_boxes = face_recognition.face_locations(image)
        if len(face_bounding_boxes) == 1:
            self.X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            self.y.append(name)

    def data_del(self, name):
        for i in range(self.y.count(name)):
            del self.y[self.y.index(name)]
            del self.X[self.y.index(name)]

    def data_rename(self, oldname, newname):
        for i in range(self.y.count(oldname)):
            self.y[self.y.index(oldname)] = newname

    # Create and train the KNN classifier
    def create_KNN_classifier(self):
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree', weights='distance')
        knn_clf.fit(self.X, self.y)

        # Save the trained KNN classifier
        with open("trained_knn_model1.clf", 'wb') as f:
            pickle.dump(knn_clf, f)

        return knn_clf

    def predict(self, X_img_path, knn_clf=None, distance_threshold=0.6):
        # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        #     raise Exception("Invalid image path: {}".format(X_img_path))

        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            with open("trained_knn_model1.clf", 'rb') as f:
                knn_clf = pickle.load(f)

        # Load image file and find face locations
        # X_img = face_recognition.load_image_file(X_img_path)
        X_face_locations = face_recognition.face_locations(X_img_path)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(X_img_path, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

    def show_prediction_labels_on_image(self, img, predictions):
        for name, (top, right, bottom, left) in predictions:
            # Draw a box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return img


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    frame = cv2.imread(img_path)
    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imwrite('ret.jpg', frame)

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)
    mygener = GenerateClass()
    print("start")
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if ret:
            # # Resize frame of video to 1/4 size for faster face recognition processing
            # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            # rgb_small_frame = small_frame[:, :, ::-1]

            predictions = mygener.predict(frame, mygener.knn_clf)

            img = mygener.show_prediction_labels_on_image(frame, predictions)

            cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()



    # # STEP 1: Train the KNN classifier and save it to disk
    # # Once the model is trained and saved, you can skip this step next time.
    # print("Training KNN classifier...")
    # classifier = train("face_data", model_save_path="trained_knn_model.clf", n_neighbors=2)
    # print("Training complete!")

    # # STEP 2: Using the trained classifier, make predictions for unknown images
    # # for image_file in os.listdir("knn_examples/test"):
    # #     full_file_path = os.path.join("knn_examples/test", image_file)

    # # print("Looking for faces in {}".format(image_file))

    #     # Find all people in the image using a trained classifier model
    #     # Note: You can pass in either a classifier file name or a classifier model instance
    # predictions = predict("jie2.jpg", model_path="trained_knn_model.clf")

    #     # Print results on the console
    # for name, (top, right, bottom, left) in predictions:
    #     print("- Found {} at ({}, {})".format(name, left, top))

    #     # Display results overlaid on an image
    # show_prediction_labels_on_image("jie2.jpg", predictions)
