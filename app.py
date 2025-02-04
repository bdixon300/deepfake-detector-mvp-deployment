import flask
import dlib
import cv2
import pafy
import pandas as pd
import os
from PIL import Image
from imutils import face_utils, resize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# 2D CNN encoder using pretrained VGG16 (input is sequence of images)
class VGGCNN(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=25088):
        """Load the pretrained vgg 16 and replace top fc layer."""
        super(VGGCNN, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        vgg = models.vgg16(pretrained=True)
         # delete the last fc layer.
        modules = list(vgg.children())[:-1]
        self.vgg = nn.Sequential(*modules)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # VGG CNN
            with torch.no_grad():
                # VGG feature extraction
                x = self.vgg(x_3d[:, t, :, :, :])
                # flatten output of conv
                x = x.view(x.size(0), -1)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class LSTM(nn.Module):
    def __init__(self, CNN_embed_dim=25088, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.5, num_classes=2):
        super(LSTM, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        # RNN hidden layers
        self.h_RNN_layers = h_RNN_layers
        # RNN hidden nodes
        self.h_RNN = h_RNN
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True, # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, 128)
        self.bn1 = nn.BatchNorm1d(self.h_FC_dim, momentum=0.01)
        self.fc2 = nn.Linear(self.h_FC_dim, 64)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.01)
        self.fc3 = nn.Linear(64, self.num_classes)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        
        x = self.bn1(self.fc1(RNN_out[:, -1, :])) # Use value at last time step in sequence
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        x = torch.sigmoid(x)


        return x

# load detection model parameters
cnn = VGGCNN()
cnn.cuda()
cnn.load_state_dict(torch.load('./model/full_data_cnnmodel_for_lstm_2.pth'), strict=False)
cnn.eval()
lstm = LSTM()
lstm.load_state_dict(torch.load('./model/extended_lstm_architecture_cnn_lstm_epoch_3.pth'))
lstm.cuda()
lstm.eval()

# Setup mouth extractor
p = "..\mouth-extraction-preprocessing\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


app = flask.Flask(__name__, template_folder='templates')

def analyse_video(filename):
    # Setup reading video
    vidcap = cv2.VideoCapture(filename)
    success = True
    frame_count = 1
    voter_tally = 0
    X = []
    print("Evaluating: {}".format(filename))
    while success and frame_count < 720:
        success,image = vidcap.read()
        if not success:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 0)
        largest_face_size = 0

        if len(faces) == 0:
            continue

        for (i, face) in enumerate(faces):
            # Make the prediction and transfom it to numpy array
            #face = face.rect
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            size = face.width() * face.height()
            if largest_face_size < size:
                largest_face_size = size

                # Mouth region uses these indices for dlib
                (i, j) = (48, 68)
                # clone the original image so we can draw on it, then
                # display the name of the face part on the image
                clone = image.copy()

                # loop over the subset of facial landmarks, drawing the
                # specific face part
                for (x, y) in shape[i:j]:
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                    roi = image[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (224,224))
        X.append(transforms.ToTensor()(roi))

        if frame_count % 20 == 0:
            X = torch.stack(X, dim=0)
            X = X.unsqueeze(0)
            outputs = lstm(cnn(X.cuda()))
            _, predicted = torch.max(outputs.data, 1)
            voter_tally += predicted.sum()
            X = []
            print("current voter talley: {}, current frame sequences processed: {}".format(voter_tally, frame_count / 20))
            current_probability = (voter_tally.item() / (frame_count / 20)) * 100
            yield str(round(current_probability, 2))
        frame_count += 1
    print(frame_count)
    print("voter talley: {}".format(voter_tally))
    if voter_tally < (frame_count / 20) / 2:
        print("fake: video: {}".format(filename))
        yield "Fake"
    else:
        #real_vids += 1
        print("real: video: {}".format(filename))
        yield "Real"

@app.route('/', methods=['GET'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

@app.route('/detect', methods=['POST'])
def detect():
    url = flask.request.json['url']
    filename = pafy.new(url).getbest().url
    return flask.Response(flask.stream_with_context(analyse_video(filename)), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run()