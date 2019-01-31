# author: syshen
# date:   2019/01/14
# 
import sys
import dlib
import cv2
import os
import tqdm

def _init_model(model_path):
    model = dlib.shape_predictor(model_path)   
    return model

def predict_shape(im, model, scale):
    im_scale = cv2.resize(im.copy(), (0, 0), fx = scale, fy = scale)
    im_scale = cv2.cvtColor(im_scale, cv2.COLOR_BGR2RGB)
    bbox = dlib.rectangle(0, 0, im_scale.shape[1], im_scale.shape[0])
    results = model(im_scale, bbox)
    print(results, results.num_parts)
    shape = []
    for ix in range(results.num_parts):
        shape.append(results.part(ix))

    return shape

def main(model_path, files_path):
    model = _init_model(model_path)
    predict_files = os.listdir(files_path)
    for ix, file_path in enumerate(predict_files):
        im = cv2.imread(files_path + '/' + file_path)
        if im is None:
            continue
        shape = predict_shape(im, model, 0.5)
        #print(ix)
        for index, point in enumerate(shape):
            #print(type(point), point.x)
            cv2.circle(im, (point.x * 2, point.y * 2), 5, (0, 0, 255), 5)
        #cv2.namedWindow('im', 0)
        cv2.imshow('im', im)
        cv2.waitKey(0)

if __name__ == '__main__':
    model_path = './result.dat'
    files_path = face_file_path = r'E:\label'
    main(model_path, files_path)