# author: syshen
# date:   2019/01/14
# 
import sys
import dlib
import cv2
import os
from tqdm import tqdm
import argparse

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--model_path', dest='model_path',
                      help='model path',
                      default='./result.dat', type=str)
  parser.add_argument('--files_path', dest='files_path',
                    help='test img path',
                    default='./labels/', type=str)
  parser.add_argument('--save_path', dest='save_path',
                    help='save test draw img path',
                    default='./resluts/', type=str)

  args = parser.parse_args()
  return args

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

def main():
    isShow = False
    args = parse_args()
    model_path = args.model_path
    files_path = args.files_path
    save_path = args.save_path
    if not os.path.exists(save_path):
       os.mkdir(save_path)
    model = _init_model(model_path)
    predict_files = os.listdir(files_path)
    for ix, file_name in enumerate(predict_files):
        im = cv2.imread(files_path + '/' + file_name)
        if im is None:
            continue
        shape = predict_shape(im, model, 0.5)
        draw_im = im.copy()
        #print(ix)
        for index, point in enumerate(shape):
            #print(type(point), point.x)
            cv2.circle(draw_im, (point.x * 2, point.y * 2), 5, (0, 0, 255), 5)
        cv2.imwrite(save_path + '/' + file_name, draw_im)
        #cv2.namedWindow('im', 0)
        if isShow:
           cv2.imshow('im', im)
           cv2.waitKey(0)

if __name__ == '__main__':
    main()