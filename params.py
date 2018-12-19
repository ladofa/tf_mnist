# 각종 파라미터를 모아놓음

import argparse


parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')

parser.add_argument('--class-num', type=int, default=300)
parser.add_argument('--input-rows', type=int, default=224)
parser.add_argument('--input-cols', type=int, default=224)



#################################################
#이하 컴퓨터, 테스트마다 달라짐
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--buffer-size', type=int, default=100)
parser.add_argument('--model', type=str, default='simple')
parser.add_argument('--mode', type=str, default='train')

parser.add_argument('--keep-image', type=int, default=0)
parser.add_argument('--reuse-pool-size', type=int, default=0)
parser.add_argument('--reuse-times', type=int, default=5)


parser.add_argument('--train-json', type=str, default='d:/dataset/MNIST')
parser.add_argument('--train-image-dir', type=str, default='d:/dataset/MNIST/mnistasjpg/mini50')

parser.add_argument('--eval-json', type=str, default='d:/dataset/food')
parser.add_argument('--eval-image-dir', type=str, default='d:/dataset/food')

parser.add_argument('--aug-core-num', type=int, default=11)
parser.add_argument('--max-epoch', type=int, default=30)
parser.add_argument('--lr', type=str, default='0.001')



args = parser.parse_args()

if __name__ == '__main__':
    print(args)