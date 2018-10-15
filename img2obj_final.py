# from NeuralNetwork import NeuralNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import shutil
from collections import OrderedDict
import cv2
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter


###FashionMNIST Data Loading####
# 100 classes, each 600 images,3*32*32

CIFAR100_train = datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transforms.Compose([
        # transforms.RandomCrop(32),se
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # this should be put after the image processing
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]))
CIFAR100_test = datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]))
b = a+3

print('FashionMNIST Data is loaded!')
################################
# GPU
if torch.cuda.is_available():
    print('GPU Device Name:', torch.cuda.get_device_name(0))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("{} is Applied".format(str(device)))

################################


class Lenet(nn.Module):
    def __init__(self):

        super().__init__()  # class hereitance
        # input must be 32*32
        classes = 100
        self.channel = 3
        # output->28. input size-?input size (N,Cin,H,W), if input size is not 28, padding=2
        self.conv1 = nn.Conv2d(self.channel, 6, kernel_size=(5, 5))  # padding
        self.relu1 = nn.ReLU()
        nn.init.xavier_uniform_(self.conv1.weight)
        self.pooling1 = nn.MaxPool2d(
            kernel_size=(2, 2), stride=2)  # downsampling

        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        nn.init.xavier_uniform_(self.conv1.weight)
        self.relu2 = nn.LeakyReLU()
        nn.init.xavier_uniform_(self.conv1.weight)
        self.pooling2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.FC1 = nn.Linear(16*5*5, 120, bias=False)
        self.relu3 = nn.LeakyReLU()
        self.FC2 = nn.Linear(120, 84, bias=False)
        self.relu4 = nn.LeakyReLU()
        self.FC3 = nn.Linear(84, classes)  # 84,__->output label numbers
        # For matrices, it’s 1. For others, it’s 0.
        self.logsoftmax = nn.LogSoftmax(dim=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm1d(120)
        self.bn3 = nn.BatchNorm1d(84)
        # dropout
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv_net = nn.Sequential(OrderedDict([
            ('C1', self.conv1),
            ('Dropout1', self.dropout1),
            ('activation_map1', self.relu1),
            ('S2', self.pooling1),
            ('C3', self.conv2),
            ('activation_map2', self.relu2),
            ('S4', self.pooling2),
        ]))

        self.FC = nn.Sequential(OrderedDict([
            ('F5', self.FC1),
            # batchnorm should be after the FC. Remember to turn FC's bias t False
            ('BatchNorm', self.bn2),
            ('activation_map3', self.relu3),
            ('F6', self.FC2),
            ('BatchNorm', self.bn2),
            ('activation_map4', self.relu4),
            ('OUTPUT', self.FC3),
        ]))

    def forward(self, img):
        # print('Lenet forward')
        result = self.conv_net(img)
        # print('result',result.shape)#result torch.Size([batchsize, 120, 1, 1])
        result = x = F.dropout(result, training=self.training)
        # the size -1 is inferred from other dimensions
        result = result.view(result.size(0), -1)
        # print('result__',result.shape)#result torch.Size([batchsize, 120])
        result = self.logsoftmax(self.FC(result))

        return result


class img2obj:
    def __init__(self):
        self.pixel = 32*32*3
        ##############SGD##############
        self.mini_batch_size = 60
        self.mini_batch_size_test = 250
        self.epochs = 100
        self.learning_rate = 0.1
        ###############################
        self.model = Lenet()
        ###loading best result###
        self.model.to(device)
        # self.loss_sum = nn.MSELoss()
        self.loss_sum = nn.CrossEntropyLoss()
        # self.updateParams = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
        # self.updateParams = torch.optim.Adam(
        #     self.model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        self.updateParams = torch.optim.Adadelta(
            self.model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.updateParams, milestones=[10, 30, 40], gamma=0.1)
        # torch.optim.rm
        ###############################
        self.train_data = torch.utils.data.DataLoader(
            CIFAR100_train, batch_size=self.mini_batch_size, shuffle=True, num_workers=0)
        self.test_data = torch.utils.data.DataLoader(
            CIFAR100_test, batch_size=self.mini_batch_size_test, shuffle=True, num_workers=0)
        self.classes = ['apples', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                        'bottles', 'bowls', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'cans', 'castle',
                        'caterpillar',
                        'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'kangaroo', 'couch',
                        'crocodile', 'cups', 'crab', 'dinosaur', 'elephant', 'dolphin', 'flatfish', 'forest', 'girl',
                        'fox', 'hamster', 'house', 'computer keyboard', 'lamp', 'lawn-mower', 'leopard', 'lion',
                        'lizard',
                        'lobster', 'man', 'maple', 'motorcycle', 'mountain', 'mouse', 'mushrooms', 'oak', 'oranges',
                        'orchids', 'otter', 'palm', 'pears', 'pickup truck', 'pine', 'plain', 'plates', 'poppies',
                        'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'roses', 'sea', 'seal',
                        'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                        'sunflowers', 'sweet peppers', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                        'train', 'trout', 'tulips', 'turtle', 'wardrobe', 'whale', 'willow', 'wolf', 'woman', 'worm'
                        ]
        self.target = len(self.classes)
        print('#####Model Initialization is completed and ready for the training process.#####')
        print('\n')
        time.sleep(0.1)
        model_file = "better_img2obj_model_checkpoint.pth.tar"
        if os.path.isfile(model_file):
            print("#############Loading the pre-trained model#############")
            checkpoint = torch.load(model_file)
            self.start_epoch = checkpoint['epoch']
            self.best_accuracy = checkpoint['best_accuracy']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.updateParams.load_state_dict(checkpoint['optimizer'])
            self.training_accuracy = checkpoint['training_accuracy']
            self.validation_accuracy = checkpoint['validation_accuracy']
            self.training_loss = checkpoint['training_loss']
            self.validation_loss = checkpoint['validation_loss']
            self.time_list = checkpoint['time']
            print('\n')
            print('preivous model accuracy:', self.best_accuracy)
            print('\n')
        else:
            self.start_epoch = 0
            self.best_accuracy = 0
            self.training_accuracy = []
            self.validation_accuracy = []
            self.training_loss = []
            self.validation_loss = []
            self.time_list = []
            print('NEW model accuracy:', self.best_accuracy)

        ############visualize################
        self.log_dir = './tf_ww'
        self.writer = SummaryWriter(self.log_dir)

    def train(self):

        def save_checkpoint(state, better, file='img2obj_model_checkpoint.pth.tar'):
            torch.save(state, file)
            if better:
                shutil.copyfile(
                    file, 'better_img2obj_model_checkpoint.pth.tar')

        def training(epochs):
            # def onehot(y):
            #     onehot_target = torch.zeros(self.mini_batch_size, self.target)
            #     for i in range(self.mini_batch_size):
            #         onehot_target[i][y[i]] = 1
            #     return onehot_target
            step = 0

            self.model.train()  # initializing the training
            print("CNN training starts__epoch: {}, LR= {}".format(
                epochs, self.scheduler.get_lr()))
            training_loss = 0
            total = 0
            final_score = 0
            self.scheduler.step()
            self.loss = 0

            for batch_index, (training_set, target_train) in enumerate(self.train_data):
                # print('training size',training_set.shape)#->[250, 3, 32, 32]
                if training_set.requires_grad:
                    print('AutoGrad is ON!')
                self.updateParams.zero_grad()  # zero gradient before the backward
                # .requires_grad_() #require grad
                training_set = training_set.to(device)
                target_train = target_train.to(device).detach()
                result = self.model(training_set)
                # print('result shape',result.shape)#->[250,10]
                # print('traget shape',target.shape)#->250
                # print('test',result.requires_grad) -> require_grad=true, thus the data can be backwareed
                # assert target_loss.shape==torch.Size([self.mini_batch_size]),"target size is not correct"
                batch_loss = self.loss_sum(result, target_train)
                # wihout .item(),in gpu model, not enough memory
                training_loss += batch_loss.item()
                batch_loss.backward()
                self.updateParams.step()  # performs a parameter update based on the current gradient
                _, predict = torch.max((result), 1)  # dim=1->each row
                final_score += predict.eq(target_train).cpu(
                ).sum().type(torch.DoubleTensor).item()

            training_loss_mean = training_loss / \
                (len(self.train_data.dataset)/self.mini_batch_size)
            training_accuracy = 100*final_score/len(self.train_data.dataset)

            # self.scalar_pytorch_train_loss.add_record(step, float(training_loss_mean))
            # #weight_list = self.model.conv1.weight.view(6*3*5*5, -1)
            # #self.histogram0.add_record(step, weight_list)
            # step+=1

            # dummy_input = Variable(torch.randn(4, 3, 32, 32)).to(device)
            # torch.onnx.export(self.model, dummy_input, "pytorch_cifar10.onnx")

            print(
                "Training-epoch-{}-training_loss_mean: {:.4f}".format(epochs, training_loss_mean))
            print(
                "Training-epoch-{}-training_accuracy: {:.4f}%".format(epochs, training_accuracy))
            # self.writer.add_image('Output', vutils.make_grid(output.data, normalize=True, scale_each=True), niter)
            return (training_loss_mean, training_accuracy)

        def validation(epochs):
            def onehot_vali(y):
                onehot_target = torch.zeros(
                    self.mini_batch_size_test, self.target)
                for i in range(self.mini_batch_size_test):
                    onehot_target[i][y[i]] = 1
                return (onehot_target)
                # onehot_target:(sample size(btachsize),feature)
            self.model.eval()
            validation_loss = 0
            total = 0
            final_score = 0

            with torch.no_grad():  # temporarily set all the requires_grad flag to false

                for test_data, target_test in self.test_data:
                    # target_vali = onehot_vali(target)
                    ###
                    # target = target.type(torch.FloatTensor)
                    # target_vali = target_vali.type(torch.FloatTensor)
                    ###
                    test_data = test_data.to(device)
                    target_test = target_test.to(device)
                    # target_vali = target_vali.to(device)

                    result = self.model(test_data)
                    batch_loss = self.loss_sum(result, target_test)
                    validation_loss += batch_loss
                    _, predict = torch.max(
                        (result), 1)  # dim=1->each row
                    final_score += predict.eq(target_test).cpu(
                    ).sum().type(torch.DoubleTensor).item()

            validation_loss_mean = validation_loss / \
                (len(self.test_data.dataset)/(self.mini_batch_size_test))
            validation_accuracy = 100*final_score/len(self.test_data.dataset)

            print(
                "Validation-epoch-{}-Validation_loss_mean: {:.4f}".format(epochs, validation_loss_mean))
            print('Validation Accuracy: {:.4f}%'.format(validation_accuracy))

            self.model_accuracy_cur_epoch = validation_accuracy

            return (validation_loss_mean, validation_accuracy)

        if __name__ == "__main__":
            print("######FashionMNIST Training-Validation Starts######")
            epoch_iter = range(1, self.epochs)

            self.model_accuracy_cur_epoch = 0
            if self.start_epoch == self.epochs:
                pass
            else:
                for i in range(self.start_epoch+1, self.epochs):
                    time_begin = time.time()
                    training_result = training(i)
                    self.training_loss.append(training_result[0])
                    self.training_accuracy.append(training_result[1])
                    vali_result = validation(i)
                    self.validation_loss.append(vali_result[0])
                    self.validation_accuracy.append(vali_result[1])
                    time_end = time.time()-time_begin
                    self.time_list.append(time_end)
                    progress = float(i*100//len(epoch_iter))
                    print('Progress: {:.4f}%'.format(progress))
                    print('\n')
                    #######################################
                    niter = i
                    # tensorboard --logdir=tf_sg --port 6066
                    self.writer.add_scalars('Loss', {
                        'Training Loss': training_result[0],
                        'Validation Loss': vali_result[0]},
                        niter
                    )

                    self.writer.add_scalars('Accuracy', {
                        'Training Accuracy': training_result[1],
                        'Validation Accuracy': vali_result[1]},
                        niter
                    )

                    # self.writer.add_histogram(
                    #     'Trainging Gradient', self.gradient)

                    # print('parameter length: ',len(model_grad))

                    # for name, param in self.model.named_parameters():
                    #     print(name,param.grad)

                    self.writer.add_histogram(
                        'weight-conv1', self.model.conv1.weight)
                    self.writer.add_histogram(
                        'weight-conv2', self.model.conv2.weight)
                    self.writer.add_histogram(
                        'weight-fc1', self.model.FC1.weight)
                    self.writer.add_histogram(
                        'weight-fc2', self.model.FC2.weight)
                    self.writer.add_histogram(
                        'weight-fc3', self.model.FC3.weight)

                    self.writer.add_histogram(
                        'bias-conv1', self.model.conv1.bias)
                    self.writer.add_histogram(
                        'bias-conv2', self.model.conv2.bias)
                    # self.writer.add_histogram('bias-fc1',self.model.FC1.bias)
                    # self.writer.add_histogram('bias-fc2', self.model.FC2.bias)
                    self.writer.add_histogram('bias-fc3', self.model.FC3.bias)

                    self.writer.add_histogram(
                        'weight-conv1_grad', self.model.conv1.weight.grad)
                    self.writer.add_histogram(
                        'weight-conv2_grad', self.model.conv2.weight.grad)
                    self.writer.add_histogram(
                        'weight-fc1_grad', self.model.FC1.weight.grad)
                    self.writer.add_histogram(
                        'weight-fc2_grad', self.model.FC2.weight.grad)
                    self.writer.add_histogram(
                        'weight-fc3_grad', self.model.FC3.weight.grad)

                    self.writer.add_histogram(
                        'bias-conv1_grad', self.model.conv1.bias.grad)
                    self.writer.add_histogram(
                        'bias-conv2_grad', self.model.conv2.bias.grad)
                    # self.writer.add_histogram('bias-fc1',self.model.FC1.bias.grad)
                    # self.writer.add_histogram('bias-fc2', self.model.FC2.bias.grad)
                    self.writer.add_histogram(
                        'bias-fc3_grad', self.model.FC3.bias.grad)

                    #######################################
                    better = self.model_accuracy_cur_epoch > self.best_accuracy
                    self.best_accuracy = max(
                        self.best_accuracy, self.model_accuracy_cur_epoch)
                    if better:
                        torch.save(self.model.state_dict(), 'CNN_MODEL.pt')
                    # save_checkpoint({'epoch': i,
                    #             'best_accuracy': self.best_accuracy,
                    #             'state_dict': self.model.state_dict(),
                    #             'optimizer': self.updateParams.state_dict(),
                    #             'training_loss': self.training_loss,
                    #             'training_accuracy': self.training_accuracy,
                    #             'validation_loss': self.validation_loss,
                    #             'validation_accuracy': self.validation_accuracy,
                    #             'time': self.time_list,
                    #             }, better)
                    print('Model Updated, proceeding to next epoch, best accuracy= {}'.format(
                        self.best_accuracy))
                # save the model after training
                # torch.save(self.model.state_dict(), 'CNN_MODEL.pt')

            # ploting

            plt.figure(1)
            sns.set_style('whitegrid')
            plt.plot(epoch_iter, self.training_loss, color='red', linestyle='solid', linewidth='3.0',
                     marker='p', markerfacecolor='red', markersize='10', label='Training Loss')
            plt.plot(epoch_iter, self.validation_loss, color='green', linestyle='solid', linewidth='3.0',
                     marker='o', markerfacecolor='green', markersize='10', label='Validation Loss')
            plt.ylabel('Loss', fontsize=18)
            plt.xlabel('Epochs', fontsize=18)
            title = "img2obj Result-loss"
            plt.title(title, fontsize=12)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.show()

            plt.figure(2)
            sns.set_style('whitegrid')
            plt.plot(epoch_iter, self.training_accuracy, color='blue', linestyle='solid', linewidth='3.0',
                     marker='s', markerfacecolor='blue', markersize='10', label='training Loss')
            plt.plot(epoch_iter, self.validation_accuracy, color='green', linestyle='solid', linewidth='3.0',
                     marker='s', markerfacecolor='green', markersize='10', label='Validation Loss')
            title = "img2obj Result-accuracy"
            plt.title(title, fontsize=12)
            plt.xlabel('Epochs', fontsize=18)
            plt.title("Model Accuracy", fontsize=14)
            plt.legend(fontsize=14)
            plt.show()

            plt.figure(3)
            sns.set_style('whitegrid')
            plt.plot(epoch_iter, self.time_list, color='blue', linestyle='solid', linewidth='3.0',
                     marker='s', markerfacecolor='blue', markersize='10', label='Validation Loss')
            plt.ylabel('Time (s)', fontsize=18)
            plt.xlabel('Epochs', fontsize=18)
            plt.title("Speed", fontsize=14)
            plt.legend(fontsize=14)
            plt.show()

    def forward(self, img):
        # image size processing
        # input_image=torch.unsqueeze(img.type(torch.FloatTensor),0)
        img = img.type(torch.FloatTensor).to(device)  # load the image to GPU
        img = img[None]  # input image should be [x,3,32,32]
        img = Variable(img, requires_grad=True)

        self.model.load_state_dict(torch.load('CNN_MODEL.pt'))
        self.model.eval()
        result = self.model(img)
        value, predict = torch.max(result, 1)
        top5_prob, top5_label = torch.topk(result, 5)
        predict = predict.cpu()  # move the result back to cpu to return and print
        # top5_prob,top5_label=top5_prob.detach().cpu(),top5_label.detach().cpu()
        # print('predict',predict.numpy()[0])
        # print('proba',1e6*top5_prob.numpy()[0])
        # print('label',top5_label.numpy())
        return self.classes[predict.numpy()[0]]
        # return top5_prob,top5_label

    def view(self, filepath='example.png'):

        img = Image.open(filepath)
        img_numpy = np.array(img)
        loader = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                 std=[0.5, 0.5, 0.5])])
        img = loader(img)
        # if img.requires_grad:
        #     print('AutoGrad is On')
        # else:
        #     img=Variable(img, requires_grad=True)
        #     print('AutoGrad is turning on')
        #     if img.requires_grad:
        #         print('AutoGrad is On')
        predict = self.forward(img)
        cv2.namedWindow(predict, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(predict, 640, 480)
        cv2.imshow(predict, img_numpy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cam(self, device_id=0):

        def preprocessing(image):

            input_image = cv2.resize(
                image, (32, 32), interpolation=cv2.INTER_NEAREST)
            image_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            return image_tensor(input_image)

        # Create a VideoCapture object with 0 as the argument to start live stream.
        camera = cv2.VideoCapture(device_id)
        font = cv2.FONT_HERSHEY_SIMPLEX  # Set font for text display on video
        # Set default viewing window
        camera.set(3, 1280)
        camera.set(4, 720)

        while True:
            read, frame = camera.read()
            if read:
                input_image_tensor = preprocessing(frame)
                prediction = self.forward(input_image_tensor)
                cv2.putText(frame, prediction, (250, 50), font,
                            2, (255, 200, 100), 5, cv2.LINE_AA)
                cv2.imshow('Camera', frame)

            else:
                print('\nError is reading video frame from the webcam..Exiting..')
                break

            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord('q'):
                break


img2obj().train()
