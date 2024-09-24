CIFAR10_LABEL = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, 
                 "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}
DIRS = {'wrn-28-32': './net_weights/Clean/wrn-28-10-dropout0.3.pth',   
        }

sub_test_root = 'cifar10_test.pt'
sub_train_root = 'cifar10_train.pt'

cifar10_root = './data/cifar10'
adv_test_pt_root = './data/cifar10_adv_wrn28_test'
normal_test_root = './data/cifar10/test'

adv_train_pt_root = './data/cifar10_adv_wrn28_train'
normal_train_root = './data/cifar10/train'

defense_model_root = './net_weights/cifar10_adv'

