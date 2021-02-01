class ProArgs:
    def __init__(self, data_path='../../data/',img_size=320,learning_rate = 1e-3,batch_size = 20,num_epochs=5):
        self.data_path = data_path
        self.csv = self.data_path + 'train.csv'
        self.csv_annotations = self.data_path + 'train_annotations.csv'
        self.image_path = self.data_path + 'train/'
        self.name = 'Project'
        self.cls = 'class'
        self.annotations = 'annotations'
        self.labels = 'labels'
        self.data = 'data'
        self.patients = 'PatientID'
        self.uid = 'StudyInstanceUID'
        self.swan_ganz = 'Swan Ganz Catheter Present'
        self.all_labels = ['ETT - Abnormal', 'ETT - Borderline',
                           'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
                           'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal',
                           'CVC - Borderline', 'CVC - Normal']
        self.ett = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal']

        # Data Split
        self.test_quant = 0.1
        self.val_quant = 0.1

        # Hyper-parameters
        self.img_size = img_size
        self.in_channel = 1
        self.conv_channels = 8
        self.num_classes = 4
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = 0


