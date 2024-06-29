import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class Image():

    """
    Intilizers of class
    imagie array is imagies in list for model
    y = animal corresponding to imagie array
    train_x,test_x,train_y,Test_y,val_x,val_y = is training, test, vaildation set
    label encoders is for labeling animals to 0 - 89 of 90 animals
    Compelte_model = is the trained model 
    complete_model_mobilenet = is the trained model mobilenet
    """
    def __init__(self):

        # imagies into their pixl shape
        self.image_array = []
        
        # path to data and animal imagies
        self.path = os.path.join(os.getcwd(), "Data", "animals")
        
        # Values coresponding to imagie array
        self.y = []

        # x train set
        self.train_x = None
        
        # x test set
        self.test_x = None
        
        # y train set
        self.train_y = None

        # y test set
        self.test_y = None

        # y valdiation set
        self.val_x = None

        # y valdiation set
        self.val_y = None
        
        # labeling animals to 0 - 89 of 90 animals
        self.label_encoders = None

        # trained model 
        self.complete_model = None

        # Mobile net model
        self.complete_model_mobilenet = None
    
    """
    Gets imagies of animals and get also the animal to slef.y
    """
    def read_csv_get_types(self):

        # gets values inside the directory being used all files and folders
        for name in os.listdir(self.path):

            # combine to path directory 
            item_path = os.path.join(self.path, name)

            # since dir has data nad animals when the hold.py and test file are grabbed they don't exist in DATA animals so it never goes into if
            if os.path.isdir(item_path):

                # loop through the directory of the aniaml
                # The path would be your directory to Data\Animals and name from the loop which is animal type and loop through imagies
                for image in os.listdir(item_path):

                    # Reads imagie into pixel style (pixels length,pixels width, RGB)
                    # THe pixels with is the # of pixels of imagie
                    pic = cv2.imread(os.path.join(self.path, name, image))

                    # Rises imgaies so all imagies are same 200 250 is doable but memory allocation of more than 32 GBs
                    resized_img = cv2.resize(pic, (160, 160))

                    # Makes sure imagie is RGB
                    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

                    # Apends imagie 
                    self.image_array.append(img_rgb)

                    # Append name of animal to self.y so 0 of image array corresponds to self.y 0
                    self.y.append(name)

        # Label Encoder
        lb = LabelEncoder()

        # Transform self.y into labels coresponding to each animal
        self.label_encoders = lb.fit_transform(self.y)


        # Makes animals into catagorical
        self.y = to_categorical(self.label_encoders)


        # Converts array of imagies to numpy for model
        self.image_array = np.array(self.image_array)

        # Normalizes pixel values 
        """
        I did research and most models i saw use this to help with 8 bit imagies 
        Hlep with consistency 
        and feature scaling
        """
        self.image_array = self.image_array / 255.0

        # Cnverts y label encode values to numpy array
        self.y = np.array(self.y)

    """
    Splits into train and test values and prints shape
    Reason is for model shape is same as shape being made 
    NOTE: You can change the imagie size and change model to a different resize
    """
    def split(self):

        # Split into Train and test sets
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.image_array, self.y, test_size=0.30, shuffle=True)
        self.test_x, self.val_x, self.test_y, self.val_y = train_test_split(self.test_x,  self.test_y, test_size=0.10, shuffle=True)
        
        # Print Shape
        print("Train_y shape:", self.train_y.shape)
        print("Train_x shape:", self.train_x.shape)

    """
    Making of conv2d Model for animal imagies
    """
    def model_make(self):

        """
        Model Convo 2d
        Tried several thing: 
        Adding Dropout after pooling caused model to be worse
        Tried make Conv2d filters smaller which didn't imporve 
        Tried adding another conv2d and match or go higher in filter and didn't do well

        Convo2d is for imagies for CNN
        Max pooling is to help spatial dimensions of Convo 2d or all CNN
        Flatten is used to flatten data of imagie before classfication of Imagie
        Dropout is a regularization technique that randomly sets a fraction of input units to zero during training to prevent overfitting
        Dense is the layer where each neuorn is connected in fornt and back for fully connected network

        Early stopping help with vaildation over fiting

        Activations:
        Relu = Rectified Linear Unit
        Softmax = used in output layer of probalblities 

        optimizer:
        Adam =  reliable optimizer that works well across a wide range of tasks

        loss:
        most models found online use categorical_crossentropy messures probaility distrubution

        Metrics:
        Accuracy of data
        """
        model1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(160, 160, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            #tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            #tf.keras.layers.Dropout(0.1),
            #tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            #tf.keras.layers.MaxPooling2D((2, 2)),
            #tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            #tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(90, activation='softmax')
        ])


        # Complies Model with adam optimizer and loss of crossentropy with monitoing metrics of accuracy,MAE and MSE
        model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Early stopping help with vaildation over fiting
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Summary of model
        model1.summary()

        # Fit model with training with training data and 15 epochs and batch size of 64 also shuffle the data
        model1.fit(self.train_x, self.train_y,epochs=25,batch_size=64,shuffle=True,callbacks=[early_stopping],validation_data=(self.val_x, self.val_y)) 

        # save model
        self.complete_model = model1

    """
    Making of metNEt Model for animal imagies
    """
    def model_mak_MobileNet(self):

        """
        MobileNet Model

        MobileNetV2 is for Imagies and help models achieve higher accuracy from a pretrained model
        Global pooling takes Average of all values and reduces it's values helps with overfitting
        Dropout is a regularization technique that randomly sets a fraction of input units to zero during training to prevent overfitting
        Dense is the layer where each neuorn is connected in fornt and back for fully connected network

        
        Early stopping help with vaildation over fiting

        Activations:
        Relu = Rectified Linear Unit
        Softmax = used in output layer of probalblities 

        optimizer:
        Adam =  reliable optimizer that works well across a wide range of tasks

        loss:
        most models found online use categorical_crossentropy messures probaility distrubution

        Metrics:
        Accuracy of data
        """
        model2 = tf.keras.models.Sequential([
            tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet'),
            tf.keras.layers.GlobalAveragePooling2D(),
            #tf.keras.layers.Dropout(0.2),  
            #tf.keras.layers.Dense(128, activation='relu'), 
            #tf.keras.layers.Dropout(0.2),  
            #tf.keras.layers.Dense(256, activation='relu'),  
            #tf.keras.layers.Dropout(0.2),  
            tf.keras.layers.Dense(90, activation='softmax') 
        ])

        # Complies Model with adam optimizer and loss of crossentropy with monitoing metrics of accuracy,MAE and MSE
        model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Early stopping help with vaildation over fiting
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Summary of model
        model2.summary()

        # Fit model with training with training data and 15 epochs and batch size of 64 also shuffle the data
        model2.fit(self.train_x, self.train_y,epochs=25,batch_size=64,shuffle=True,callbacks=[early_stopping],validation_data=(self.val_x, self.val_y)) 

        
        # save model
        self.complete_model_mobilenet = model2
        

    """
    Tests accuarcy of models with test set
    MObileNet and covno2d (CNN)
    """
    def testing(self):

        # Loss and accuarcy of CNN Convo2d
        test_loss, test_accuracy = self.complete_model.evaluate(self.test_x, self.test_y)
        
        # Prints CNN COnvo2d loss and accuracy
        print("Convo2d Model: loss - ",test_loss," Accuracy - ", test_accuracy)
       
        # Loss and accuracy of mobile net
        test_loss, test_accuracy = self.complete_model_mobilenet.evaluate(self.test_x, self.test_y)
        
        # Prints MobileNet loss and accuracy
        print("MobileNet Model: loss - ",test_loss," Accuracy - ", test_accuracy)

"""
Main runnning of code
"""
def main():

    p = Image()
    p.read_csv_get_types()
    p.split()
    p.model_make()
    p.model_mak_MobileNet()
    p.testing()


if __name__ == "__main__":
    main()
