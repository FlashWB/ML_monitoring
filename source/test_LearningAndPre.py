from sklearn import svm,metrics
from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target)) #zip 将元素打包成一个元组
for index, (image, label) in enumerate(images_and_labels[:4]):  
    # for index, item in enumerate(list1, 1) #index是索引，item是元素，enumerate()迭代 
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation = 'nearest')
    plt.title('Training:%i' % label)
# flatten the image 变平 
n_samples = len(digits.images) # 返回图片个数
data = digits.images.reshape((n_samples, -1))  # 改变友矩阵形式

# create a SVM
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# We predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()









