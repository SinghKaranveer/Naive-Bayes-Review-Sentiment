import math, os, pickle, re, sys

class Bayes_Classifier:

   def __init__(self, trainDirectory = "movies_reviews/"):
      '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text.'''
      self.h_good = {}  #initialize the dictionary to store good reviews
      self.h_bad = {}    #dictionary that stores bad reviews
      self.num_good = 0 # store the number of good reviews
      self.num_bad = 0  # store the number of bad reviews
      if os.path.exists('./db_good') == False or os.path.exists('./db_bad') == False: # check to see if database files exist
         self.train(trainDirectory)
      else: 
         self.h_good = self.load('db_good')  # load the data base files if they exist
         self.h_bad = self.load('db_bad')
         '''
         we must loop through files to get the count of negative and positve reviews
         '''
         for file in os.listdir(trainDirectory):
            filename = os.fsdecode(file)
            rating = filename.split('-')[1] # extract the rating from the filename
            if rating == '1':   # increment appropriate counter
               self.num_bad += 1
            elif rating == '5':
               self.num_good += 1

      # sum all keys in dictionary is the total frequencies         
      self.total_good_freq = sum(self.h_good.values())  
      self.total_bad_freq = sum(self.h_bad.values())


   def train(self, trainDirectory):   
      '''Trains the Naive Bayes Sentiment Classifier.'''
      
      for file in os.listdir(trainDirectory): # loop through files in training directory
         filename = os.fsdecode(file)
         rating = filename.split('-')[1]  # extract the rating from the file name
         f = self.loadFile(os.path.join(trainDirectory, filename)) # load the file
         s = self.tokenize(f)
         if rating == '5': 
            self.num_good += 1 # keep track of number of good ratings
            for word in s:
               word = word.lower()  # convert to lower case for consistancy 
               if word in self.h_good:
                  self.h_good[word] += 1  # increment counter if word is in dict
               else:
                  self.h_good[word] = 1  # if word not in dict, add to dict
         elif rating == '1':
            self.num_bad += 1  # keep track of number of bad ratings
            for word in s:
               word = word.lower()  
               if word in self.h_bad:
                  self.h_bad[word] += 1  # increment counter if word is in dict
               else:
                  self.h_bad[word] = 1 # if word not in dict, add to dict

      # save off results in files
      self.save(self.h_good, './db_good')
      self.save(self.h_bad, './db_bad')
            
   def classify(self, sText):
      '''Given a target string sText, this function returns the most likely document
      class to which the target string belongs. This function should return one of three
      strings: "positive", "negative" or "neutral".
      '''
      s = self.tokenize(sText) # create a list of the words in the string

      # calculate the a priori probability of the two classes.  We take the log in order 
      # to avoid the underflow problem.
      prob_bad = math.log(self.num_bad / (self.num_bad + self.num_good))
      prob_good = math.log(self.num_good / (self.num_bad + self.num_good))

      for word in s: # loop through all words
         word = word.lower()  # convert to lower case for consistancy.  
         prob = self.word_prob(word, 5)  # find probability of it being in good review
         prob_good += math.log(prob) # update the probability
         prob = self.word_prob(word, 1)  # find probability of it being in bad review
         prob_bad += math.log(prob) # update the probability 
      
      # return string based on which probablity is higher
      if (abs(prob_good - prob_bad)) < (abs(prob_good * 0.005)):  # check for neutral 
         return "neutral"
      if prob_bad > prob_good:
         return "negative"
      elif prob_good > prob_bad:
         return "positive"
   
   '''
   This function will find the probablity of a word occuring given whether it is 
   from a good review or bad review
   '''
   def word_prob(self, word, word_class):
      if word_class == 5:  # if it is a good review
         if word in self.h_good:
            word_freq = self.h_good[word] # find frequency of word
         else:
            word_freq = 0  # word has 0 frequency
         return (word_freq + 1) / (self.total_good_freq + 1) # add one for smoothing and find probability
      elif word_class == 1:
         if word in self.h_bad:
            word_freq = self.h_bad[word]
         else:
            word_freq = 0
         return (word_freq + 1) / (self.total_bad_freq + 1)


   def loadFile(self, sFilename):
      '''Given a file name, return the contents of the file as a string.'''
      f = open(sFilename, "r", errors="ignore")
      sTxt = f.read()
      f.close()
      return sTxt
   
   def save(self, dObj, sFilename):
      '''Given an object and a file name, write the object to the file using pickle.'''

      f = open(sFilename, "wb")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
   def load(self, sFilename):
      '''Given a file name, load and return the object stored in the file.'''

      f = open(sFilename, "rb")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText): 
      '''Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order).'''

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)

      return lTokens

'''
This function will create an instance of the classifier and given a path to 
test files, will perform classification on the files in that directory.  
It will return the percent of files that were correctly classified.
'''
def test_classifier(path):
   #initialize some counters for calculating performance metrics
   true_positives = 0
   false_positive = 0
   false_negative = 0
   num_postive = 0
   num_negative = 0
   num_neutral = 0
   classifier = Bayes_Classifier()
   files_list = []
   for f in os.walk(path):
      files_list = f[2] # add all files into the list of files
      break
   count = 0  # initialize counter for number of files in directory
   correct = 0 # initialize counter for number of correctly predicted files
   for f in files_list:
      count += 1
      truth = f.split('-')[1] # capture the ground truth
      path_to_file = os.path.join(path, f)
      s = classifier.loadFile(path_to_file) # load file contents as string
      predict = classifier.classify(s)  # classify that string

      # check whether the predicted value is same as ground truth
      if predict == 'positive':
         predict = '5'
         num_postive += 1
      elif predict == 'negative':
         predict = '1'
         num_negative += 1
      else:
         num_neutral += 1
      if predict == truth:
         correct += 1 # if correctly predicted, increment correct counter
         if predict == '5':
            true_positives += 1
      else:
         if predict == '5' and truth == '1':
            false_positive += 1
         elif predict == '1' and truth == '5':
            false_negative += 1
   recall = (true_positives) / (true_positives + false_negative)
   precision = (true_positives) / (true_positives + false_positive)
   f_measure = (2*precision*recall) / (precision + recall)
   
   # calculate the performance measures

   print("Number of Test Cases = " + str(count))
   print("Number Positive = " + str(num_postive))
   print("Number Negative = " + str(num_negative))
   print("Number Neutral = " + str(num_neutral))
   print("Precision = " + str(precision))
   print("Recall = " + str(recall))
   print("F-Measure = " + str(f_measure))
   print("Accuracy = " + str(correct / count))


   return correct / count # return percent correctly predicted



if __name__ == '__main__':
   if len(sys.argv) != 2:
      print("Please Input Path to test directory")
      sys.exit()
   bayes = Bayes_Classifier()
   test_classifier(sys.argv[1])

