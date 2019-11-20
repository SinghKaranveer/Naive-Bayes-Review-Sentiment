import math, os, pickle, re, sys

class Bayes_Classifier:

   def __init__(self, trainDirectory = "movies_reviews/"):
      '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text.'''
      self.h_good = {}  #initialize the dictionary to store good reviews
      self.h_bad = {}    #dictionary that stores bad reviews
      #self.bigram_dict_good = {}
      #self.bigram_dict_bad = {}
      self.num_good = 0 # store the number of good reviews
      self.num_bad = 0  # store the number of bad reviews
      if os.path.exists('./db_good_best') == False or os.path.exists('./db_bad_best') == False: # check to see if database files exist
         self.train(trainDirectory)
      else: 
         self.h_good = self.load('db_good_best')  # load the data base files if they exist
         self.h_bad = self.load('db_bad_best')
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
      self.total_good_freq = sum(self.h_good.values())# + sum(self.bigram_dict_good.values())
      self.total_bad_freq = sum(self.h_bad.values())# + sum(self.bigram_dict_bad.values())


   def add_to_dict(self, item, h):
      if item in h:
         h[item] += 1
      else:
         h[item] = 1

   '''
   Given a tolkenized string, will return a list of tuples of 
   bigrams
   '''
   def generate_bigrams(self, s):
      output = []
      if len(s) < 2:
         return output
      for i in range(len(s) - 1):
         output.append((s[i], s[i+1]))
      return output

   def train(self, trainDirectory):   
      '''Trains the Naive Bayes Sentiment Classifier.'''
      ignore_words = ['the', 'a']
      for file in os.listdir(trainDirectory): # loop through files in training directory
         filename = os.fsdecode(file)
         rating = filename.split('-')[1]  # extract the rating from the file name
         s = self.loadFile(os.path.join(trainDirectory, filename)) # load the file
         s = "".join(c for c in s if c not in ('.',':','-',',',';'))
         s = self.tokenize(s)
         bigrams = self.generate_bigrams(s)
         if rating == '5': 
            self.num_good += 1 # keep track of number of good ratings
            for word in s:   
               word = word.lower()  # convert to lower case for consistancy 
               if word in self.h_good:
                  self.h_good[word] += 1  # increment counter if word is in dict
               else:
                  self.h_good[word] = 1  # if word not in dict, add to dict
            for bi in bigrams:
               word = bi[0].lower(), bi[1].lower()
               if word in self.h_good:
                  self.h_good[word] += 1
               else:
                  self.h_good[word] = 1
         elif rating == '1':
            self.num_bad += 1  # keep track of number of bad ratings
            for word in s:
               word = word.lower()  
               if word in self.h_bad:
                  self.h_bad[word] += 1  # increment counter if word is in dict
               else:
                  self.h_bad[word] = 1 # if word not in dict, add to dict
            for bi in bigrams:
               word = bi[0].lower(), bi[1].lower()
               if bi in self.h_bad:
                  self.h_bad[word] += 1
               else:
                  self.h_bad[word] = 1

      self.save(self.h_good, './db_good_best')
      self.save(self.h_bad, './db_bad_best')

   def classify(self, sText):
      '''Given a target string sText, this function returns the most likely document
      class to which the target string belongs. This function should return one of three
      strings: "positive", "negative" or "neutral".
      '''
      total_bad = math.log(len(self.h_bad) / (len(self.h_bad) + len(self.h_good)))
      total_good = math.log(len(self.h_good) / (len(self.h_bad) + len(self.h_good)))
      ignore_words = ['the', 'a',]
      sText = "".join(c for c in sText if c not in ('.',':','-'))

      words = self.tokenize(sText)
      for word in words:
         if word in ignore_words:
            continue
         word = word.lower()
         p_good, p_bad = self.word_prob(word)
         total_good += p_good
         total_bad += p_bad
      bigrams = self.generate_bigrams(words)
      for word in bigrams:
         word = (word[0].lower(), word[1].lower())
         p_good, p_bad = self.word_prob(word)
         total_good += p_good
         total_bad += p_bad
      if (abs(total_good - total_bad)) < (abs(total_good * 0.001)):
         return "neutral"
      if total_good > total_bad:
         return 'positive'
      else:
         return 'negative'


   
   '''
   This function will find the probablity of a word occuring given whether it is 
   from a good review or bad review
   '''
   def word_prob(self, word):

      #if the word is not in either dictionary, ignore it
      good_words = ['love', ('love', 'it'), 'awesome', 'great', ('very', 'good'), 'amazing']
      bad_words = ["hate", ("hate", "it"), "awful", "shit"]
      good_multiplier = 1
      bad_multiplier = 1
      if word not in self.h_bad and word not in self.h_good:
         return 0, 0
      if word in good_words:
         good_multiplier = 2
      if word in bad_words:
         bad_multiplier = 2
      if word in self.h_good:
            good_freq = self.h_good[word] * good_multiplier # find frequency of word
      else:
         good_freq = 0  # word has 0 frequency
      if word in self.h_bad:
         bad_freq = self.h_bad[word] * bad_multiplier
      else:
         bad_freq = 0
      p_good = math.log((good_freq + 1) / (self.total_good_freq))
      p_bad = math.log((bad_freq + 1) / (self.total_bad_freq)) 
      return p_good, p_bad

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

   # keep track of some key values to be used in performance measurment
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
   incorrect_list = []
   count = len(files_list)
   correct = 0 # initialize counter for number of correctly predicted files
   for f in files_list:
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

   # calculate the performance measures
   recall = (true_positives) / (true_positives + false_negative)
   precision = (true_positives) / (true_positives + false_positive)
   f_measure = (2*precision*recall) / (precision + recall)
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

