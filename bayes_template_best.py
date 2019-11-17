import math, os, pickle, re

class Bayes_Classifier:

   def __init__(self, trainDirectory = "movies_reviews/movies_reviews"):
      '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text.'''
      self.h_good = {}  #initialize the dictionary to store good reviews
      self.h_bad = {}    #dictionary that stores bad reviews
      self.bigram_dict_good = {}
      self.bigram_dict_bad = {}
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
      if not os.path.exists('./db_bigram_bad') or not os.path.exists('./db_bigram_good'):
         self.generate_bigram_table(trainDirectory)
      else:
         self.bigram_dict_good = self.load('db_bigram_good')  # load the data base files if they exist
         self.bigram_dict_bad = self.load('db_bigram_bad')

      # sum all keys in dictionary is the total frequencies         
      self.total_good_freq = sum(self.h_good.values())  
      self.total_bad_freq = sum(self.h_bad.values())

   def generate_bigram_table(self, directory):
      for file in os.listdir(directory):
         filename = os.fsdecode(file)
         rating = filename.split('-')[1]  # extract the rating from the file name
         f = self.loadFile(os.path.join(directory, filename))
         s = self.tokenize(f)
         bigrams = self.generate_bigrams(s)
         for item in bigrams:
            item = (item[0].lower(), item[1].lower())
            if rating == '5':
               self.add_to_dict(item, self.bigram_dict_good)
            elif rating == '1':
               self.add_to_dict(item, self.bigram_dict_bad)
      self.save(self.bigram_dict_bad, './db_bigram_bad')
      self.save(self.bigram_dict_good, './db_bigram_good')

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
      uni_good, uni_bad = self.unigram_probability(sText)
      bi_good, bi_bad = self.bigram_probability(sText)
      total_good = uni_good + bi_good
      total_bad = uni_bad + bi_bad
      if total_good > total_bad:
         return 'positive'
      else:
         return 'negative'

   def bigram_probability(self, sText):
      s = self.tokenize(sText) # create a list of the words in the string
      
      # calculate the a priori probability of the two classes.  We take the log in order 
      # to avoid the underflow problem.
      prob_bad = math.log(self.num_bad / (self.num_bad + self.num_good))
      prob_good = math.log(self.num_good / (self.num_bad + self.num_good))

      bigrams = self.generate_bigrams(s)
      for bi in bigrams:
         bi = (bi[0].lower(), bi[1].lower())
         prob = self.bi_prob(bi, 5)
         prob_good += math.log(prob)
         prob = self.bi_prob(bi, 1)
         prob_bad += math.log(prob)
      return prob_good, prob_bad

   


   def unigram_probability(self, sText):
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
      
      # return both probabilities
      return prob_good, prob_bad
   
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

   def bi_prob(self, word, word_class):
      if word_class == 5:  # if it is a good review
         if word in self.bigram_dict_good:
            word_freq = self.bigram_dict_good[word] # find frequency of bigram
         else:
            word_freq = 0  # word has 0 frequency
         return (word_freq + 1) / (sum(self.bigram_dict_good.values()) + 1) # add one for smoothing and find probability
      elif word_class == 1:
         if word in self.bigram_dict_bad:
            word_freq = self.bigram_dict_bad[word]
         else:
            word_freq = 0
         return (word_freq + 1) / (sum(self.bigram_dict_bad.values()) + 1) # add one for smoothing and find probability


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
      elif predict == 'negative':
         predict = '1'
      if predict == truth:
         correct += 1 # if correctly predicted, increment correct counter
   return correct / count # return percent correctly predicted
      



if __name__ == '__main__':
   bayes = Bayes_Classifier()
   print(bayes.classify("My AI class is not boring"))

