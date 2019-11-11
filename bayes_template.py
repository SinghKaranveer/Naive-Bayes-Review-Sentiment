import math, os, pickle, re

class Bayes_Classifier:

   def __init__(self, trainDirectory = "movies_reviews/movies_reviews"):
      '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text.'''
      self.h_good = {}
      self.h_bad = {}
      if os.path.exists('./db_good') == False or os.path.exists('./db_bad') == False:
         self.train(trainDirectory)
      else: 
         self.h_good = self.load('db_good')
         self.h_bad = self.load('db_bad')
         print(self.h_good)

   def train(self, trainDirectory):   
      '''Trains the Naive Bayes Sentiment Classifier.'''
      for file in os.listdir(trainDirectory):
         filename = os.fsdecode(file)
         rating = filename.split('-')[1]
         print(rating)
         print((os.path.join(trainDirectory, filename)))
         f = self.loadFile(os.path.join(trainDirectory, filename))
         s = self.tokenize(f)
         if rating == '5':
            for word in s:
               word = word.lower()
               if word in self.h_good:
                  self.h_good[word] += 1
               else:
                  self.h_good[word] = 1
         elif rating == '1':
            for word in s:
               word = word.lower()
               if word in self.h_bad:
                  self.h_bad[word] += 1
               else:
                  self.h_bad[word] = 1
      self.save(self.h_good, './db_good')
      self.save(self.h_bad, './db_bad')
            
   def classify(self, sText):
      '''Given a target string sText, this function returns the most likely document
      class to which the target string belongs. This function should return one of three
      strings: "positive", "negative" or "neutral".
      '''

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

if __name__ == '__main__':
   bayes = Bayes_Classifier()