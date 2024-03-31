import random
import unittest

class word_spoilers:

    diacritics = [
            ('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'),
            ('ž', 'z'), ('š', 's'), ('č', 'c'), ('ř', 'r'), ('ď', 'd'), ('ť', 't'), ('ň', 'n'),
            ('ů', 'u')
    ]
    
    def spoil_with_prob(self, word, prob, spoil_level):
        print("hi")
        
        probability_of_true = prob  # Probability of True
        probability_of_false = 1 - probability_of_true  # Probability of False
        
        # Make a random binary choice based on the probabilities
        result = random.choices([True, False], weights=[probability_of_true, probability_of_false], k=1)[0]
        if result:
            
            return self.random_spoil(word, spoil_level)
        else: 
            return (word, word)

    def random_spoil(self,word,level):
        f_list = [self.remove_diacritics, self.add_diacritics, self.y_change, self.letter_switch]
        spoil_function = random.choice(f_list)

        if level=="low":
            spoiled_word = spoil_function(word)
        elif level == "mid":
            spoiled_word = spoil_function(word)
            spoiled_word = spoil_function(spoiled_word)
            spoiled_word = spoil_function(spoiled_word)
        elif level == "high":
            spoiled_word = spoil_function(word)
            spoiled_word = spoil_function(spoiled_word)
            spoiled_word = spoil_function(spoiled_word)
            spoiled_word = spoil_function(spoiled_word)
            spoiled_word = spoil_function(spoiled_word)

        return (spoiled_word, word)

    def letter_switch(self, word):
        characters = list(word)
        if len(characters) == 1:
            return word
        
        x = random.randint(1,len(characters)-1)
        y = random.randint(0,x-1)

        characters[x], characters[y] = characters[y], characters[x]
        return "".join(characters)
    
    def remove_diacritics(self, word):
        characters = list(word)
        
        # get characters that have diacritics
        characters_of_interest = []
        characters_of_interest = [(idx,char) for idx,char in enumerate(characters) if any(char == diacritic[0] for diacritic in self.diacritics)]

        if not characters_of_interest:
            return word
        
        # randomly choose one to spoil
        idx_to_spoil = random.randint(0,len(characters_of_interest)-1)
        (idx,char) = characters_of_interest[idx_to_spoil]
        
        # find oposit in diacritic table
        for tuple in self.diacritics:
            if char == tuple[0]:
                new_char = tuple[1]
                break

        # spoil character
        characters[idx] = new_char

        return "".join(characters)
    
    def add_diacritics(self,word):
        characters = list(word)
        
        # get characters that have diacritics
        characters_of_interest = []
        characters_of_interest = [(idx,char) for idx,char in enumerate(characters) if any(char == diacritic[1] for diacritic in self.diacritics)]
        
        if not characters_of_interest:
            return word
        
        # randomly choose one to spoil
        idx_to_spoil = random.randint(0,len(characters_of_interest)-1)
        (idx,char) = characters_of_interest[idx_to_spoil]
        
        # find oposit in diacritic table
        for tuple in self.diacritics:
            if char == tuple[1]:
                new_char = tuple[0]
                break

        # spoil character
        characters[idx] = new_char

        return "".join(characters)
    #def capitalize(self,word):
    #    characters = list(word)
    
    def y_change(self,word):
        characters = list(word)
        
        # get characters that have y i 
        characters_of_interest = []
        characters_of_interest = [(idx,char) for idx,char in enumerate(characters) if char == 'y' or char == 'i' ]
        
        if not characters_of_interest:
            return word
        
        # randomly choose one to spoil
        idx_to_spoil = random.randint(0,len(characters_of_interest)-1)
        (idx,char) = characters_of_interest[idx_to_spoil]
        
        # find oposit in diacritic table
        if char == 'y':
            characters[idx] = 'i'
        elif char == 'i':
            characters[idx] = 'y'

        return "".join(characters)


class TestWordSpoilers(unittest.TestCase):

    def setUp(self):
        self.spoilers = word_spoilers()

    def test_random_spoil(self):
        word = "example"
        self.assertIsInstance(self.spoilers.random_spoil(word), tuple)
    
    def test_letter_switch(self):
        word = "example"
        self.assertNotEqual(self.spoilers.letter_switch(word), word)
    
    def test_letter_switch_1c(self):
        word = "e"
        self.assertEqual(self.spoilers.letter_switch(word), "e")
    
    def test_letter_switch_2c(self):
        word = "ex"
        self.assertEqual(self.spoilers.letter_switch(word), "xe")
    
    def test_remove_diacritics(self):
        word = "příklad"
        self.assertNotEqual(self.spoilers.remove_diacritics(word), word)

    def test_add_diacritics(self):
        word = "example"
        self.assertNotEqual(self.spoilers.add_diacritics(word), word)

    def test_y_change(self):
        word = "yes"
        self.assertNotEqual(self.spoilers.y_change(word), word)

    def test_remove_diacritics_empty(self):
        word = "example"
        self.assertEqual(self.spoilers.remove_diacritics(word), word)

    def test_add_diacritics_empty(self):
        word = ""
        self.assertEqual(self.spoilers.add_diacritics(word), word)

    def test_y_change_empty(self):
        word = "example"
        self.assertEqual(self.spoilers.y_change(word), word)

if __name__ == '__main__':
    unittest.main()