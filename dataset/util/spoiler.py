import random
from typing import Iterator

class WordSpoiler:
    def __init__(self, spoil_prob: float = 0) -> None:
        """
        Initializes the object with a spoil probability.

        Parameters:
            spoil_prob (float): The probability of the object spoiling.
        """
        self.spoil_prob = spoil_prob

    def spoil(self, word: str) -> str:
        # Make a random binary choice based on the probabilities
        result = random.choices([False, True], weights=[1 - self.spoil_prob, self.spoil_prob])[0]
        if result:
            return self.random_spoil(word)
        
        return word

    def random_spoil(self, word: str, level: str = "low") -> str:
        spoil_function = random.choice((
            self.typo,
            self.missing,
            self.extra
        ))

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

        return spoiled_word

    def typo(self, word: str) -> str:
        """
        This function randomly replaces a character in the input word with a new character.
        
        Parameters:
            word (str): The input word where a character will be replaced.
        
        Returns:
            str: The word with a character replaced.
        """
        index = random.randint(0, len(word) - 1)
        new_char = chr(random.randint(97, 122))
        word = word[:index] + new_char + word[index + 1:]

        return word

    def missing(self, word: str) -> str:
        """
        A function that generates a new string by removing a random character from the input word.
        
        Parameters:
            word (str): The input word from which a random character will be removed.
        
        Returns:
            str: The modified word with one character removed.
        """
        index = random.randint(0, len(word) - 1)
        word = word[:index] + word[index + 1:]

        return word

    def extra(self, word: str) -> str:
        """
        Inserts a random character into the given word at a random index and returns the modified word.

        Parameters:
            word (str): The word to insert the random character into.

        Returns:
            str: The word with a random character inserted at a random index.
        """
        index = random.randint(0, len(word))
        new_char = chr(random.randint(97, 122))
        word = word[:index] + new_char + word[index:]

        return word

    def spoil_words_generator(self, words: list[str], num_samples: int) -> Iterator[tuple[str, str]]:
        """
        Generate a spoiled words generator that yields a tuple of a word and its spoiled version.

        Parameters:
            words (list[str]): A list of words to choose from.
            num_samples (int): The number of samples to generate.

        Yields:
            tuple[str, str]: A tuple containing a word from the list and its spoiled version.
        """
        for _ in range(num_samples):
            word = random.choice(words)
            yield (word, self.spoil(word))
