from word_spoilers import word_spoilers
from pypdf import PdfReader 

class dataset_generator:
    pdfFile = ""
    input_words = []
    Spoiler = word_spoilers()

    def __init__(self,file_name) -> None:
        self.pdfFile = PdfReader(file_name) 
        for idx, page in enumerate(self.pdfFile.pages): 
            text = page.extract_text()
            self.input_words.extend(text.split())
            #if idx == 10:
            #    break
        print(f"Number of words: {len(self.input_words)}")
    
    def generete_output(self, output_file, style, spoil_prob, spoil_level):
        if style == "side_by_side":
            with open(output_file, "w") as file:
                for word in self.input_words:
                    (new_word,old_word) = self.Spoiler.spoil_with_prob(word, spoil_prob, spoil_level)    
                    file.write(f"{new_word} {old_word}\n")

        elif style == "two_files_rows":
            with open(f"{output_file}-correct", "w") as correct_file:
                with open(f"{output_file}-incorrect", "w") as incorrect_file:
                    for word in self.input_words:
                        (new_word,old_word) = self.Spoiler.spoil_with_prob(word, spoil_prob, spoil_level)    
                        correct_file.write(f"{old_word}\n")
                        incorrect_file.write(f"{new_word}\n")
        
        elif style == "promts":
            pass
