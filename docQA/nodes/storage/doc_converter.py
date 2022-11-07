from striprtf.striprtf import rtf_to_text
import docx2txt
import subprocess


class DocConverter:
    """
    A class to convert different docs into a .txt doc
    """
    @staticmethod
    def convert_doc(filepath: str, filepath_to_save: str, encoding: str = 'utf-8'):
        """
        A method which can convert .docx, .doc, .rtf, .txt into .txt
        :param filepath: what file should be converted
        :param filepath_to_save: where to save converted .txt file
        :param encoding: what encoding to use
        :return: filepath of the converted doc
        """
        if filepath.endswith('.docx'):
            text = docx2txt.process(filepath)
            with open(filepath_to_save.replace('.docx', '.txt'), 'w', encoding=encoding) as w:
                w.write(text)

            return filepath_to_save.replace('.docx', '.txt')

        elif filepath.endswith('.doc'):
            text = subprocess.check_output(['antiword', filepath]).decode(encoding)
            with open(filepath_to_save.replace('.doc', '.txt'), 'w', encoding=encoding) as w:
                w.write(text)

            return filepath_to_save.replace('.doc', '.txt')

        elif filepath.endswith('.rtf'):
            with open(filepath_to_save.replace('.rtf', '.txt'), 'w', encoding=encoding) as w:
                with open(filepath, 'r', encoding=encoding) as r:
                    w.write(r.read())

            return filepath_to_save.replace('.rtf', '.txt')

        elif filepath.endswith('.txt'):
            with open(filepath_to_save, 'w', encoding=encoding) as w:
                with open(filepath, 'r', encoding=encoding) as r:
                    w.write(r.read())

            return filepath_to_save
