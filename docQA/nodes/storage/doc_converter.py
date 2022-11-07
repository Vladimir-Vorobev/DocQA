from striprtf.striprtf import rtf_to_text
import docx2txt
import subprocess


class DocConverter:
    @staticmethod
    def convert_doc(filepath, filepath_to_save, encoding='utf-8'):
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
