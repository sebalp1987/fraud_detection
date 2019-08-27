import os


# File system interaction
class FSUtil:
    @staticmethod
    def remove(filenames):
        if isinstance(filenames, str):
            files = [filenames]
        else:
            files = filenames

        for file in files:
            if os.path.exists(file):
                os.remove(file)
