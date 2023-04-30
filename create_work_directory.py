import os


def create_images_directory(filepath=None):
    """
    The function that starts at the same moment as bot starts working
    and creates a folder for temporary keeping user's images
    :param filepath: The directory for folder that will be created
    :return: str, The directory of created folder
    """
    if filepath is None:
        filepath = f"{os.getcwd()}\\users_images"
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    return filepath


images_dir = create_images_directory()