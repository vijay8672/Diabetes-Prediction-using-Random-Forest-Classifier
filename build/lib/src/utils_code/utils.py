from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path: str) -> list[str]:
    """This function will return the list of requirements."""
    
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # Strip to remove any additional whitespace
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements