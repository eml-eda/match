class PartitioningPattern:
    """Class that defines a MATCH pattern, with the name of the pattern itself, a function to retrieve
    the pattern object and finally a function to secure the requirements
    """
    def __init__(self,name,pattern,additional_checks):
        self.name=name
        self.pattern=pattern
        self.additional_checks=additional_checks