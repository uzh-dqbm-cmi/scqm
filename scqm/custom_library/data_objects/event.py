class Event:
    """Generic event class"""

    def __init__(self, name: str, id: str = "no_id", date=None):
        """Instantiate specific event object

        Args:
            name (str): Name of event
            id (str, optional): Id of specific event. Defaults to "no_id".
            date (_type_, optional): Date of event. Defaults to None.
        """
        self.name = name
        self.id = id
        self.date = date
