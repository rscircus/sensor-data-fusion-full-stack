class Measurement:
    """A measurement

    Argument:
    ---------

    time:
        In Unix epoch.

    location:
        List type of n-dim.

    type:
        Can be of type "true", "false".
    """

    def __init__(self, time, location, type="true"):
        self.time = time
        self.location = location
        self.set_type(type)

    def set_type(self, type=""):
        if type in ["true", "false"]:
            self.type = type
        else:
            raise Exception("Type can only be in {true, false}.")

    def get_location(self):
        return self.location

    def __str__(self):
        return (
            "time: "
            + str(self.time)
            + " | position: "
            + str(self.position)
            + " | type: "
            + str(self.type)
        )
