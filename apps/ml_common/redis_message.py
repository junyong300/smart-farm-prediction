class ModelOption(object):
    model: str = None
    deviceId: int = None

    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
    
class RedisMessage:
    type: str = None
    pattern: str = None
    channel: str = None
    id: str = None
    modelOption: ModelOption = None

    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

        wdata = eval(dictionary['data'])
        data = eval(wdata['data'])
        self.id = wdata['id']
        self.modelOption = ModelOption(data)
        del self.data
    