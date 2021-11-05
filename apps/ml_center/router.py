import models

'''
pattern: ml.a.b.c
    a: command (train, test, ...)
    b: model type (internal, ...)
    c: sub model type (self, ...)
'''

def route(msg):
    cmd, modelType, subModelType = parseMsg(msg)
    model = models.create(modelType, subModelType)

    if cmd == 'train':
        model.train()
    elif cmd == 'predict':
        model.predict()

def parseMsg(msg):
    patternList = str(msg['pattern']).split(".")
    cmd = patternList[1]
    data = eval(eval(msg['data'])['data'])
    modelType = data['model']
    subModelType = data['subModel']
    return cmd, modelType, subModelType
